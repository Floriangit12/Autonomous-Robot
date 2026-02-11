#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CreateDatasetOcc3d.py

Dataset d'occupancy implicite pour un MLP "à la Tesla" à partir de CARLA :

- Multi-LiDAR en Z-stack + 6 caméras persistantes
- Accumulation multi-poses de l'ego (fenêtre [i-window_back ; i+window_forward])
- Tous les points sont exprimés dans le repère ROBOT à T0 (frame courante)
- Dans chaque callback LiDAR :
    * on récupère les hits sémantiques
    * on génère des points EMPTY le long du rayon (avant le hit)
    * on transforme hits + empty en repère robot et on accumule
- À la fin de la frame :
    * on voxelise la zone [-16,16] x [-16,16] x [-8,8] en cubes de 0.5 m
    * voxels avec hits -> Occupied, label sémantique (Building / Road / Vehicle, etc.)
    * voxels avec uniquement des empty -> classe Empty
    * voxels sans aucune observation -> classe Unknown (occlusion / jamais vus)
    * pour chaque voxel retenu : 10-20 points aléatoires
    * on sous-échantillonne pour avoir ~[points_min_saved ; points_max_saved] points
    * ratios approx (occ / empty / unknown) contrôlés par paramètres
- Preview 3D dense optionnelle (tous les points accumulés, hits + empty)
- Profiling
"""

import carla
import numpy as np
import json
import os
import time
import cv2
from datetime import datetime
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Optional
import gc
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import traceback
import sys
from collections import defaultdict
import random
from queue import Queue
from threading import Thread
# Sentinel pour les points EMPTY rajoutés dans les callbacks LiDAR
LIDAR_EMPTY_SENTINEL = 254
LIDAR_UNKNOWN_SENTINEL = 253


# Thread pour le npz 

class AsyncWriter(Thread):
    def __init__(self, save_fn):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=50) # Buffer de 50 frames
        self.save_fn = save_fn
        self.start()

    def run(self):
        while True:
            data, frame_id, ref_pos = self.queue.get()
            self.save_fn(data, frame_id, ref_pos)
            self.queue.task_done()


# ==========================
# PROFILING
# ==========================
class PerfStats:
    def __init__(self):
        self.t_tot = defaultdict(float)
        self.count = defaultdict(int)
        self.samples = defaultdict(list)

    def add(self, label: str, dt: float):
        self.t_tot[label] += dt
        self.count[label] += 1
        self.samples[label].append(dt)

    def global_report(self):
        items = sorted(self.t_tot.items(), key=lambda x: x[1], reverse=True)
        print("\n" + "=" * 70)
        print("📊 PROFILING GLOBAL (temps total par section)")
        for label, total in items:
            n = self.count[label]
            arr = self.samples[label]
            mn = min(arr) if arr else 0.0
            mx = max(arr) if arr else 0.0
            avg = (total / n) if n else 0.0
            print(f" - {label:<28s} total={total:8.3f}s  avg={avg:7.4f}s  "
                  f"min={mn:7.4f}s  max={mx:7.4f}s  (n={n})")
        print("=" * 70 + "\n")


class SectionTimer:
    def __init__(self, stats: PerfStats, label: str, silent: bool = False):
        self.stats = stats
        self.label = label
        self.silent = silent
        self.t0 = None
        self.dt = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.dt = time.time() - self.t0
        self.stats.add(self.label, self.dt)
        if not self.silent:
            print(f"   ⌛ {self.label}: {self.dt:.4f}s")
        return False


# ==========================
# CLASSES D'OCCUPATION
# ==========================
CARLA_22 = [
    (1,  "Road",          (128, 64, 128)),
    (2,  "SideWalk",      (244, 35, 232)),
    (3,  "Building",      (70, 70, 70)),
    (4,  "Wall",          (102, 102, 156)),
    (5,  "Fence",         (190, 153, 153)),
    (6,  "Pole",          (153, 153, 153)),
    (7,  "TrafficLight",  (250, 170, 30)),
    (8,  "TrafficSign",   (220, 220, 0)),
    (9,  "Vegetation",    (107, 142, 35)),
    (10, "Terrain",       (152, 251, 152)),
    (11, "Sky",           (70, 130, 180)),
    (12, "Pedestrian",    (220, 20, 60)),
    (13, "Rider",         (255, 0, 0)),
    (14, "Car",           (0, 0, 142)),
    (15, "Truck",         (0, 0, 70)),
    (16, "Bus",           (0, 60, 100)),
    (17, "Train",         (0, 80, 100)),
    (18, "Motorcycle",    (0, 0, 230)),
    (19, "Bicycle",       (119, 11, 32)),
    (20, "Static",        (110, 190, 160)),
    (21, "Dynamic",       (170, 120, 50)),
    (22, "Other",         (55, 90, 80)),
    (23, "Water",         (45, 60, 150)),
    (24, "RoadLine",      (157, 234, 50)),
    (25, "Ground",        (81, 0, 81)),
    (26, "Bridge",        (150, 100, 100)),
    (27, "RailTrack",     (230, 150, 140)),
    (28, "GuardRail",     (180, 165, 180)),

    # classe empty et unknow 
    (253, "Unknown",     (128, 128, 128)),
    (254, "Empty",       (255, 255, 255))
]


@dataclass
class VoxelConfig:
    x_range: Tuple[float, float] = (-16.0, 16.0)
    y_range: Tuple[float, float] = (-16.0, 16.0)
    z_range: Tuple[float, float] = (-2.0, 8.0)
    voxel_size: float = 0.5

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        nx = int((self.x_range[1] - self.x_range[0]) / self.voxel_size)
        ny = int((self.y_range[1] - self.y_range[0]) / self.voxel_size)
        nz = int((self.z_range[1] - self.z_range[0]) / self.voxel_size)
        return (max(nx, 1), max(ny, 1), max(nz, 1))


# ==========================
# ACCUMULATEUR LiDAR
# ==========================
class LidarAccumulatorUntilTarget:
    """Accumule sur plusieurs ticks jusqu'à atteindre un quota de points."""
    def __init__(self, target_points=20_000_000):
        self.lock = threading.Lock()
        self.target_points = int(target_points)
        self.points: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.total_points = 0

    def reset(self, target_points=None):
        with self.lock:
            self.points.clear()
            self.labels.clear()
            self.total_points = 0
            if hasattr(self, "counts_by_tag"):
                self.counts_by_tag.clear()
            if target_points is not None:
                self.target_points = int(target_points)
            gc.collect()
            
    def get_tag_counts(self):
        with self.lock:
            if hasattr(self, "counts_by_tag"):
                return dict(self.counts_by_tag)
            return {}
    
    def add(self, pts, lbls, tag=None):
        with self.lock:
            self.points.append(pts.astype(np.float32).copy())
            self.labels.append(lbls.astype(np.uint8).copy())
            self.total_points += len(pts)

            if tag is not None:
                if not hasattr(self, "counts_by_tag"):
                    self.counts_by_tag = defaultdict(int)
                self.counts_by_tag[int(tag)] += len(pts)

    def is_complete(self):
        with self.lock:
            return self.total_points >= self.target_points

    def get(self):
        with self.lock:
            if not self.points:
                return None, None
            pts = np.vstack(self.points)
            lbls = np.hstack(self.labels)
            self.points.clear()
            self.labels.clear()
            return pts, lbls


# ==========================
# METEO
# ==========================
class WeatherManager:
    def __init__(self, world):
        self.world = world
        self.presets = [
            ("clear_noon", {
                'cloudiness': 15.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wind_intensity': 10.0,
                'sun_azimuth_angle': 180.0, 'sun_altitude_angle': 85.0, 'fog_density': 2.0, 'fog_distance': 150.0,
                'wetness': 0.0, 'fog_falloff': 0.2, 'scattering_intensity': 1.0, 'mie_scattering_scale': 0.03,
                'rayleigh_scattering_scale': 0.0331, 'dust_storm': 0.0
            }),
            ("overcast_morning", {
                'cloudiness': 80.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wind_intensity': 8.0,
                'sun_azimuth_angle': 90.0, 'sun_altitude_angle': 20.0, 'fog_density': 5.0, 'fog_distance': 120.0,
                'wetness': 10.0, 'fog_falloff': 0.3, 'scattering_intensity': 1.2, 'mie_scattering_scale': 0.04,
                'rayleigh_scattering_scale': 0.04, 'dust_storm': 0.0
            }),
            ("rainy_noon", {
                'cloudiness': 90.0, 'precipitation': 60.0, 'precipitation_deposits': 50.0, 'wind_intensity': 50.0,
                'sun_azimuth_angle': 180.0, 'sun_altitude_angle': 60.0, 'fog_density': 20.0, 'fog_distance': 50.0,
                'wetness': 70.0, 'fog_falloff': 0.5, 'scattering_intensity': 1.0, 'mie_scattering_scale': 0.05,
                'rayleigh_scattering_scale': 0.05, 'dust_storm': 0.0
            }),
            ("foggy_evening", {
                'cloudiness': 50.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wind_intensity': 10.0,
                'sun_azimuth_angle': 270.0, 'sun_altitude_angle': 15.0, 'fog_density': 50.0, 'fog_distance': 30.0,
                'wetness': 20.0, 'fog_falloff': 0.9, 'scattering_intensity': 2.0, 'mie_scattering_scale': 0.1,
                'rayleigh_scattering_scale': 0.08, 'dust_storm': 0.0
            }),
            ("clear_night", {
                'cloudiness': 10.0, 'precipitation': 0.0, 'precipitation_deposits': 0.0, 'wind_intensity': 5.0,
                'sun_azimuth_angle': 0.0, 'sun_altitude_angle': -10.0, 'fog_density': 2.0, 'fog_distance': 150.0,
                'wetness': 0.0, 'fog_falloff': 0.2, 'scattering_intensity': 0.8, 'mie_scattering_scale': 0.02,
                'rayleigh_scattering_scale': 0.03, 'dust_storm': 0.0
            }),
            ("rainy_night", {
                'cloudiness': 95.0, 'precipitation': 80.0, 'precipitation_deposits': 70.0, 'wind_intensity': 40.0,
                'sun_azimuth_angle': 0.0, 'sun_altitude_angle': -8.0, 'fog_density': 25.0, 'fog_distance': 60.0,
                'wetness': 90.0, 'fog_falloff': 0.6, 'scattering_intensity': 0.9, 'mie_scattering_scale': 0.05,
                'rayleigh_scattering_scale': 0.04, 'dust_storm': 0.0
            }),
        ]

    def apply_by_id(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.presets):
            print(f"⚠️ weather-id {idx} invalide. Utilisation de 0.")
            idx = 0
        name, p = self.presets[idx]
        weather = carla.WeatherParameters(
            cloudiness=p['cloudiness'], precipitation=p['precipitation'], precipitation_deposits=p['precipitation_deposits'],
            wind_intensity=p['wind_intensity'], sun_azimuth_angle=p['sun_azimuth_angle'], sun_altitude_angle=p['sun_altitude_angle'],
            fog_density=p['fog_density'], fog_distance=p['fog_distance'], wetness=p['wetness'], fog_falloff=p['fog_falloff'],
            scattering_intensity=p['scattering_intensity'], mie_scattering_scale=p['mie_scattering_scale'],
            rayleigh_scattering_scale=p['rayleigh_scattering_scale'], dust_storm=p['dust_storm']
        )
        self.world.set_weather(weather)
        for _ in range(2):
            self.world.tick()
        # time.sleep(0.1)
        print(f" 🌤️ Météo appliquée (FIXE): [{idx}] {name}")
        return name


# ==========================
# CAPTEURS PERSISTANTS
# ==========================
class PersistentSensorManager:
    def __init__(
        self,
        world,
        enable_blur: bool = True,
        capture_points_target: int = 20_000_000,
        z_min: float = 0.3,
        z_max: float = 2.3,
        z_step: float = 0.5,
        h_fov: float = 60.0,
        v_upper: float = 6.0,
        v_lower: float = -6.0,
        lidar_channels: int = 16,
        lidar_pps: int = 400_000,
        lidar_range: float = 100.0,
        perf: Optional[PerfStats] = None,
        cam_height_noise_pct: float = 0.0,
        cam_angle_noise_pct: float = 0.0,
        empty_points_per_hit: int = 2,
    ):
        self.world = world
        self.enable_blur = enable_blur
        self.sensor_ids = set()
        self.lidars = []
        self.cameras = []
        self.sensors_created = False
        self.current_robot_transform = None
        self.reference_robot_transform = None
        self.camera_data: Dict[str, np.ndarray] = {}
        self.camera_received: Dict[str, bool] = {}
        self.lock = threading.Lock()
        self.capture_points_target = int(capture_points_target)
        self.lidar_accumulator = LidarAccumulatorUntilTarget(self.capture_points_target)
        self.current_pose_j = None
        self.params = {
            'z_min': z_min, 'z_max': z_max, 'z_step': z_step,
            'h_fov': h_fov, 'v_upper': v_upper, 'v_lower': v_lower,
            'channels': int(lidar_channels), 'pps': int(lidar_pps), 'range': float(lidar_range)
        }
        self.perf = perf or PerfStats()
        self.LIDAR_CONFIGS = self._build_z_stack_configs()
        self.active_lidar_mask = [True] * len(self.LIDAR_CONFIGS)
        self.default_lidar_layout = [cfg['dz'] for cfg in self.LIDAR_CONFIGS]

        robot_height = 0.4
        self.CAMERA_CONFIGS = [
            {'dx': 0.2,  'dy': 0.0,  'dz': robot_height, 'pitch': 0.0, 'yaw': 0,    'name': 'front',       'fov': 71.4},
            {'dx': -0.2, 'dy': 0.0,  'dz': robot_height, 'pitch': 0.0, 'yaw': 180,  'name': 'back',        'fov': 71.4},
            {'dx': 0.0,  'dy': -0.2, 'dz': robot_height, 'pitch': 0.0, 'yaw': -90,  'name': 'left',        'fov': 71.4},
            {'dx': 0.0,  'dy': 0.2,  'dz': robot_height, 'pitch': 0.0, 'yaw': 90,   'name': 'right',       'fov': 71.4},
            {'dx': 0.2,  'dy': -0.2, 'dz': robot_height, 'pitch': 0.0, 'yaw': -45,  'name': 'front_left',  'fov': 71.4},
            {'dx': 0.2,  'dy': 0.2,  'dz': robot_height, 'pitch': 0.0, 'yaw': 45,   'name': 'front_right', 'fov': 71.4},
            {'dx': -0.2, 'dy': -0.2, 'dz': robot_height, 'pitch': 0.0, 'yaw': -135, 'name': 'back_left',   'fov': 71.4},
            {'dx': -0.2, 'dy': 0.2,  'dz': robot_height, 'pitch': 0.0, 'yaw': 135,  'name': 'back_right',  'fov': 71.4},
        ]
        
        self.cam_height_noise_pct = float(cam_height_noise_pct)
        self.cam_angle_noise_pct = float(cam_angle_noise_pct)

        self.empty_points_per_hit = max(int(empty_points_per_hit), 0)

        print("🎯 CAPTURE LiDAR/CAM — repère robot T0 + multi-poses")

    def _build_z_stack_configs(self):
        cfgs = []
        z = self.params['z_min']
        while z <= self.params['z_max'] + 1e-6:
            cfgs.append({
                'dx': 0.0, 'dy': 0.0, 'dz': z,
                'channels': self.params['channels'],
                'pps': str(self.params['pps']),
                'range': self.params['range'],
                'upper_fov': self.params['v_upper'],
                'lower_fov': self.params['v_lower'],
                'horizontal_fov': self.params['h_fov'],
                'name': f'z_{z:.2f}'
            })
            z += self.params['z_step']
        return cfgs

    @staticmethod
    def _rpy_to_matrix(roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ], dtype=np.float32)

    def _to_robot_frame(self, pts_local, sensor_transform):
        # Matrix capteur -> Monde
        m_sensor_world = np.array(sensor_transform.get_matrix())
        
        # Matrix Monde -> Robot T0 (Inverse de la matrice du robot à T0)
        ref_pos = self.reference_robot_transform or self.current_robot_transform
        robot_tf = carla.Transform(
            carla.Location(**ref_pos['location']),
            carla.Rotation(**ref_pos['rotation'])
        )
        m_world_robot = np.array(robot_tf.get_inverse_matrix())

        # On combine les deux pour avoir : Capteur -> Monde -> Robot
        m_combined = np.dot(m_world_robot, m_sensor_world)

        # Application aux points
        points_4d = np.ones((pts_local.shape[0], 4))
        points_4d[:, :3] = pts_local
        pts_robot = np.dot(m_combined, points_4d.T).T
        
        return pts_robot[:, :3].astype(np.float32)

    def set_reference_robot(self, position: dict):
        """Fixe le repère ROBOT T0 pour la frame courante."""
        self.reference_robot_transform = position

    def cleanup_orphan_sensors(self):
        with SectionTimer(self.perf, "cleanup_orphan_sensors"):
            actors = self.world.get_actors()
            cleaned = 0
            for a in actors:
                if 'sensor' in a.type_id and a.id not in self.sensor_ids:
                    try:
                        a.destroy()
                        cleaned += 1
                    except Exception:
                        pass
            if cleaned:
                print(f" ✅ {cleaned} capteurs orphelins supprimés")
                [self.world.tick() for _ in range(2)]
                # time.sleep(0.1)
            return cleaned

    def apply_camera_blur(self, image, weather_preset=None):
        if not self.enable_blur:
            return image
        blur_intensity = 1
        if weather_preset:
            if 'foggy' in weather_preset:
                blur_intensity = 5
            elif 'rainy' in weather_preset or 'storm' in weather_preset:
                blur_intensity = 3
            elif 'night' in weather_preset:
                blur_intensity = 2
        k = (2 * blur_intensity + 1, 2 * blur_intensity + 1)
        with SectionTimer(self.perf, "camera_blur"):
            blurred = cv2.GaussianBlur(image, k, 0)
            if weather_preset and ('rainy' in weather_preset):
                noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
                blurred = cv2.add(blurred, noise)
        return blurred


    def move_cameras_to_position(self, position: dict):
        """Déplace UNIQUEMENT les caméras à une pose donnée (ici T0)."""
        ego_tf = carla.Transform(
            carla.Location(**position['location']),
            carla.Rotation(**position['rotation'])
        )

        for cam, cfg in zip(self.cameras, self.CAMERA_CONFIGS):
            if cam and cam.is_alive:
                cam_loc_world = ego_tf.transform(carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz']))
                cam_rot_world = carla.Rotation(
                    pitch=ego_tf.rotation.pitch + cfg['pitch'],
                    yaw=ego_tf.rotation.yaw + cfg['yaw'],
                    roll=ego_tf.rotation.roll
                )
                cam.set_transform(carla.Transform(cam_loc_world, cam_rot_world))


    def move_lidars_to_position(self, position: dict):
        """Déplace UNIQUEMENT les LiDARs à une pose j."""
        ego_tf = carla.Transform(
            carla.Location(**position['location']),
            carla.Rotation(**position['rotation'])
        )

        for lidar, cfg in zip(self.lidars, self.LIDAR_CONFIGS):
            if lidar and lidar.is_alive:
                sensor_loc_local = carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz'])
                sensor_loc_world = ego_tf.transform(sensor_loc_local)
                sensor_rot_world = carla.Rotation(
                    pitch=ego_tf.rotation.pitch,
                    yaw=ego_tf.rotation.yaw,
                    roll=ego_tf.rotation.roll
                )
                lidar.set_transform(carla.Transform(sensor_loc_world, sensor_rot_world))




    def create_sensors_once(self, start_transform: carla.Transform):
        if self.sensors_created:
            return True
        print("🔧 Création des capteurs (Z-STACK + cams)...")
        self.cleanup_orphan_sensors()
        try:
            with SectionTimer(self.perf, "create_sensors_total"):
                bp_library = self.world.get_blueprint_library()

                # LiDARs
                with SectionTimer(self.perf, "create_lidars"):
                    for i, cfg in enumerate(self.LIDAR_CONFIGS):
                        lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
                        lidar_bp.set_attribute('channels', str(cfg['channels']))
                        lidar_bp.set_attribute('points_per_second', str(cfg['pps']))
                        lidar_bp.set_attribute('rotation_frequency', '20')
                        lidar_bp.set_attribute('range', str(cfg['range']))
                        lidar_bp.set_attribute('upper_fov', str(cfg['upper_fov']))
                        lidar_bp.set_attribute('lower_fov', str(cfg['lower_fov']))
                        lidar_bp.set_attribute('horizontal_fov', str(cfg['horizontal_fov']))
                        try:
                            lidar_bp.set_attribute('role_name', 'virtual_sensor')
                        except Exception:
                            pass

                        tf = carla.Transform(
                            start_transform.location + carla.Location(
                                x=cfg['dx'], y=cfg['dy'], z=cfg['dz']),
                            carla.Rotation()
                        )
                        lidar = self.world.spawn_actor(lidar_bp, tf)
                        self.lidars.append(lidar)
                        self.sensor_ids.add(lidar.id)

                        def make_cb(sensor, idx=i):
                            def _cb(data):
                                if not self.active_lidar_mask[idx]:
                                    return
                                try:
                                    arr = np.frombuffer(data.raw_data, dtype=np.dtype([
                                        ('x', np.float32), ('y', np.float32), ('z', np.float32),
                                        ('CosAngle', np.float32),
                                        ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
                                    ]))
                                    if len(arr) == 0:
                                        return

                                    pts_local = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
                                    sensor_tf = sensor.get_transform()

                                    # Hits en repère robot
                                    pts_robot_hits = self._to_robot_frame(pts_local, sensor_tf)
                                    lbl_hits = arr['ObjTag'].astype(np.uint8)

                                    # -------- EMPTY avant hit --------
                                    pts_robot_empty = np.zeros((0, 3), dtype=np.float32)
                                    lbl_empty = np.zeros((0,), dtype=np.uint8)

                                    if self.empty_points_per_hit > 0:
                                        n_hits = len(pts_local)
                                        k = self.empty_points_per_hit
                                        t = np.random.rand(n_hits, k).astype(np.float32) * 0.98  # (0..0.98)
                                        pts_empty_local = (pts_local[:, None, :] * t[..., None]).reshape(-1, 3)
                                        pts_robot_empty = self._to_robot_frame(pts_empty_local, sensor_tf)
                                        lbl_empty = np.full((len(pts_robot_empty),), LIDAR_EMPTY_SENTINEL, dtype=np.uint8)

                                    # -------- UNKNOWN derrière hit jusqu'à 16 m (1 point par hit) --------
                                    # On prend un point sur la prolongation du rayon : p_unk = p_hit * s, avec s in (1 .. 16/||p_hit||]
                                    pts_robot_unk = np.zeros((0, 3), dtype=np.float32)
                                    lbl_unk = np.zeros((0,), dtype=np.uint8)

                                    max_range_m = 16.0
                                    n = pts_local.shape[0]

                                    # norme du hit dans repère capteur
                                    d = np.linalg.norm(pts_local, axis=1).astype(np.float32)  # (n,)
                                    valid = (d > 1e-3) & (d < max_range_m - 1e-3)
                                    if np.any(valid):
                                        d_v = d[valid]
                                        pts_hit_v = pts_local[valid]  # (nv,3)

                                        # scale max pour aller à 16m sur le même rayon
                                        s_max = (max_range_m / d_v)  # (nv,)

                                        # tire un scale aléatoire entre juste derrière le hit (1.02) et s_max
                                        # clamp si jamais s_max < 1.02 (cas limite)
                                        s_min = 1.02
                                        s_hi = np.maximum(s_max, s_min + 1e-3)
                                        r = np.random.rand(s_hi.shape[0]).astype(np.float32)
                                        s = s_min + r * (s_hi - s_min)  # (nv,)

                                        pts_unk_local = pts_hit_v * s[:, None]  # (nv,3)
                                        pts_robot_unk = self._to_robot_frame(pts_unk_local, sensor_tf)
                                        lbl_unk = np.full((len(pts_robot_unk),), LIDAR_UNKNOWN_SENTINEL, dtype=np.uint8)

                                    # concat
                                    pts_concat = np.vstack([pts_robot_hits, pts_robot_empty, pts_robot_unk])
                                    lbl_concat = np.hstack([lbl_hits, lbl_empty, lbl_unk])

                                    self.lidar_accumulator.add(pts_concat, lbl_concat, tag=self.current_pose_j)

                                except Exception:
                                    print(f"Erreur LiDAR (id={sensor.id}):")
                                    traceback.print_exc()

                            return _cb


                        lidar.listen(make_cb(lidar))

                # Cameras
                with SectionTimer(self.perf, "create_cameras"):
                    for cfg in self.CAMERA_CONFIGS:
                        cam_bp = bp_library.find('sensor.camera.rgb')
                        # --- RÉSOLUTIONS ---
                        # Version optimisée pour BiFPN (Multiple de 32) basée sur 4080x3072
                        cam_bp.set_attribute('image_size_x', '512')
                        cam_bp.set_attribute('image_size_y', '384')

                        # --- OPTIQUE ---
                        # Utilisation du Horizontal FOV exact de votre fiche technique
                        cam_bp.set_attribute('fov', '71.4')

                        # # --- PHYSIQUE DU CAPTEUR ---
                        # # Simulation de l'ouverture f/1.9
                        # cam_bp.set_attribute('exposure_mode', 'Manual')
                        cam_bp.set_attribute('fstop', '1.9')

                        # # --- NETTOYAGE (Optique pure sans bruit) ---
                        # # Désactivation de la distorsion pour simuler le post-process Google
                        # cam_bp.set_attribute('lens_k', '0.0')
                        # cam_bp.set_attribute('lens_kcube', '0.0')
                        # cam_bp.set_attribute('lens_circle_multiplier', '0.0')
                        # cam_bp.set_attribute('lens_circle_falloff', '0.0')
                        # # Optionnel : Vitesse d'obturation standard pour éviter trop de flou de mouvement
                        # # cam_bp.set_attribute('shutter_speed', '200.0') # 1/200s
                        # cam_bp.set_attribute('iso', '6000')
                        # cam_bp.set_attribute('exposure_compensation', '2.0')
                        # cam_bp.set_attribute('shutter_speed', '10.0') # Plus lent = plus de lumière
                        # cam_bp.set_attribute('exposure_mode', 'histogram')
                        # cam_bp.set_attribute('exposure_min_bright', '0.1')
                        # cam_bp.set_attribute('exposure_max_bright', '0.1') # Verrouille la clarté
                        # cam_bp.set_attribute('exposure_compensation', '1.5') # Boost lumineux
                        try:
                            cam_bp.set_attribute('role_name', 'virtual_sensor')
                        except Exception:
                            pass

                        tf = carla.Transform(
                            start_transform.location + carla.Location(
                                x=cfg['dx'], y=cfg['dy'], z=cfg['dz']),
                            carla.Rotation(pitch=cfg['pitch'], yaw=cfg['yaw'])
                        )
                        cam = self.world.spawn_actor(cam_bp, tf)
                        self.cameras.append(cam)
                        self.sensor_ids.add(cam.id)
                        self.camera_data[cfg['name']] = None
                        self.camera_received[cfg['name']] = False

                        def make_cam_cb(name):
                            def _cb(image):
                                try:
                                    image.convert(carla.ColorConverter.Raw)
                                    array = np.frombuffer(image.raw_data, dtype=np.uint8)
                                    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
                                    with self.lock:
                                        self.camera_data[name] = array.copy()
                                        self.camera_received[name] = True
                                except Exception:
                                    pass
                            return _cb

                        cam.listen(make_cam_cb(cfg['name']))

                self.sensors_created = True
                print(f"✅ {len(self.lidars)} LiDARs et {len(self.cameras)} caméras créés")
                for _ in range(2):
                    self.world.tick()
                # time.sleep(0.1)
                return True
        except Exception:
            print("❌ Erreur création capteurs:")
            traceback.print_exc()
            return False

    def set_lidar_layout(self, dz_list: Optional[List[float]]):
        """Active seulement certains LiDAR (par hauteur) ou tous si None."""
        if dz_list is None:
            self.active_lidar_mask = [True] * len(self.LIDAR_CONFIGS)
            return
        tol = 1e-3
        new_mask = []
        for cfg in self.LIDAR_CONFIGS:
            active = any(abs(cfg['dz'] - dz) <= tol for dz in dz_list)
            new_mask.append(active)
        self.active_lidar_mask = new_mask

    def randomize_lidar_params(self,
                               z_jitter: Tuple[float, float] = (-0.2, 0.2),
                               yaw_jitter: Tuple[float, float] = (-10.0, 15.0),
                               channels_range: Tuple[int, int] = (512, 1024),
                               upper_fov: Tuple[float, float] = (40.0, 60.0),
                               lower_fov: Tuple[float, float] = (-60.0, -40.0),
                               pps_range: Tuple[int, int] = (10_000_000, 12_100_000)):
        # Petit random sur les LiDARs existants
        for lidar, cfg in zip(self.lidars, self.LIDAR_CONFIGS):
            if not (lidar and lidar.is_alive):
                continue
            tf = lidar.get_transform()
            loc = tf.location
            rot = tf.rotation

            dz = random.uniform(*z_jitter)
            loc.z = cfg['dz'] + dz
            if loc.z < 0.1:
                loc.z = 0.1
            dyaw = random.uniform(*yaw_jitter)
            rot.yaw += dyaw

            lidar.set_transform(carla.Transform(loc, rot))

            ch = random.randint(channels_range[0], channels_range[1])
            pps = random.randint(pps_range[0], pps_range[1])
            ufov = random.uniform(*upper_fov)
            lfov = random.uniform(*lower_fov)
            if lfov > -1.0 * ufov:
                lfov = -1.0 * ufov
            try:
                lidar.set_attribute('channels', str(ch))
                lidar.set_attribute('upper_fov', str(ufov))
                lidar.set_attribute('lower_fov', str(lfov))
                lidar.set_attribute('points_per_second', str(pps))
                lidar.set_attribute('rotation_frequency', '20')
            except Exception:
                pass

    def move_sensors_to_position(self, position: dict):
        if not self.sensors_created:
            return False
        try:
            # 1. Transform de l'Ego-véhicule (le parent)
            ego_tf = carla.Transform(
                carla.Location(**position['location']),
                carla.Rotation(**position['rotation'])
            )
            self.current_robot_transform = position

            # 2. Placement des LiDARs
            for lidar, cfg in zip(self.lidars, self.LIDAR_CONFIGS):
                if lidar and lidar.is_alive:
                    # On calcule la position mondiale du capteur
                    # .transform() prend un point LOCAL et le projette dans le MONDE
                    sensor_loc_local = carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz'])
                    sensor_loc_world = ego_tf.transform(sensor_loc_local)
                    
                    # Pour la rotation, on additionne les angles (valide pour les capteurs fixes)
                    sensor_rot_world = carla.Rotation(
                        pitch=ego_tf.rotation.pitch,
                        yaw=ego_tf.rotation.yaw,
                        roll=ego_tf.rotation.roll
                    )
                    
                    lidar.set_transform(carla.Transform(sensor_loc_world, sensor_rot_world))

            # 3. Placement des Caméras
            for cam, cfg in zip(self.cameras, self.CAMERA_CONFIGS):
                if cam and cam.is_alive:
                    rel_z = cfg['dz']
                    if self.cam_height_noise_pct > 0.0:
                        rnd = random.uniform(-self.cam_height_noise_pct, self.cam_height_noise_pct) / 100.0
                        rel_z *= (1.0 + rnd)

                    # Projection de la caméra
                    cam_loc_world = ego_tf.transform(carla.Location(x=cfg['dx'], y=cfg['dy'], z=rel_z))
                    
                    # Rotation combinée (Ego + Orientation de la caméra)
                    cam_rot_world = carla.Rotation(
                        pitch=ego_tf.rotation.pitch + cfg['pitch'],
                        yaw=ego_tf.rotation.yaw + cfg['yaw'],
                        roll=ego_tf.rotation.roll
                    )
                    
                    cam.set_transform(carla.Transform(cam_loc_world, cam_rot_world))
                    
            return True
        except Exception as e:
            print(f"❌ Erreur déplacement: {e}")
            return False
    def start_new_accumulation(self, target_points: Optional[int] = None):
        self.lidar_accumulator.reset(target_points or self.capture_points_target)
        for name in self.camera_received:
            self.camera_received[name] = False

    def capture_current_frame(self, weather_preset=None):
        with SectionTimer(self.perf, "accumulate_lidar_get"):
            points, labels = self.lidar_accumulator.get()
        if points is None or len(points) == 0:
            print(" ❌ Pas de points LiDAR")
            return None
        print(f" ✅ {len(points):,} points LiDAR (hits + empty) accumulés (repère ROBOT)")
        images = {}
        with SectionTimer(self.perf, "camera_copy_blur"):
            with self.lock:
                for name, img in self.camera_data.items():
                    if img is not None:
                        images[name] = self.apply_camera_blur(img.copy(), weather_preset)
        return {
            'points': points,
            'labels': labels,
            'images': images,
            'scan_duration': 0.0,
            'ticks': 0
        }

    def cleanup(self):
        print("🧹 Nettoyage des capteurs...")
        with SectionTimer(self.perf, "sensors_stop"):
            for lidar in self.lidars:
                if lidar and lidar.is_alive:
                    try:
                        lidar.stop()
                    except Exception:
                        pass
            for cam in self.cameras:
                if cam and cam.is_alive:
                    try:
                        cam.stop()
                    except Exception:
                        pass
        destroyed = 0
        with SectionTimer(self.perf, "sensors_destroy"):
            for s in self.lidars + self.cameras:
                if s and s.is_alive:
                    try:
                        s.destroy()
                        destroyed += 1
                    except Exception:
                        pass
        print(f" {destroyed} capteurs détruits")
        self.lidars.clear()
        self.cameras.clear()
        self.sensor_ids.clear()
        self.sensors_created = False
        print("✅ Nettoyage terminé")


# ==========================
# GÉNÉRATEUR DE DATASET
# ==========================
class FastDatasetGenerator:
    def __init__(
        self,
        output_dir="carla_robot_dataset",
        enable_blur=True,
        preview_interval=50,
        tm_port=8000,
        seed=42,
        z_min=0.3, z_max=2.3, z_step=0.5,
        h_fov=360.0, v_upper=15.0, v_lower=-15.0,
        lidar_channels=512, lidar_pps=1_000_000, lidar_range=500.0,
        previews_dir_name="previews",
        map_name="Town03",
        trajectory_json=None,
        weather_id: int = 0,
        profile: bool = True,
        capture_points_target: int = 2_000_000,
        points_min_saved: int = 50_000,
        points_max_saved: int = 80_000,
        cube_size_m: float = 0.05,
        cam_height_noise_pct: float = 5.0,
        cam_angle_noise_pct: float = 5.0,
        # fenêtre ego
        window_back: int = 20,
        window_forward: int = 20,
        proximity_radius: float = 0.5,
        lidar_layout_if_clear: Optional[List[float]] = None,
        allowed_semantic_tags: Optional[List[int]] = None,
        max_ticks_per_pose: int = 60,
        randomize_clear_poses: bool = True,
        # occupancy implicite
        implicit_voxel_size: float = 0.5,
        implicit_points_per_voxel_min: int = 10,
        implicit_points_per_voxel_max: int = 20,
        implicit_ratio_occ: float = 0.4,
        implicit_ratio_empty: float = 0.4,
        implicit_ratio_unknown: float = 0.2,
        voxel_keep_ratio_empty: float = 0.3,
        voxel_keep_ratio_unknown: float = 0.1,
        lidar_empty_points_per_hit: int = 2,
    ):
        self.output_dir = output_dir
        self.previews_dir = os.path.join(output_dir, previews_dir_name)
        self.enable_blur = enable_blur
        self.preview_interval = preview_interval

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        os.makedirs(self.previews_dir, exist_ok=True)

        self.client = None
        self.world = None
        self.sensor_manager: PersistentSensorManager = None
        self.weather_manager: WeatherManager = None
        self.global_frame_counter = 0

        self.voxel_cfg = VoxelConfig(
            x_range=(-16.0, 16.0),
            y_range=(-16.0, 16.0),
            z_range=(-2.0, 8.0),
            voxel_size=implicit_voxel_size
        )

        
        self.tm_port = tm_port
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.z_min = z_min
        self.z_max = z_max
        self.z_step = z_step
        self.h_fov = h_fov
        self.v_upper = v_upper
        self.v_lower = v_lower
        self.lidar_channels = lidar_channels
        self.lidar_pps = lidar_pps
        self.lidar_range = lidar_range
        self.map_name = map_name
        self.trajectory_json = trajectory_json
        self.traj_data = None
        self.positions = []
        self.weather_id = int(weather_id)
        self.fixed_weather_name = None
        self.perf = PerfStats() if profile else PerfStats()
        self.capture_points_target = int(capture_points_target)
        self.points_min_saved = int(points_min_saved)
        self.points_max_saved = int(points_max_saved)
        self.cube_size_m = float(cube_size_m)
        self.cam_height_noise_pct = float(cam_height_noise_pct)
        self.cam_angle_noise_pct = float(cam_angle_noise_pct)

        self.window_back = int(window_back)
        self.window_forward = int(window_forward)
        self.proximity_radius = float(proximity_radius)
        self.lidar_layout_if_clear = lidar_layout_if_clear
        self.allowed_semantic_tags = allowed_semantic_tags
        self.max_ticks_per_pose = int(max_ticks_per_pose)
        self.randomize_clear_poses = bool(randomize_clear_poses)

        # occupancy implicite
        self.points_per_voxel_min = int(implicit_points_per_voxel_min)
        self.points_per_voxel_max = int(implicit_points_per_voxel_max)

        r_occ = float(implicit_ratio_occ) 
        r_empty = float(implicit_ratio_empty) 
        r_unk = float(implicit_ratio_unknown) 
        s = max(r_occ + r_empty + r_unk, 1e-6)
        self.ratio_occ = r_occ / s
        self.ratio_empty = r_empty / s
        self.ratio_unknown = r_unk / s

        self.voxel_keep_ratio_empty = float(voxel_keep_ratio_empty)
        self.voxel_keep_ratio_unknown = float(voxel_keep_ratio_unknown)

        self.lidar_empty_points_per_hit = int(lidar_empty_points_per_hit)

        # -----------------------
        # OCC UPSAMPLING CONFIG
        # -----------------------
        self.occ_upsample_radius = 0.08      # rayon R
        self.occ_upsample_prob = 1.0         # proba d'ajouter 1 point par point occupé (1.0 = toujours)
        self.occ_offset_bank_size = 8192     # taille de la banque d'offsets

        # Banque d'offsets uniformes dans une boule de rayon R (pas juste sur la sphère)
        # -> plus réaliste + évite "coquille" autour du point
        rng = np.random.default_rng(12345)

        v = rng.normal(size=(self.occ_offset_bank_size, 3)).astype(np.float32)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

        rad = (rng.random(self.occ_offset_bank_size, dtype=np.float32) ** (1.0 / 3.0)) * float(self.occ_upsample_radius)
        self._occ_offset_bank = v * rad[:, None]   # (K,3) float32

        # Thread de sauvegarde : 
        self.writer = AsyncWriter(self.save_frame)

        print("🤖 CONFIGURATION DATASET OCCUPANCY IMPLICITE (multi-poses)")
        print(f"  Carte: {self.map_name}")
        print(f"  Grille: {self.voxel_cfg.x_range} x {self.voxel_cfg.y_range} x {self.voxel_cfg.z_range}")
        print(f"  Taille voxel: {self.voxel_cfg.voxel_size} m")
        print(f"  Points/voxel: [{self.points_per_voxel_min} ; {self.points_per_voxel_max}]")
        print(f"  Fenêtre ego: back={self.window_back}, fwd={self.window_forward}")
        print(f"  proximity_radius={self.proximity_radius}")
        print(f"  lidar_layout_if_clear={self.lidar_layout_if_clear}")
        print(f"  ratios (occ/empty/unknown) ≈ "
              f"{self.ratio_occ:.2f}/{self.ratio_empty:.2f}/{self.ratio_unknown:.2f}")
        print(f"  empty_points_per_hit={self.lidar_empty_points_per_hit}")


    def connect(self):
        try:
            with SectionTimer(self.perf, "connect_total"):
                self.client = carla.Client('localhost', 2000)
                self.client.set_timeout(120.0)

                if not self.trajectory_json:
                    print("❌ Erreur: Aucun fichier JSON de trajectoire spécifié")
                    return False
                if not os.path.isfile(self.trajectory_json):
                    print(f"❌ Erreur: Fichier trajectoire '{self.trajectory_json}' introuvable")
                    return False
                with SectionTimer(self.perf, "read_trajectory_json"):
                    with open(self.trajectory_json, 'r', encoding='utf-8') as f:
                        self.traj_data = json.load(f)
                if 'positions' not in self.traj_data or not self.traj_data['positions']:
                    print(f"❌ Le fichier trajectoire '{self.trajectory_json}' ne contient pas de positions valides")
                    return False

                self.map_name = self.traj_data.get('map_name', self.map_name)
                self.positions = self.traj_data['positions']
                print(f"✅ Trajectoire chargée: {len(self.positions)} positions | Carte: {self.map_name}")

                with SectionTimer(self.perf, "load_world"):
                    self.world = self.client.load_world(self.map_name)
                print("✅ Monde CARLA prêt")

                self._deep_cleanup_world()

                with SectionTimer(self.perf, "apply_world_settings"):
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    settings.no_rendering_mode = False
                    self.world.apply_settings(settings)

                with SectionTimer(self.perf, "apply_weather"):
                    self.weather_manager = WeatherManager(self.world)
                    self.fixed_weather_name = self.weather_manager.apply_by_id(self.weather_id)
                print(f"✅ Connecté à CARLA (météo fixe: {self.fixed_weather_name})")
                return True
        except Exception:
            print("❌ Erreur connexion:")
            traceback.print_exc()
            return False

    def _deep_cleanup_world(self):
        print("🧹 Nettoyage du monde (sensors/vehicles/walkers)...")
        with SectionTimer(self.perf, "world_deep_cleanup"):
            actors = self.world.get_actors()
            batch = []
            for a in actors:
                try:
                    if 'sensor' in a.type_id:
                        batch.append(carla.command.DestroyActor(a))
                    elif 'vehicle' in a.type_id:
                        batch.append(carla.command.DestroyActor(a))
                    elif 'walker' in a.type_id and 'controller' in a.type_id:
                        batch.append(carla.command.DestroyActor(a))
                    elif 'walker.pedestrian' in a.type_id:
                        batch.append(carla.command.DestroyActor(a))
                except Exception:
                    pass
            if batch:
                self.client.apply_batch_sync(batch, True)
            for _ in range(2):
                self.world.tick()

    def _render_preview(self, points_xyz: np.ndarray, labels: np.ndarray, frame_id: int, out_path: str):
        if points_xyz.size == 0:
            return

        # LUT id -> RGB
        id_to_rgb = {}
        for cid, _name, rgb in CARLA_22:
            id_to_rgb[int(cid)] = tuple(rgb)
            
        DEFAULT_COLOR = (0, 0, 0)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Frame {frame_id} - points={len(points_xyz):,}")

        marker_size = max(0.1, int(self.cube_size_m * 200))

        lbl = labels.astype(int)
        colors = [id_to_rgb.get(v, DEFAULT_COLOR) for v in lbl]
        colors = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in colors]

        x = points_xyz[:, 0]
        y = points_xyz[:, 1]
        z = points_xyz[:, 2]

        ax.scatter(x, y, z, c=colors, marker='.', s=marker_size, depthshade=False)

        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        zmin, zmax = np.min(z), np.max(z)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=20, azim=45)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

    # ---------- logique ego -----------

    def _ego_window_indices(self, idx: int) -> List[int]:
        lo = max(0, idx - self.window_back)
        hi = min(len(self.positions) - 1, idx + self.window_forward)
        return list(range(lo, hi + 1))

    def _pose_has_actor_at_T0(self,
                              pose: dict,
                              actors_at_T0: List[dict],
                              radius: float) -> bool:
        cx = pose['ego_location']['x']
        cy = pose['ego_location']['y']
        cz = pose['ego_location']['z']
        r2 = radius * radius
        for a in actors_at_T0:
            tid = a.get('type', '')
            # if ('vehicle' not in tid) and ('walker' not in tid) and ('pedestrian' not in tid):
            #     continue
            lx = a['location']['x']
            ly = a['location']['y']
            lz = a['location']['z']
            dx = lx - cx
            dy = ly - cy
            dz = lz - cz
            if dx * dx + dy * dy + dz * dz <= r2:
                return True
        return False

    # ---------- OCCUPANCY IMPLICITE ----------

    def _build_implicit_from_points(
        self,
        pts_robot: np.ndarray,
        lbl_raw: np.ndarray,
        target_total_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        EMPTY_LABEL = 254
        UNKNOWN_LABEL = 253
        if pts_robot is None or pts_robot.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

        vx = self.voxel_cfg
        vs = float(vx.voxel_size)
        x_min, x_max = vx.x_range
        y_min, y_max = vx.y_range
        z_min, z_max = vx.z_range

        pts_robot = np.asarray(pts_robot, dtype=np.float32)
        lbl_raw = np.asarray(lbl_raw)

        # ROI
        x = pts_robot[:, 0]
        y = pts_robot[:, 1]
        z = pts_robot[:, 2]
        mask_in = (
            (x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max)
        )
        if not np.any(mask_in):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

        pts = pts_robot[mask_in]
        lbl_raw = lbl_raw[mask_in]

        # split hit / empty / unknown-ray
        mask_empty = (lbl_raw == LIDAR_EMPTY_SENTINEL)
        mask_unknown = (lbl_raw == LIDAR_UNKNOWN_SENTINEL)
        mask_hit = ~(mask_empty | mask_unknown)

        pts_hits = pts[mask_hit]
        lbl_hits_raw = lbl_raw[mask_hit].astype(np.uint8, copy=False)

        pts_empty = pts[mask_empty]
        pts_unknown_ray = pts[mask_unknown]

        lbl_hits_id = (
            lbl_hits_raw.astype(np.uint8, copy=False)
            if lbl_hits_raw.size else np.zeros((0,), dtype=np.uint8)
        )

        nx, ny, nz = vx.grid_shape
        n_voxels = int(nx) * int(ny) * int(nz)

        inv_vs = 1.0 / vs

        def voxel_flat_ids(p: np.ndarray) -> np.ndarray:
            if p.size == 0:
                return np.zeros((0,), dtype=np.int64)
            ix = ((p[:, 0] - x_min) * inv_vs).astype(np.int32)
            iy = ((p[:, 1] - y_min) * inv_vs).astype(np.int32)
            iz = ((p[:, 2] - z_min) * inv_vs).astype(np.int32)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            np.clip(iz, 0, nz - 1, out=iz)
            return (ix + nx * (iy + ny * iz)).astype(np.int64, copy=False)

        flat_hit = voxel_flat_ids(pts_hits)
        flat_empty = voxel_flat_ids(pts_empty)
        flat_unknown = voxel_flat_ids(pts_unknown_ray)

        # label par voxel OCC = label du 1er hit du voxel (comme avant)
        voxel_occ_label = np.full((n_voxels,), -1, dtype=np.uint8)
        if flat_hit.size:
            uniq_hit, first_idx = np.unique(flat_hit, return_index=True)
            voxel_occ_label[uniq_hit] = lbl_hits_id[first_idx]

        rng = np.random.default_rng()

        # ===== OCCUPIED (hits réels) =====
        points_occ = np.zeros((0, 3), dtype=np.float16)
        labels_occ = np.zeros((0,), dtype=np.uint8)

        if pts_hits.size:
            order = np.argsort(flat_hit, kind="mergesort")
            flat_sorted = flat_hit[order]
            uniq_v, start_idx, counts = np.unique(flat_sorted, return_index=True, return_counts=True)

            n_vox = int(uniq_v.size)
            if n_vox:
                n_pts_per_vox = rng.integers(
                    self.points_per_voxel_min,
                    self.points_per_voxel_max + 1,
                    size=n_vox,
                    dtype=np.int32
                )

                # ----- PRE-ALLOC avec marge pour upsampling -----
                # si on ajoute ~1 point par point: facteur 2
                R = self.occ_upsample_radius
                p_up = self.occ_upsample_prob

                base_total = int(n_pts_per_vox.sum())
                if R > 0.0 and p_up > 0.0 and hasattr(self, "_occ_offset_bank"):
                    # espérance des points ajoutés
                    extra_est = int(np.ceil(base_total * min(max(p_up, 0.0), 1.0)))
                else:
                    extra_est = 0

                out_pts = np.empty((base_total + extra_est, 3), dtype=np.float32)
                out_lbl = np.empty((base_total + extra_est,), dtype=np.uint8)

                write = 0
                K = self._occ_offset_bank.shape[0] if (R > 0.0 and extra_est > 0) else 0

                for i in range(n_vox):
                    v = int(uniq_v[i])
                    s = int(start_idx[i])
                    c = int(counts[i])
                    e = s + c
                    n_pts = int(n_pts_per_vox[i])

                    idxs = order[s:e]
                    chosen = rng.choice(idxs, size=n_pts, replace=(c < n_pts))

                    pts_vox = pts_hits[chosen]
                    lbl_vox = voxel_occ_label[v]

                    # --- write base ---
                    out_pts[write:write + n_pts] = pts_vox
                    out_lbl[write:write + n_pts] = lbl_vox
                    base_slice_start = write
                    write += n_pts

                    # --- inline upsampling OCC ---
                    if extra_est > 0:
                        # combien de points on ajoute pour ce voxel ?
                        # option A: 1 point ajouté par point (p_up=1)
                        # option B: binomial pour en ajouter une fraction (p_up < 1)
                        if p_up >= 1.0:
                            n_add = n_pts
                            add_idx_local = np.arange(n_pts, dtype=np.int32)
                        else:
                            # tire parmi les n_pts, sans replacement
                            n_add = int(rng.binomial(n_pts, p_up))
                            if n_add <= 0:
                                continue
                            add_idx_local = rng.choice(n_pts, size=n_add, replace=False)

                        # centres = sous-ensemble des points déjà écrits pour ce voxel
                        centers = out_pts[base_slice_start:base_slice_start + n_pts][add_idx_local]

                        # offsets pré-calculés
                        off_idx = rng.integers(0, K, size=n_add, dtype=np.int32)
                        new_pts = centers + self._occ_offset_bank[off_idx]

                        # clip ROI (évite sortir)
                        np.clip(new_pts[:, 0], x_min, x_max, out=new_pts[:, 0])
                        np.clip(new_pts[:, 1], y_min, y_max, out=new_pts[:, 1])
                        np.clip(new_pts[:, 2], z_min, z_max, out=new_pts[:, 2])

                        out_pts[write:write + n_add] = new_pts
                        out_lbl[write:write + n_add] = lbl_vox
                        write += n_add

                # trim exact
                points_occ = out_pts[:write]
                labels_occ = out_lbl[:write]

        # ===== EMPTY (LiDAR uniquement, capé par voxel) =====
        points_empty = np.zeros((0, 3), dtype=np.float16)
        labels_empty = np.zeros((0,), dtype=np.uint8)

        if pts_empty.size:
            order_e = np.argsort(flat_empty, kind="mergesort")
            flat_e_sorted = flat_empty[order_e]
            uniq_v_e, start_e, count_e = np.unique(flat_e_sorted, return_index=True, return_counts=True)

            n_vox_e = uniq_v_e.size
            if n_vox_e:
                n_pts_per_vox_e = rng.integers(
                    self.points_per_voxel_min,
                    self.points_per_voxel_max + 1,
                    size=n_vox_e,
                    dtype=np.int32
                )
                total_e = int(n_pts_per_vox_e.sum())
                out_pts_e = np.empty((total_e, 3), dtype=np.float32)
                out_lbl_e = np.full((total_e,), EMPTY_LABEL, dtype=np.uint8)

                write = 0
                for i in range(n_vox_e):
                    s = int(start_e[i])
                    c = int(count_e[i])
                    e = s + c
                    n_pts = int(n_pts_per_vox_e[i])

                    idxs = order_e[s:e]
                    chosen = rng.choice(idxs, size=n_pts, replace=(c < n_pts))

                    out_pts_e[write:write + n_pts] = pts_empty[chosen]
                    write += n_pts

                points_empty = out_pts_e
                labels_empty = out_lbl_e

        # ===== UNKNOWN (LiDAR unknown-ray uniquement, capé par voxel) =====
        points_unknown = np.zeros((0, 3), dtype=np.float16)
        labels_unknown = np.zeros((0,), dtype=np.uint8)

        if pts_unknown_ray.size:
            order_u = np.argsort(flat_unknown, kind="mergesort")
            flat_u_sorted = flat_unknown[order_u]
            uniq_v_u, start_u, count_u = np.unique(flat_u_sorted, return_index=True, return_counts=True)

            n_vox_u = uniq_v_u.size
            if n_vox_u:
                n_pts_per_vox_u = rng.integers(
                    self.points_per_voxel_min,
                    self.points_per_voxel_max + 1,
                    size=n_vox_u,
                    dtype=np.int32
                )
                total_u = int(n_pts_per_vox_u.sum())
                out_pts_u = np.empty((total_u, 3), dtype=np.float32)
                out_lbl_u = np.full((total_u,), UNKNOWN_LABEL, dtype=np.uint8)

                write = 0
                for i in range(n_vox_u):
                    s = int(start_u[i])
                    c = int(count_u[i])
                    e = s + c
                    n_pts = int(n_pts_per_vox_u[i])

                    idxs = order_u[s:e]
                    chosen = rng.choice(idxs, size=n_pts, replace=(c < n_pts))

                    out_pts_u[write:write + n_pts] = pts_unknown_ray[chosen]
                    write += n_pts

                points_unknown = out_pts_u
                labels_unknown = out_lbl_u

        n_occ, n_emp, n_unk = points_occ.shape[0], points_empty.shape[0], points_unknown.shape[0]
        print(f"   → Pools: occ={n_occ} pts, empty={n_emp} pts, unk={n_unk} pts (avant ratios)")

        # si pas de cible : retourne tout
        if target_total_points <= 0:
            pts_final = np.vstack([points_occ, points_empty, points_unknown]).astype(np.float32, copy=False)
            lbl_final = np.hstack([labels_occ, labels_empty, labels_unknown]).astype(np.uint8, copy=False)
            return pts_final, lbl_final

        # quotas
        n_occ_target = int(target_total_points * self.ratio_occ)
        n_emp_target = int(target_total_points * self.ratio_empty)
        n_unk_target = int(target_total_points - n_occ_target - n_emp_target)

        def sample_class(pts_c: np.ndarray, lbl_c: np.ndarray, target: int):
            if pts_c.shape[0] == 0 or target <= 0:
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)
            if pts_c.shape[0] <= target:
                return pts_c, lbl_c
            idx = rng.choice(pts_c.shape[0], size=target, replace=False)
            return pts_c[idx], lbl_c[idx]

        pts_occ_s, lbl_occ_s = sample_class(points_occ, labels_occ, n_occ_target)
        pts_emp_s, lbl_emp_s = sample_class(points_empty, labels_empty, n_emp_target)
        pts_unk_s, lbl_unk_s = sample_class(points_unknown, labels_unknown, n_unk_target)

        pts_final = np.vstack([pts_occ_s, pts_emp_s, pts_unk_s]).astype(np.float16, copy=False)
        lbl_final = np.hstack([lbl_occ_s, lbl_emp_s, lbl_unk_s]).astype(np.uint8, copy=False)

        print(f"   → Points finals (occ/empty/unk): "
            f"{pts_occ_s.shape[0]}/{pts_emp_s.shape[0]}/{pts_unk_s.shape[0]} "
            f"(total={pts_final.shape[0]:,} cible={target_total_points:,})")

        return pts_final, lbl_final






    # ---------- SAVE FRAME ----------

    def save_frame(self, frame_data: dict, frame_id: int, ref_position: dict):
        
        EMPTY_COLOR = (255, 255, 255)   # comme tu veux
        UNKNOWN_COLOR = (128, 128, 128) # comme tu veux
        

        
        import random
        formatted_id = f"{frame_id:06d}"

        pts = frame_data['points']
        lbl = frame_data['labels']

        # filtrage sémantique (uniquement sur les HITS, pas les empty sentinel)
        # NOTE: lbl ici = ObjTag CARLA (uint8) + sentinel 255
        if self.allowed_semantic_tags is not None:
            mask_empty = (lbl == LIDAR_EMPTY_SENTINEL)
            mask_hit = ~mask_empty
            if np.any(mask_hit):
                allowed = np.isin(lbl[mask_hit], np.array(self.allowed_semantic_tags, dtype=np.uint8))
                keep_hit_idx = np.where(mask_hit)[0][allowed]
            else:
                keep_hit_idx = np.array([], dtype=np.int64)

            keep_idx = np.concatenate([keep_hit_idx, np.where(mask_empty)[0]])
            keep_idx = np.unique(keep_idx)
            pts = pts[keep_idx]
            lbl = lbl[keep_idx]

        # nombre total de points final (implicit)
        target_total_points = random.randint(self.points_min_saved, self.points_max_saved)

        with SectionTimer(self.perf, "build_implicit_grid"):
            pts_final, lbl_final = self._build_implicit_from_points(
                pts,
                lbl,
                target_total_points=target_total_points
            )

        print(f"   → implicit occupancy: {len(pts_final):,} pts (cible={target_total_points:,})")

        # --- mapping complet: Empty/Unknown + 22 CARLA ---
        class_names = np.array(
            ["Empty", "Unknown"] + [name for (_id, name, _rgb) in CARLA_22],
            dtype=object
        )
        class_ids = np.array(
            [-2, -1] + [int(_id) for (_id, _name, _rgb) in CARLA_22],
            dtype=np.uint8
        )
        class_colors = np.array(
            [EMPTY_COLOR, UNKNOWN_COLOR] + [tuple(_rgb) for (_id, _name, _rgb) in CARLA_22],
            dtype=np.uint8
        )  # shape (24,3)

        # Sauvegarde NPZ
        points_path = os.path.join(self.output_dir, "points", f"frame_{formatted_id}.npz")
        with SectionTimer(self.perf, "save_points_npy"):
            np.savez(
                points_path,
                points=pts_final.astype(np.float16),
                labels=lbl_final.astype(np.uint8),   # IMPORTANT: -1/-2 possible
                class_names=class_names,             # toujours 24 classes
                class_ids=class_ids
            )

        # Sauvegarde des images
        saved_images = 0
        with SectionTimer(self.perf, "save_images_jpg"):
            for name, img in frame_data['images'].items():
                if img is not None:
                    img_path = os.path.join(self.output_dir, "images", f"frame_{formatted_id}_{name}.jpg")
                    cv2.imwrite(img_path, img)  # img est déjà en BGR
                    saved_images += 1

        # PREVIEW dense (sur les points finals occupancy implicite)
        preview_path = None
        if self.preview_interval > 0 and (frame_id % self.preview_interval == 0):
            preview_path = os.path.join(self.previews_dir, f"frame_{formatted_id}.jpg")
            with SectionTimer(self.perf, "preview_render"):
                self._render_preview(pts_final, lbl_final, frame_id, preview_path)
            print(f"📸 Preview généré: {preview_path}")

        meta = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'timestamp_sim': ref_position.get('timestamp_sim', None),
            'num_points_saved': int(len(pts_final)),
            'num_points_captured': int(len(frame_data['points'])),
            'num_images': saved_images,
            'scan_duration': frame_data.get('scan_duration', 0),
            'ticks': frame_data.get('ticks', 0),
            'position': ref_position,
            'reference_frame': 'ROBOT_CENTERED',
            'capture_mode': 'MULTI_EGO_ACCUM_ZSTACK_IMPLICIT_OCCUPANCY',
            'voxel_config': {
                'voxel_size': self.voxel_cfg.voxel_size,
                'x_range': self.voxel_cfg.x_range,
                'y_range': self.voxel_cfg.y_range,
                'z_range': self.voxel_cfg.z_range,
            },
            'preview_jpeg': preview_path,
            'weather_id': self.weather_id,
            'weather_name': self.fixed_weather_name,
            'points_path': points_path
        }

        meta_path = os.path.join(self.output_dir, "metadata", f"frame_{formatted_id}.json")
        with SectionTimer(self.perf, "save_metadata_json"):
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

        print(f" 💾 Frame {frame_id}: points={len(pts_final):,} "
            f"| preview={os.path.basename(preview_path) if preview_path else '—'}")


    # ---------- MAIN LOOP ----------

    def generate(self, max_frames=1000):
        if not self.connect():
            return

        self.sensor_manager = PersistentSensorManager(
            self.world,
            enable_blur=self.enable_blur,
            capture_points_target=self.capture_points_target,
            z_min=self.z_min, z_max=self.z_max, z_step=self.z_step,
            h_fov=self.h_fov, v_upper=self.v_upper, v_lower=self.v_lower,
            lidar_channels=self.lidar_channels,
            lidar_pps=self.lidar_pps,
            lidar_range=self.lidar_range,
            perf=self.perf,
            cam_height_noise_pct=self.cam_height_noise_pct,
            cam_angle_noise_pct=self.cam_angle_noise_pct,
            empty_points_per_hit=self.lidar_empty_points_per_hit,
        )

        max_frames = min(max_frames, len(self.positions))
        print("\n🚀 GÉNÉRATION DATASET OCCUPANCY IMPLICITE (multi-poses)")
        print(f"  Frames max: {max_frames}\n")

        start_time = time.time()
        position_count = 0
        try:
            for pos_idx in range(max_frames):
                position_count += 1
                base_pos = self.positions[pos_idx]
                ref_position = {
                    'location': base_pos['ego_location'],
                    'rotation': base_pos['ego_rotation'],
                    'timestamp_sim': base_pos['timestamp_sim']
                }
                print(f"\n📍 Position {position_count}/{max_frames}: "
                      f"x={ref_position['location']['x']:.2f}, "
                      f"y={ref_position['location']['y']:.2f}, "
                      f"z={ref_position['location']['z']:.2f}")

                # 1) création des capteurs à la première pose
                if pos_idx == 0:
                    start_transform = carla.Transform(
                        carla.Location(**ref_position['location']),
                        carla.Rotation(**ref_position['rotation'])
                    )
                    if not self.sensor_manager.create_sensors_once(start_transform):
                        print("❌ Impossible de créer les capteurs")
                        return

                # 2) repère robot T0 pour cette frame
                self.sensor_manager.set_reference_robot(ref_position)
                self.sensor_manager.move_cameras_to_position(ref_position)
                # 3) acteurs à T0
                actors_at_T0 = base_pos.get('other_actors', [])

                # 4) indices de la fenêtre ego
                ego_indices = self._ego_window_indices(pos_idx)
                lo = ego_indices[0]
                hi = ego_indices[-1]
                print(f"🪟 ego window pos_idx={pos_idx} -> [{lo} .. {hi}] (len={len(ego_indices)})")
                print(f"   back_count={sum(j < pos_idx for j in ego_indices)} | "
                    f"center={pos_idx in ego_indices} | "
                    f"fwd_count={sum(j > pos_idx for j in ego_indices)}")
                # 5) nouvelle accumulation
                self.sensor_manager.start_new_accumulation(self.capture_points_target)

                max_ticks_total = self.max_ticks_per_pose * len(ego_indices)
                ticks_done = 0
                points_by_j = defaultdict(int)
                ticks_by_j = defaultdict(int)
                # 6) boucle d'accum : on balaye toutes les poses tant que quota non atteint
                while (not self.sensor_manager.lidar_accumulator.is_complete()
                       and ticks_done < max_ticks_total):

                    for j in ego_indices:
                        pose_j = self.positions[j]
                        pose_j_dict = {
                            'location': pose_j['ego_location'],
                            'rotation': pose_j['ego_rotation'],
                            'timestamp_sim': pose_j['timestamp_sim']
                        }

                        has_actor_here = self._pose_has_actor_at_T0(
                            pose_j,
                            actors_at_T0,
                            self.proximity_radius
                        )

                        # Layout clear vs complet
                        if (not has_actor_here) and (self.lidar_layout_if_clear is not None):
                            self.sensor_manager.set_lidar_layout(self.lidar_layout_if_clear)
                            if self.randomize_clear_poses:
                                self.sensor_manager.randomize_lidar_params()
                        else:
                            self.sensor_manager.set_lidar_layout(None)

                        self.sensor_manager.move_lidars_to_position(pose_j_dict)
                        self.sensor_manager.current_pose_j = j
                        self.world.tick()
                        ticks_done += 1

                        if self.sensor_manager.lidar_accumulator.is_complete() or ticks_done >= max_ticks_total:
                            break

                # 7) récupération de l'accumulation
                frame_data = self.sensor_manager.capture_current_frame(self.fixed_weather_name)
                counts = self.sensor_manager.lidar_accumulator.get_tag_counts()
                items = sorted(counts.items(), key=lambda x: x[0])
                total = sum(v for _, v in items)
                print("📌 Points par pose j:")
                for jj, n in items:
                    rel = "T0" if jj == pos_idx else ("BACK" if jj < pos_idx else "FWD")
                    print(f"   j={jj} [{rel}] -> {n} pts ({100.0*n/max(total,1):.1f}%)")
                if frame_data and len(frame_data['points']) > 0:
                    unique_frame_id = self.global_frame_counter
                    with SectionTimer(self.perf, "save_frame_total"):

                        # self.save_frame(frame_data, unique_frame_id, ref_position)
                        self.writer.queue.put((frame_data, unique_frame_id, ref_position)) # Thread de sauvegarde 
                    self.global_frame_counter += 1
                    del frame_data
                    gc.collect()
                else:
                    print(" ⚠️ Aucune donnée LiDAR suffisante pour cette frame")

                # Logging progression
                if position_count % 5 == 0 and position_count > 0:
                    elapsed = time.time() - start_time
                    fps = self.global_frame_counter / elapsed if elapsed > 0 else 0
                    remaining_positions = max_frames - position_count
                    eta = remaining_positions / fps if fps > 0 else 0
                    print(f"{'=' * 50}"
                          f" PROGRESSION: Pos {position_count}/{max_frames}"
                          f" | Frames: {self.global_frame_counter}"
                          f" | FPS: {fps:.2f}"
                          f" | ETA: {eta/60:.1f} min"
                          f"{'=' * 50}")

        except KeyboardInterrupt:
            print("⚠️ Interruption par l'utilisateur")
        except Exception as e:
            print(f"❌ Erreur durant la génération: {e}")
            traceback.print_exc()
        finally:
            print("🧹 Nettoyage final...")
            if self.sensor_manager:
                self.sensor_manager.cleanup()
            self._deep_cleanup_world()
            try:
                self.weather_manager.apply_by_id(0)
            except Exception:
                pass
            total_time = time.time() - start_time
            print(f"{'=' * 60}✅ GÉNÉRATION TERMINÉE"
                  f" | Référentiel: CENTRÉ SUR LE ROBOT"
                  f" | Positions traitées: {position_count}"
                  f" | Frames totales: {self.global_frame_counter}"
                  f" | Dataset: {self.output_dir}/"
                  f" | Temps total: {total_time/60:.1f} min"
                  f"{'=' * 60}")
            self.perf.global_report()


# ==========================
# CLI
# ==========================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='CARLA Robot Dataset - Implicit Occupancy (multi-ego, Z-stack LiDARs)'
    )
    parser.add_argument('--frames', type=int, default=10000, help='Nombre de positions à traiter')
    parser.add_argument('--output', type=str, default='CarlaUE4/carla_occ3d_dataset/Town01_mod0', help='Dossier sortie')
    parser.add_argument('--previews-dir-name', type=str, default='previews', help="Sous-dossier d\'aperçus JPEG")
    parser.add_argument('--no-blur', action='store_true', help='Désactiver le flou sur les caméras')
    parser.add_argument('--preview-interval', type=int, default=50, help='Intervalle de frames pour la preview')
    parser.add_argument('--tm-port', type=int, default=8000, help='Port du Traffic Manager')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    parser.add_argument('--z-min', type=float, default=0.5, help='Hauteur min relative LiDAR (m)')
    parser.add_argument('--z-max', type=float, default=3, help='Hauteur max relative LiDAR (m)')
    parser.add_argument('--z-step', type=float, default=4, help='Pas entre LiDARs (m)')
    parser.add_argument('--h-fov', type=float, default=360.0, help='FOV horizontal (deg)')
    parser.add_argument('--v-upper', type=float, default=40.0, help='FOV vertical haut (deg)')
    parser.add_argument('--v-lower', type=float, default=-40.0, help='FOV vertical bas (deg)')
    parser.add_argument('--lidar-channels', type=int, default=256, help='Canaux LiDAR')
    parser.add_argument('--lidar-pps', type=int, default=500_000, help='Points/seconde LiDAR')
    parser.add_argument('--lidar-range', type=float, default=1000, help='Portée LiDAR (m)')

    parser.add_argument('--map', type=str, default='Town10HD_Opt', help='Carte CARLA')
    parser.add_argument('--trajectory-json', type=str,
                        default="carla_trajectories/Town10HD_Opt_fast_20251010_194958_veh25.json",
                        help='Trajectoire à rejouer')
    parser.add_argument('--weather-id', type=int, default=0,
                        help='0 clear_noon | 1 overcast_morning | ...')
    parser.add_argument('--profile', action='store_true', default=True, help='Activer le profiling')
    parser.add_argument('--capture-points', type=int, default=800_000,
                        help='Quota de points LiDAR à capturer par frame (hits + empty)')
    parser.add_argument('--points-min-saved', type=int, default=55_000,
                        help='Nb min de points occupancy sauvegardés par frame')
    parser.add_argument('--points-max-saved', type=int, default=60_000,
                        help='Nb max de points occupancy sauvegardés par frame')
    parser.add_argument('--cube-size-m', type=float, default=0.005,
                        help='Taille “visuelle” du marker dans la preview (m)')
    parser.add_argument('--cam-height-noise-pct', type=float, default=15.0,
                        help='Bruit hauteur caméra en % (±)')
    parser.add_argument('--cam-angle-noise-pct', type=float, default=10.0,
                        help='Bruit angles caméra (pitch/yaw/roll) en % (±)')

    # fenêtre ego
    parser.add_argument('--window-back', type=int, default=1, help='Nombre de poses passées de l’ego à charger')
    parser.add_argument('--window-forward', type=int, default=1, help='Nombre de poses futures de l’ego à charger')
    parser.add_argument('--proximity-radius', type=float, default=0.2,
                        help='Rayon pour détecter piéton/véhicule à T0')
    parser.add_argument('--max-ticks-per-pose', type=int, default=50,
                        help='Ticks max à laisser CARLA tourner pour chaque pose')
    parser.add_argument('--allowed-semantic-tags', type=str, default='',
                        help='Classes sémantiques CARLA à garder pour les HITS, ex: "7,10" pour road+vehicle')
    parser.add_argument('--lidar-layout-if-clear', type=str, default='',
                        help='Liste de hauteurs LiDAR à activer quand zone clear, ex: "0.4,0.8,1.2"')
    parser.add_argument('--no-randomize-clear', action='store_true',
                        help='Ne pas randomizer les LiDAR sur les poses “clear”')

    # occupancy implicite
    parser.add_argument('--implicit-voxel-size', type=float, default=0.5,
                        help="Taille des voxels occupancy implicite (m)")
    parser.add_argument('--implicit-points-per-voxel-min', type=int, default=1,
                        help="Nb min de points par voxel")
    parser.add_argument('--implicit-points-per-voxel-max', type=int, default=2,
                        help="Nb max de points par voxel")
    parser.add_argument('--implicit-ratio-occ', type=float, default=0.8,
                        help="Ratio approx de points occupés dans le dataset implicite")
    parser.add_argument('--implicit-ratio-empty', type=float, default=0.1,
                        help="Ratio approx de points empty")
    parser.add_argument('--implicit-ratio-unknown', type=float, default=0.1,
                        help="Ratio approx de points unknown")
    parser.add_argument('--voxel-keep-ratio-empty', type=float, default=0.1,
                        help="Proportion de voxels empty gardés")
    parser.add_argument('--voxel-keep-ratio-unknown', type=float, default=0.1,
                        help="Proportion de voxels unknown gardés")
    parser.add_argument('--implicit-empty-points-per-hit', type=int, default=4,
                        help="Nb de points empty à tirer par hit LiDAR dans les callbacks")

    args = parser.parse_args()

    allowed_semantic_tags = None
    if args.allowed_semantic_tags:
        allowed_semantic_tags = [int(x.strip()) for x in args.allowed_semantic_tags.split(',') if x.strip()]

    lidar_layout_if_clear = None
    if args.lidar_layout_if_clear:
        lidar_layout_if_clear = [float(x.strip()) for x in args.lidar_layout_if_clear.split(',') if x.strip()]

    print("=" * 60)
    print("🤖 CARLA ROBOT DATASET GENERATOR — IMPLICIT OCCUPANCY")
    print(f" Carte: {args.map}")
    print(f" Aperçus JPEG → {os.path.join(args.output, args.previews_dir_name)}")
    print(f" Preview toutes les {args.preview_interval} frames")
    print(f" window_back={args.window_back}, window_forward={args.window_forward}")
    print(f" proximity_radius={args.proximity_radius}")
    print(f" lidar_layout_if_clear={lidar_layout_if_clear}")
    print(f" allowed_semantic_tags={allowed_semantic_tags}")
    print("=" * 60)

    generator = FastDatasetGenerator(
        output_dir=args.output,
        enable_blur=not args.no_blur,
        preview_interval=args.preview_interval,
        tm_port=args.tm_port,
        seed=args.seed,
        z_min=args.z_min, z_max=args.z_max, z_step=args.z_step,
        h_fov=args.h_fov, v_upper=args.v_upper, v_lower=args.v_lower,
        lidar_channels=args.lidar_channels, lidar_pps=args.lidar_pps, lidar_range=args.lidar_range,
        previews_dir_name=args.previews_dir_name,
        map_name=args.map,
        trajectory_json=args.trajectory_json,
        weather_id=args.weather_id,
        profile=args.profile,
        capture_points_target=args.capture_points,
        points_min_saved=args.points_min_saved,
        points_max_saved=args.points_max_saved,
        cube_size_m=args.cube_size_m,
        cam_height_noise_pct=args.cam_height_noise_pct,
        cam_angle_noise_pct=args.cam_angle_noise_pct,
        window_back=args.window_back,
        window_forward=args.window_forward,
        proximity_radius=args.proximity_radius,
        lidar_layout_if_clear=lidar_layout_if_clear,
        allowed_semantic_tags=allowed_semantic_tags,
        max_ticks_per_pose=args.max_ticks_per_pose,
        randomize_clear_poses=not args.no_randomize_clear,
        implicit_voxel_size=args.implicit_voxel_size,
        implicit_points_per_voxel_min=args.implicit_points_per_voxel_min,
        implicit_points_per_voxel_max=args.implicit_points_per_voxel_max,
        implicit_ratio_occ=args.implicit_ratio_occ,
        implicit_ratio_empty=args.implicit_ratio_empty,
        implicit_ratio_unknown=args.implicit_ratio_unknown,
        voxel_keep_ratio_empty=args.voxel_keep_ratio_empty,
        voxel_keep_ratio_unknown=args.voxel_keep_ratio_unknown,
        lidar_empty_points_per_hit=args.implicit_empty_points_per_hit,
    )
    generator.generate(max_frames=args.frames)


if __name__ == "__main__":
    main()
