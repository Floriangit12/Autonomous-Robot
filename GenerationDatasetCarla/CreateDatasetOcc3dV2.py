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
    * on voxelise la zone [-16,16] x [-16,16] x [-2,4] en cubes de 0.5 m
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
    z_range: Tuple[float, float] = (-2.0, 4.0)
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
        for _ in range(1):
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
        window_back: int = 2,
        window_forward: int = 2,
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

        # Cam config :
        self.cam_capture_w = 2048
        self.cam_capture_h = 1536

        # Résolution finale (réseau)
        self.cam_out_w = 512
        self.cam_out_h = 384

        self._img_q = Queue(maxsize=256)  # buffer, évite les stalls
        self._img_workers = []
        self._img_workers_n = max(2, (os.cpu_count() or 4) // 2)

        for _ in range(self._img_workers_n):
            t = Thread(target=self._image_worker, daemon=True)
            t.start()
            self._img_workers.append(t)



        self.window_back = int(window_back)
        self.window_forward = int(window_forward)
        self.lidar_slot_ids = list(range(-self.window_back, self.window_forward + 1))
        self.lidar_rigs: Dict[int, List[carla.Actor]] = {}      # slot -> list[actors]
        self.lidar_actor_to_slot: Dict[int, int] = {}           # actor_id -> slot
        self.current_pose_j = None
        self.window_back = int(window_back)
        self.window_forward = int(window_forward)
        self.lidar_slot_ids = list(range(-self.window_back, self.window_forward + 1))
        self.bank_global_offset_world = (0.0, 0.0, 0.0)

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

    def _enqueue_image(self, cam_name: str, raw_bytes: bytes, w: int, h: int):
        """
        Enqueue non-bloquant : si la queue est pleine, on drop l'image
        (mieux: perdre une frame que bloquer toute la sim).
        """
        try:
            self._img_q.put_nowait((cam_name, raw_bytes, w, h))
        except Exception:
            # Queue pleine -> drop
            pass

    def _image_worker(self):
        """
        Worker: raw BGRA -> BGR -> resize -> store (uint8) en 512x384
        """
        while True:
            cam_name, raw_bytes, w, h = self._img_q.get()
            try:
                # raw BGRA (CARLA Raw)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                arr = arr.reshape((h, w, 4))

                # BGR (OpenCV) : slice view, resize fait la copie finale
                bgr = arr[:, :, :3]

                # Downsample haute qualité + rapide (AREA = bon pour réduire)
                small = cv2.resize(
                    bgr,
                    (self.cam_out_w, self.cam_out_h),
                    interpolation=cv2.INTER_AREA
                )

                # Store (petit, contiguous)
                with self.lock:
                    self.camera_data[cam_name] = small  # uint8 (H,W,3)
                    self.camera_received[cam_name] = True
            except Exception:
                pass
            finally:
                self._img_q.task_done()



    @staticmethod
    def _T_translate(dx: float, dy: float, dz: float) -> np.ndarray:
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = dx
        T[1, 3] = dy
        T[2, 3] = dz
        return T
    
    def _to_robot_frame(self, pts_local: np.ndarray, sensor_transform: carla.Transform):
        if pts_local is None or pts_local.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # sensor -> world
        M_sw = np.array(sensor_transform.get_matrix(), dtype=np.float32)

        # world -> robot(T0)
        ref = self.reference_robot_transform
        robot_tf = carla.Transform(
            carla.Location(**ref["location"]),
            carla.Rotation(**ref["rotation"])
        )
        M_wr = np.array(robot_tf.get_inverse_matrix(), dtype=np.float32)

        # undo global bank offset in WORLD
        dxo, dyo, dzo = self.bank_global_offset_world
        M_undo = np.eye(4, dtype=np.float32)
        M_undo[0, 3] = -dxo
        M_undo[1, 3] = -dyo
        M_undo[2, 3] = -dzo

        # full chain: sensor -> world(true) -> robot(T0)
        M_sr = M_wr @ M_undo @ M_sw

        # apply with column-vector convention using transpose trick
        pts4 = np.ones((pts_local.shape[0], 4), dtype=np.float32)
        pts4[:, :3] = pts_local.astype(np.float32, copy=False)

        pts_robot4 = (M_sr @ pts4.T).T   # colonne -> colonne
        return pts_robot4[:, :3]



    # def _to_robot_frame(self, pts_local: np.ndarray, sensor_transform: carla.Transform):
    #     if pts_local is None or pts_local.size == 0:
    #         return np.zeros((0, 3), dtype=np.float32)

    #     # sensor -> world
    #     M_sw = np.array(sensor_transform.get_matrix(), dtype=np.float32)

    #     # world -> robot(T0)
    #     ref = self.reference_robot_transform
    #     robot_tf = carla.Transform(
    #         carla.Location(**ref["location"]),
    #         carla.Rotation(**ref["rotation"])
    #     )
    #     M_wr = np.array(robot_tf.get_inverse_matrix(), dtype=np.float32)

    #     # undo global bank offset (world)
    #     dxo, dyo, dzo = self.bank_global_offset_world
    #     M_undo = np.eye(4, dtype=np.float32)
    #     M_undo[0, 3] = -dxo
    #     M_undo[1, 3] = -dyo
    #     M_undo[2, 3] = -dzo

    #     # sensor -> robot
    #     M_sr = M_wr @ M_undo @ M_sw

    #     pts4 = np.ones((pts_local.shape[0], 4), dtype=np.float32)
    #     pts4[:, :3] = pts_local.astype(np.float32, copy=False)

    #     pts_robot4 = (M_sr @ pts4.T).T
    #     return pts_robot4[:, :3]



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
                [self.world.tick() for _ in range(1)]
                # time.sleep(0.1)
            return cleaned

    # def apply_camera_blur(self, image, weather_preset=None):
    #     if not self.enable_blur:
    #         return image
    #     blur_intensity = 1
    #     if weather_preset:
    #         if 'foggy' in weather_preset:
    #             blur_intensity = 5
    #         elif 'rainy' in weather_preset or 'storm' in weather_preset:
    #             blur_intensity = 3
    #         elif 'night' in weather_preset:
    #             blur_intensity = 2
    #     k = (2 * blur_intensity + 1, 2 * blur_intensity + 1)
    #     with SectionTimer(self.perf, "camera_blur"):
    #         blurred = cv2.GaussianBlur(image, k, 0)
    #         if weather_preset and ('rainy' in weather_preset):
    #             noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    #             blurred = cv2.add(blurred, noise)
    #     return blurred


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


    # def move_lidars_to_position(self, position: dict):
    #     """Déplace UNIQUEMENT les LiDARs à une pose j."""
    #     ego_tf = carla.Transform(
    #         carla.Location(**position['location']),
    #         carla.Rotation(**position['rotation'])
    #     )

    #     for lidar, cfg in zip(self.lidars, self.LIDAR_CONFIGS):
    #         if lidar and lidar.is_alive:
    #             sensor_loc_local = carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz'])
    #             sensor_loc_world = ego_tf.transform(sensor_loc_local)
    #             sensor_rot_world = carla.Rotation(
    #                 pitch=ego_tf.rotation.pitch,
    #                 yaw=ego_tf.rotation.yaw,
    #                 roll=ego_tf.rotation.roll
    #             )
    #             lidar.set_transform(carla.Transform(sensor_loc_world, sensor_rot_world))




    def create_sensors_once(self, start_transform: carla.Transform):
        if self.sensors_created:
            return True

        print("🔧 Création des capteurs (BANK poses [-N..N] + Z-STACK + cams)...")
        self.cleanup_orphan_sensors()

        try:
            with SectionTimer(self.perf, "create_sensors_total"):
                bp_library = self.world.get_blueprint_library()

                # =========================
                # LiDAR BANK (2N+1 rigs)
                # =========================
                with SectionTimer(self.perf, "create_lidars_bank"):
                    self.lidar_rigs = {}
                    self.lidars = []

                    # On crée un rig par slot s dans [-N..N]
                    for s in self.lidar_slot_ids:
                        rig_list = []

                        for i, cfg in enumerate(self.LIDAR_CONFIGS):
                            lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
                            lidar_bp.set_attribute('channels', str(cfg['channels']))
                            lidar_bp.set_attribute('points_per_second', str(cfg['pps']))
                            lidar_bp.set_attribute('rotation_frequency', '6')
                            lidar_bp.set_attribute('range', str(cfg['range']))
                            lidar_bp.set_attribute('upper_fov', str(cfg['upper_fov']))
                            lidar_bp.set_attribute('lower_fov', str(cfg['lower_fov']))
                            lidar_bp.set_attribute('horizontal_fov', str(cfg['horizontal_fov']))
                            try:
                                lidar_bp.set_attribute('role_name', 'virtual_sensor')
                            except Exception:
                                pass

                            # Spawn provisoire : on les met tous sur start_transform, on les bougera ensuite
                            tf = carla.Transform(
                                start_transform.location + carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz']),
                                carla.Rotation()
                            )
                            lidar = self.world.spawn_actor(lidar_bp, tf)

                            self.sensor_ids.add(lidar.id)
                            self.lidars.append(lidar)
                            rig_list.append(lidar)
                            self.lidar_actor_to_slot[lidar.id] = int(s)

                            # --- callback ---
                            def make_cb(sensor, cfg_index=i, slot_id=int(s)):
                                def _cb(data):
                                    # si lidar désactivé (par hauteur) on ignore
                                    if not self.active_lidar_mask[cfg_index]:
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

                                        # hits en repère robot (T0)
                                        pts_robot_hits = self._to_robot_frame(pts_local, sensor_tf)
                                        lbl_hits = arr['ObjTag'].astype(np.uint8)

                                        # EMPTY avant hit
                                        pts_robot_empty = np.zeros((0, 3), dtype=np.float32)
                                        lbl_empty = np.zeros((0,), dtype=np.uint8)

                                        if self.empty_points_per_hit > 0:
                                            n_hits = len(pts_local)
                                            k = self.empty_points_per_hit
                                            t = np.random.rand(n_hits, k).astype(np.float32) * 0.98
                                            pts_empty_local = (pts_local[:, None, :] * t[..., None]).reshape(-1, 3)
                                            pts_robot_empty = self._to_robot_frame(pts_empty_local, sensor_tf)
                                            lbl_empty = np.full((len(pts_robot_empty),), LIDAR_EMPTY_SENTINEL, dtype=np.uint8)

                                        # UNKNOWN derrière hit (optionnel)
                                        pts_robot_unk = np.zeros((0, 3), dtype=np.float32)
                                        lbl_unk = np.zeros((0,), dtype=np.uint8)

                                        max_range_m = 16.0
                                        d = np.linalg.norm(pts_local, axis=1).astype(np.float32)
                                        valid = (d > 1e-3) & (d < max_range_m - 1e-3)
                                        if np.any(valid):
                                            d_v = d[valid]
                                            pts_hit_v = pts_local[valid]
                                            s_max = (max_range_m / d_v)
                                            s_min = 1.02
                                            s_hi = np.maximum(s_max, s_min + 1e-3)
                                            r = np.random.rand(s_hi.shape[0]).astype(np.float32)
                                            s = s_min + r * (s_hi - s_min)
                                            pts_unk_local = pts_hit_v * s[:, None]
                                            pts_robot_unk = self._to_robot_frame(pts_unk_local, sensor_tf)
                                            lbl_unk = np.full((len(pts_robot_unk),), LIDAR_UNKNOWN_SENTINEL, dtype=np.uint8)

                                        pts_concat = np.vstack([pts_robot_hits, pts_robot_empty, pts_robot_unk])
                                        lbl_concat = np.hstack([lbl_hits, lbl_empty, lbl_unk])

                                        # tag = slot_id (pose relative à T0)
                                        # (tu peux aussi stocker le vrai index j absolu ailleurs)
                                        self.lidar_accumulator.add(pts_concat, lbl_concat, tag=slot_id)

                                    except Exception:
                                        print(f"Erreur LiDAR (id={sensor.id}):")
                                        traceback.print_exc()
                                return _cb

                            lidar.listen(make_cb(lidar))

                        self.lidar_rigs[int(s)] = rig_list

                # =========================
                # Cameras (inchangé chez toi)
                # =========================
                with SectionTimer(self.perf, "create_cameras"):
                    for cfg in self.CAMERA_CONFIGS:
                        cam_bp = bp_library.find('sensor.camera.rgb')
                        # --- RÉSOLUTIONS ---
                        # Version optimisée pour BiFPN (Multiple de 32) basée sur 4080x3072
                        cam_bp.set_attribute('image_size_x', str(self.cam_capture_w))
                        cam_bp.set_attribute('image_size_y', str(self.cam_capture_h))
                        cam_bp.set_attribute('bloom_intensity', '0.0')
                        cam_bp.set_attribute('lens_flare_intensity', '0.0')
                        # --- OPTIQUE ---
                        # Utilisation du Horizontal FOV exact de votre fiche technique
                        cam_bp.set_attribute('fov', '71.4')

                        # # --- PHYSIQUE DU CAPTEUR ---
                        # # Simulation de l'ouverture f/1.9
                        # cam_bp.set_attribute('exposure_mode', 'Manual')
                        cam_bp.set_attribute('fstop', '1.9')
                        cam_bp.set_attribute('shutter_speed', '500')   # 1/10s

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
                                    raw = bytes(image.raw_data)  # copie nécessaire: buffer CARLA volatile
                                    self._enqueue_image(name, raw, image.width, image.height)
                                except Exception:
                                    pass
                            return _cb

                        cam.listen(make_cam_cb(cfg['name']))

                self.sensors_created = True
                print(f"✅ {len(self.lidars)} LiDARs et {len(self.cameras)} caméras créés")
                for _ in range(1):
                    self.world.tick()
                # time.sleep(0.1)
                return True

        except Exception:
            print("❌ Erreur création capteurs:")
            traceback.print_exc()
            return False


    def move_all_lidar_rigs(self, slot_to_pose: Dict[int, dict], global_offset_world=(0.0, 0.0, 0.0)):
        """
        slot_to_pose: dict slot s in [-N..N] -> pose dict {'location','rotation',...}
        global_offset_world: (dx,dy,dz) ajouté en world après projection
        """
        self.bank_global_offset_world = tuple(map(float, global_offset_world))
        dxo, dyo, dzo = map(float, global_offset_world)

        for slot, rig in self.lidar_rigs.items():
            pose = slot_to_pose.get(int(slot), None)
            if pose is None:
                continue

            ego_tf = carla.Transform(
                carla.Location(**pose['location']),
                carla.Rotation(**pose['rotation'])
            )

            for lidar, cfg in zip(rig, self.LIDAR_CONFIGS):
                if not (lidar and lidar.is_alive):
                    continue

                sensor_loc_local = carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz'])
                sensor_loc_world = ego_tf.transform(sensor_loc_local)

                # offset global en WORLD
                sensor_loc_world.x += dxo
                sensor_loc_world.y += dyo
                sensor_loc_world.z += dzo

                sensor_rot_world = carla.Rotation(
                    pitch=ego_tf.rotation.pitch,
                    yaw=ego_tf.rotation.yaw,
                    roll=ego_tf.rotation.roll
                )
                lidar.set_transform(carla.Transform(sensor_loc_world, sensor_rot_world))

    def _actors_near_any_lidar(self, min_dist: float = 1.0) -> bool:
        """
        True si un véhicule/piéton est trop proche d'au moins un lidar.
        min_dist en mètres
        """
        actors = self.world.get_actors()
        # filtre simple
        dyn = [a for a in actors if ('vehicle.' in a.type_id) or ('walker.pedestrian' in a.type_id)]
        if not dyn:
            return False

        min_d2 = float(min_dist) * float(min_dist)

        # positions lidars
        lidar_locs = []
        for lidar in self.lidars:
            if lidar and lidar.is_alive:
                t = lidar.get_transform()
                lidar_locs.append((t.location.x, t.location.y, t.location.z))

        if not lidar_locs:
            return False

        for a in dyn:
            loc = a.get_transform().location
            ax, ay, az = loc.x, loc.y, loc.z
            for lx, ly, lz in lidar_locs:
                dx = lx - ax
                dy = ly - ay
                dz = lz - az
                if (dx*dx + dy*dy + dz*dz) < min_d2:
                    return True
        return False

    def find_safe_global_offset(self,
                                slot_to_pose: Dict[int, dict],
                                min_dist: float = 1.0,
                                max_tries: int = 20,
                                xy_radius: float = 1.5,
                                z_offset: float = 0.0) -> Tuple[float, float, float]:
        """
        Cherche un offset global world (dx,dy,dz) qui évite la proximité.
        """
        # 1) try offset = 0
        self.move_all_lidar_rigs(slot_to_pose, global_offset_world=(0.0, 0.0, 0.0))
        self.world.tick()
        if not self._actors_near_any_lidar(min_dist=min_dist):
            return (0.0, 0.0, 0.0)

        # 2) random offsets
        for _ in range(int(max_tries)):
            ang = random.random() * 2.0 * math.pi
            rad = random.random() * float(xy_radius)
            dx = math.cos(ang) * rad
            dy = math.sin(ang) * rad
            dz = float(z_offset)

            self.move_all_lidar_rigs(slot_to_pose, global_offset_world=(dx, dy, dz))
            self.world.tick()

            if not self._actors_near_any_lidar(min_dist=min_dist):
                return (dx, dy, dz)

        # fallback
        return (0.0, 0.0, 0.0)

    def randomize_all_lidars_params(
        self,
        loc_jitter_x=(-0.2,0.2),
        loc_jitter_y=(-0.2, 0.2),
        loc_jitter_z=(-0.2, 0.2),
        pitch_jitter=(-5.0, 5.0),
        yaw_jitter=(-10.0, 15.0),
        roll_jitter=(-5.0, 5.0),
        channels_range=(512, 1024),
        upper_fov=(40.0, 60.0),
        lower_fov=(-60.0, -40.0),
        pps_range=(500_000, 2_100_000),
        rotation_frequency='6',
    ):
        for lidar in self.lidars:
            if not (lidar and lidar.is_alive):
                continue

            tf = lidar.get_transform()
            loc = tf.location
            rot = tf.rotation

            # jitter WORLD (si tu veux LOCAL, dis-le et je te donne la version)
            loc.x += random.uniform(*loc_jitter_x)
            loc.y += random.uniform(*loc_jitter_y)
            loc.z += random.uniform(*loc_jitter_z)

            rot.yaw += random.uniform(*yaw_jitter)
            rot.pitch += random.uniform(*pitch_jitter)
            rot.roll += random.uniform(*roll_jitter)
            
            lidar.set_transform(carla.Transform(loc, rot))

            # ATTENTION: selon CARLA, ces set_attribute peuvent être ignorés après spawn
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
                lidar.set_attribute('rotation_frequency', str(rotation_frequency))
            except Exception:
                pass

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
                        images[name] = img #self.apply_camera_blur(img.copy(), weather_preset)
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
        window_back: int = 2,
        window_forward: int = 2,
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
                    settings.fixed_delta_seconds = 0.2
                    settings.no_rendering_mode = False
                    
                    # 3. Configurer le Substepping pour garder une physique parfaite
                    settings.substepping = True
                    settings.max_substep_delta_time = 0.01
                    settings.max_substeps = 50
                    
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
            for _ in range(1):
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

    import time

    def _build_implicit_from_points(
        self,
        pts_robot: np.ndarray,
        lbl_raw: np.ndarray,
        target_total_points: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimisations appliquées (focus sur tes timings):
        - EMPTY / UNKNOWN: suppression de la boucle "par voxel" => sampling vectorisé
            (approximation acceptée: sampling AVEC remise dans chaque voxel, donc pas de replace=False)
        - OCC: garde la boucle (déjà rapide chez toi) + optimisation choice(replace=True)->integers
        - OCC label: mapping compressé (pas d'alloc full-grid)
        - instrumentation temps (perf_counter)
        """
        import time
        import numpy as np

        EMPTY_LABEL = 254
        UNKNOWN_LABEL = 253

        def tic():
            return time.perf_counter()

        def toc(t0, name):
            dt = time.perf_counter() - t0
            print(f"[TIME] {name:<22s}: {dt:.4f}s")

        t_global = tic()

        if pts_robot is None or pts_robot.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

        vx = self.voxel_cfg
        vs = float(vx.voxel_size)
        x_min, x_max = vx.x_range
        y_min, y_max = vx.y_range
        z_min, z_max = vx.z_range

        pts_robot = np.asarray(pts_robot, dtype=np.float32)
        lbl_raw = np.asarray(lbl_raw)

        # ---------------- ROI ----------------
        t0 = tic()
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
        toc(t0, "ROI filtering")

        # ---------------- SPLIT ----------------
        t0 = tic()
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
        toc(t0, "split classes")

        nx, ny, nz = vx.grid_shape
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

        # ---------------- VOXEL IDS ----------------
        t0 = tic()
        flat_hit = voxel_flat_ids(pts_hits)
        flat_empty = voxel_flat_ids(pts_empty)
        flat_unknown = voxel_flat_ids(pts_unknown_ray)
        toc(t0, "voxelization")

        # ---------------- OCC LABEL MAP (compressé) ----------------
        t0 = tic()
        uniq_hit = np.zeros((0,), dtype=np.int64)
        hit_labels_per_uniq = np.zeros((0,), dtype=np.uint8)
        if flat_hit.size:
            uniq_hit, first_idx = np.unique(flat_hit, return_index=True)
            hit_labels_per_uniq = lbl_hits_id[first_idx]
        toc(t0, "occ label map")

        rng = np.random.default_rng()

        # ================= OCC (boucle voxel, déjà rapide chez toi) =================
        t0_occ = tic()
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

                R = self.occ_upsample_radius
                p_up = self.occ_upsample_prob

                base_total = int(n_pts_per_vox.sum())
                if R > 0.0 and p_up > 0.0 and hasattr(self, "_occ_offset_bank"):
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

                    # OPT: replace=True -> integers
                    if c < n_pts:
                        chosen = idxs[rng.integers(0, c, size=n_pts, dtype=np.int32)]
                    else:
                        chosen = rng.choice(idxs, size=n_pts, replace=False)

                    pts_vox = pts_hits[chosen]

                    # label OCC via mapping compressé (uniq_v ⊂ uniq_hit)
                    pos = np.searchsorted(uniq_hit, v)
                    lbl_vox = hit_labels_per_uniq[pos] if uniq_hit.size else UNKNOWN_LABEL

                    out_pts[write:write + n_pts] = pts_vox
                    out_lbl[write:write + n_pts] = lbl_vox
                    base_slice_start = write
                    write += n_pts

                    # upsample OCC (inchangé fonctionnellement)
                    if extra_est > 0:
                        if p_up >= 1.0:
                            n_add = n_pts
                            add_idx_local = np.arange(n_pts, dtype=np.int32)
                        else:
                            n_add = int(rng.binomial(n_pts, p_up))
                            if n_add <= 0:
                                continue
                            add_idx_local = rng.choice(n_pts, size=n_add, replace=False)

                        base_slice = slice(base_slice_start, base_slice_start + n_pts)
                        centers = out_pts[base_slice][add_idx_local]

                        off_idx = rng.integers(0, K, size=n_add, dtype=np.int32)
                        new_pts = centers + self._occ_offset_bank[off_idx]

                        np.clip(new_pts[:, 0], x_min, x_max, out=new_pts[:, 0])
                        np.clip(new_pts[:, 1], y_min, y_max, out=new_pts[:, 1])
                        np.clip(new_pts[:, 2], z_min, z_max, out=new_pts[:, 2])

                        out_pts[write:write + n_add] = new_pts
                        out_lbl[write:write + n_add] = lbl_vox
                        write += n_add

                points_occ = out_pts[:write]
                labels_occ = out_lbl[:write]

        toc(t0_occ, "OCC loop")

        # ================= EMPTY (vectorisé, approx: AVEC remise par voxel) =================
        t0_empty = tic()
        points_empty = np.zeros((0, 3), dtype=np.float16)
        labels_empty = np.zeros((0,), dtype=np.uint8)

        if pts_empty.size:
            order_e = np.argsort(flat_empty, kind="mergesort")
            flat_e_sorted = flat_empty[order_e]
            uniq_v_e, start_e, count_e = np.unique(flat_e_sorted, return_index=True, return_counts=True)

            n_vox_e = int(uniq_v_e.size)
            if n_vox_e:
                n_pts_per_vox_e = rng.integers(
                    self.points_per_voxel_min,
                    self.points_per_voxel_max + 1,
                    size=n_vox_e,
                    dtype=np.int32
                )

                total_e = int(n_pts_per_vox_e.sum())
                # on écrit directement en float16 pour réduire BW mémoire
                out_pts_e = np.empty((total_e, 3), dtype=np.float16)
                out_lbl_e = np.full((total_e,), EMPTY_LABEL, dtype=np.uint8)

                # offsets de sortie (où écrire chaque voxel)
                out_off = np.empty((n_vox_e + 1,), dtype=np.int64)
                out_off[0] = 0
                np.cumsum(n_pts_per_vox_e, out=out_off[1:])

                # indices de sortie (flat) => quel voxel pour chaque point à générer
                # voxel_id_per_out[k] = i (index voxel) pour le k-ième point à écrire
                voxel_id_per_out = np.repeat(np.arange(n_vox_e, dtype=np.int32), n_pts_per_vox_e)

                # pour chaque point, on tire un offset local dans [0, count_voxel)
                counts_rep = count_e[voxel_id_per_out].astype(np.int64, copy=False)
                local = rng.integers(0, counts_rep, size=total_e, dtype=np.int64)

                # position globale dans l'array trié order_e : pos = start + local
                pos = start_e[voxel_id_per_out].astype(np.int64, copy=False) + local

                # gather en 1 shot
                chosen_global = order_e[pos]
                out_pts_e[:] = pts_empty[chosen_global].astype(np.float16, copy=False)

                points_empty = out_pts_e
                labels_empty = out_lbl_e

        toc(t0_empty, "EMPTY loop (vect)")

        # ================= UNKNOWN (vectorisé, approx: AVEC remise par voxel) =================
        t0_unk = tic()
        points_unknown = np.zeros((0, 3), dtype=np.float16)
        labels_unknown = np.zeros((0,), dtype=np.uint8)

        if pts_unknown_ray.size:
            order_u = np.argsort(flat_unknown, kind="mergesort")
            flat_u_sorted = flat_unknown[order_u]
            uniq_v_u, start_u, count_u = np.unique(flat_u_sorted, return_index=True, return_counts=True)

            n_vox_u = int(uniq_v_u.size)
            if n_vox_u:
                n_pts_per_vox_u = rng.integers(
                    self.points_per_voxel_min,
                    self.points_per_voxel_max + 1,
                    size=n_vox_u,
                    dtype=np.int32
                )

                total_u = int(n_pts_per_vox_u.sum())
                out_pts_u = np.empty((total_u, 3), dtype=np.float16)
                out_lbl_u = np.full((total_u,), UNKNOWN_LABEL, dtype=np.uint8)

                voxel_id_per_out = np.repeat(np.arange(n_vox_u, dtype=np.int32), n_pts_per_vox_u)

                counts_rep = count_u[voxel_id_per_out].astype(np.int64, copy=False)
                local = rng.integers(0, counts_rep, size=total_u, dtype=np.int64)
                pos = start_u[voxel_id_per_out].astype(np.int64, copy=False) + local

                chosen_global = order_u[pos]
                out_pts_u[:] = pts_unknown_ray[chosen_global].astype(np.float16, copy=False)

                points_unknown = out_pts_u
                labels_unknown = out_lbl_u

        toc(t0_unk, "UNKNOWN loop (vect)")

        n_occ, n_emp, n_unk = points_occ.shape[0], points_empty.shape[0], points_unknown.shape[0]
        print(f"   → Pools: occ={n_occ} pts, empty={n_emp} pts, unk={n_unk} pts (avant ratios)")

        # ---------------- ratios / target ----------------
        t0 = tic()
        if target_total_points <= 0:
            pts_final = np.vstack([points_occ, points_empty, points_unknown]).astype(np.float32, copy=False)
            lbl_final = np.hstack([labels_occ, labels_empty, labels_unknown]).astype(np.uint8, copy=False)
            toc(t0, "final stack (no target)")
            toc(t_global, "TOTAL")
            return pts_final, lbl_final

        n_occ_target = int(target_total_points * self.ratio_occ)
        n_emp_target = int(target_total_points * self.ratio_empty)
        n_unk_target = int(target_total_points - n_occ_target - n_emp_target)

        def sample_class(pts_c: np.ndarray, lbl_c: np.ndarray, target: int):
            if pts_c.shape[0] == 0 or target <= 0:
                return np.zeros((0, 3), dtype=np.float16), np.zeros((0,), dtype=np.uint8)
            if pts_c.shape[0] <= target:
                return pts_c, lbl_c
            idx = rng.choice(pts_c.shape[0], size=target, replace=False)
            return pts_c[idx], lbl_c[idx]

        pts_occ_s, lbl_occ_s = sample_class(points_occ, labels_occ, n_occ_target)
        pts_emp_s, lbl_emp_s = sample_class(points_empty, labels_empty, n_emp_target)
        pts_unk_s, lbl_unk_s = sample_class(points_unknown, labels_unknown, n_unk_target)

        pts_final = np.vstack([pts_occ_s, pts_emp_s, pts_unk_s]).astype(np.float16, copy=False)
        lbl_final = np.hstack([lbl_occ_s, lbl_emp_s, lbl_unk_s]).astype(np.uint8, copy=False)

        toc(t0, "final sampling+stack")

        print(
            f"   → Points finals (occ/empty/unk): "
            f"{pts_occ_s.shape[0]}/{pts_emp_s.shape[0]}/{pts_unk_s.shape[0]} "
            f"(total={pts_final.shape[0]:,} cible={target_total_points:,})"
        )

        toc(t_global, "TOTAL")
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
            [254, 253] + [int(_id) for (_id, _name, _rgb) in CARLA_22],
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
        # Sauvegarde des images (encode en parallèle)
        saved_images = 0
        with SectionTimer(self.perf, "save_images_jpg"):
            imgs = [(name, img) for name, img in frame_data['images'].items() if img is not None]

            def _encode_and_write(name_img):
                name, img = name_img
                img_path = os.path.join(self.output_dir, "images", f"frame_{formatted_id}_{name}.jpg")

                # encode en mémoire (souvent + rapide que imwrite direct)
                ok, buf = cv2.imencode(
                    ".jpg",
                    img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95]  # 95 = quasi-lossless, bien plus rapide que 100
                )
                if ok:
                    with open(img_path, "wb") as f:
                        f.write(buf.tobytes())
                    return 1
                return 0

            if imgs:
                from concurrent.futures import ThreadPoolExecutor
                n_workers = min(8, max(2, (os.cpu_count() or 8) // 2))
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    for r in ex.map(_encode_and_write, imgs):
                        saved_images += int(r)

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

        # -----------------------------------------
        # SENSOR MANAGER (avec window_back/fwd)
        # -----------------------------------------
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
            window_back=self.window_back,
            window_forward=self.window_forward,
        )

        max_frames = min(max_frames, len(self.positions))
        print("\n🚀 GÉNÉRATION DATASET OCCUPANCY IMPLICITE (BANK LiDAR multi-poses)")
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

                print(
                    f"\n📍 Position {position_count}/{max_frames} "
                    f"x={ref_position['location']['x']:.2f}, "
                    f"y={ref_position['location']['y']:.2f}"
                )

                # ------------------------------------------------
                # Création capteurs une seule fois
                # ------------------------------------------------
                if pos_idx == 0:
                    start_transform = carla.Transform(
                        carla.Location(**ref_position['location']),
                        carla.Rotation(**ref_position['rotation'])
                    )
                    if not self.sensor_manager.create_sensors_once(start_transform):
                        print("❌ Impossible de créer les capteurs")
                        return

                # ------------------------------------------------
                # Repère robot T0
                # ------------------------------------------------
                self.sensor_manager.set_reference_robot(ref_position)
                self.sensor_manager.move_cameras_to_position(ref_position)
                
                with self.sensor_manager.lock:
                    for k in self.sensor_manager.camera_received:
                        self.sensor_manager.camera_received[k] = False

                # tick "de plus" (voire 2 ticks) pour laisser la cam produire une image fraîche
                self.world.tick()
                self.world.tick()
                # attends la frame caméra (max 5 ticks)
                for _ in range(20):
                    with self.sensor_manager.lock:
                        if all(self.sensor_manager.camera_received.values()):
                            break
                    self.world.tick()
                
                x0 = float(ref_position["location"]["x"])
                y0 = float(ref_position["location"]["y"])
                z0 = float(ref_position["location"]["z"])
                pitch0 = float(ref_position["rotation"].get("pitch", 0.0))
                roll0  = float(ref_position["rotation"].get("roll", 0.0))
                # ------------------------------------------------
                # Construire mapping slot -> pose
                # ------------------------------------------------
                slot_to_pose = {}
                for s in self.sensor_manager.lidar_slot_ids:
                    j = pos_idx + s
                    if j < 0 or j >= len(self.positions):
                        continue

                    pj = self.positions[j]
                    pose_j = {
                        "location": dict(pj["ego_location"]),
                        "rotation": dict(pj["ego_rotation"]),
                        "timestamp_sim": pj["timestamp_sim"],
                    }
                    pose_j["location"]["x"] = x0
                    pose_j["location"]["y"] = y0
                    pose_j["location"]["z"] = z0
                    pose_j["rotation"]["pitch"] = pitch0
                    pose_j["rotation"]["roll"]  = roll0

                    slot_to_pose[int(s)] = pose_j
                print(f"🧭 rigs actifs: {len(slot_to_pose)}")

                # ------------------------------------------------
                # Placement initial de tout le bank
                # ------------------------------------------------
                self.sensor_manager.move_all_lidar_rigs(
                    slot_to_pose,
                    global_offset_world=(0.0, 0.0, 0.0)
                )

                # ------------------------------------------------
                # Offset global si proximité actors
                # ------------------------------------------------
                dx, dy, dz = self.sensor_manager.find_safe_global_offset(
                    slot_to_pose,
                    min_dist=1.0,
                    max_tries=15,
                    xy_radius=2.0,
                    z_offset=0.5
                )

                if abs(dx) + abs(dy) + abs(dz) > 0.0:
                    print(f"↔️ global transpose lidar bank: dx={dx:.2f} dy={dy:.2f} dz={dz:.2f}")

                self.sensor_manager.move_all_lidar_rigs(
                    slot_to_pose,
                    global_offset_world=(dx, dy, dz)
                )

                # ------------------------------------------------
                # Randomize params sur tout le bank
                # ------------------------------------------------
                if self.randomize_clear_poses:
                    self.sensor_manager.randomize_all_lidars_params()

                # ------------------------------------------------
                # ACCUMULATION SIMULTANÉE
                # ------------------------------------------------
                self.sensor_manager.start_new_accumulation(self.capture_points_target)

                ticks_done = 0

                while (
                    not self.sensor_manager.lidar_accumulator.is_complete()
                    and ticks_done < self.max_ticks_per_pose
                ):
                    self.world.tick()
                    ticks_done += 1

                print(f"⏱ ticks accumulation: {ticks_done}")

                # ------------------------------------------------
                # Récupération données
                # ------------------------------------------------
                frame_data = self.sensor_manager.capture_current_frame(self.fixed_weather_name)

                counts = self.sensor_manager.lidar_accumulator.get_tag_counts()
                items = sorted(counts.items(), key=lambda x: x[0])
                total = sum(v for _, v in items)

                print("📌 Points par slot lidar:")
                for slot, n in items:
                    print(f"   slot={slot:+d} -> {n} pts ({100.0*n/max(total,1):.1f}%)")

                if frame_data and len(frame_data['points']) > 0:
                    unique_frame_id = self.global_frame_counter

                    with SectionTimer(self.perf, "save_frame_total"):
                        self.writer.queue.put((frame_data, unique_frame_id, ref_position))

                    self.global_frame_counter += 1

                    del frame_data
                    gc.collect()
                else:
                    print("⚠️ Pas de points LiDAR pour cette frame")

                # ------------------------------------------------
                # LOG PROGRESSION
                # ------------------------------------------------
                if position_count % 5 == 0:
                    elapsed = time.time() - start_time
                    fps = self.global_frame_counter / elapsed if elapsed > 0 else 0
                    remaining = max_frames - position_count
                    eta = remaining / fps if fps > 0 else 0

                    print(
                        f"{'='*50}"
                        f" PROGRESSION {position_count}/{max_frames}"
                        f" | frames={self.global_frame_counter}"
                        f" | fps={fps:.2f}"
                        f" | ETA={eta/60:.1f} min"
                        f"{'='*50}"
                    )

        except KeyboardInterrupt:
            print("⚠️ Interruption utilisateur")

        except Exception as e:
            print(f"❌ Erreur génération: {e}")
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

            print(
                f"{'='*60}\n"
                f"✅ GÉNÉRATION TERMINÉE\n"
                f"Frames: {self.global_frame_counter}\n"
                f"Temps total: {total_time/60:.1f} min\n"
                f"{'='*60}"
            )

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

    parser.add_argument('--z-min', type=float, default=1, help='Hauteur min relative LiDAR (m)')
    parser.add_argument('--z-max', type=float, default=2, help='Hauteur max relative LiDAR (m)')
    parser.add_argument('--z-step', type=float, default=2, help='Pas entre LiDARs (m)')
    parser.add_argument('--h-fov', type=float, default=360.0, help='FOV horizontal (deg)')
    parser.add_argument('--v-upper', type=float, default=60.0, help='FOV vertical haut (deg)')
    parser.add_argument('--v-lower', type=float, default=-60.0, help='FOV vertical bas (deg)')
    parser.add_argument('--lidar-channels', type=int, default=700, help='Canaux LiDAR')
    parser.add_argument('--lidar-pps', type=int, default=500_000, help='Points/seconde LiDAR')
    parser.add_argument('--lidar-range', type=float, default=150, help='Portée LiDAR (m)')

    parser.add_argument('--map', type=str, default='Town10HD_Opt', help='Carte CARLA')
    parser.add_argument('--trajectory-json', type=str,
                        default="carla_trajectories/Town10HD_Opt_fast_20251010_194958_veh25.json",
                        help='Trajectoire à rejouer')
    parser.add_argument('--weather-id', type=int, default=0,
                        help='0 clear_noon | 1 overcast_morning | ...')
    parser.add_argument('--profile', action='store_true', default=True, help='Activer le profiling')
    parser.add_argument('--capture-points', type=int, default=1_500_000,
                        help='Quota de points LiDAR à capturer par frame (hits + empty)')
    parser.add_argument('--points-min-saved', type=int, default=20_000,
                        help='Nb min de points occupancy sauvegardés par frame')
    parser.add_argument('--points-max-saved', type=int, default=30_000,
                        help='Nb max de points occupancy sauvegardés par frame')
    parser.add_argument('--cube-size-m', type=float, default=0.005,
                        help='Taille “visuelle” du marker dans la preview (m)')
    parser.add_argument('--cam-height-noise-pct', type=float, default=15.0,
                        help='Bruit hauteur caméra en % (±)')
    parser.add_argument('--cam-angle-noise-pct', type=float, default=10.0,
                        help='Bruit angles caméra (pitch/yaw/roll) en % (±)')

    # fenêtre ego
    parser.add_argument('--window-back', type=int, default=2, help='Nombre de poses passées de l’ego à charger')
    parser.add_argument('--window-forward', type=int, default=2, help='Nombre de poses futures de l’ego à charger')
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
    parser.add_argument('--implicit-points-per-voxel-max', type=int, default=3,
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
    parser.add_argument('--implicit-empty-points-per-hit', type=int, default=1,
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
