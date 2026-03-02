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
from typing import Tuple, List, Dict, Optional, Set
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
from queue import Queue, Full
from threading import Thread
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
# Sentinel pour les points EMPTY rajoutés dans les callbacks LiDAR
LIDAR_EMPTY_SENTINEL = 254
LIDAR_UNKNOWN_SENTINEL = 253

# Debug transform (matrices + transpose). Laisse False en prod (ça spam).
DEBUG_LIDAR_TRANSFORMS = False
# À partir de quel frame on commence à imprimer le debug TF (utile sur runs longs).
DEBUG_LIDAR_TRANSFORMS_START_FRAME = 0


# Thread pour le npz 

class AsyncWriter(Thread):
    def __init__(self, save_fn):
        super().__init__(daemon=True)
        self.queue = Queue(maxsize=500)
        self.save_fn = save_fn
        self.start()

    def run(self):
        while True:
            data, frame_id, ref_pos = self.queue.get()
            try:
                self.save_fn(data, frame_id, ref_pos)
            except Exception:
                traceback.print_exc()
            self.queue.task_done()

# ==========================
# PROFILING
# ==========================
class PerfStats:
    def __init__(self):
        self.t_tot = defaultdict(float)
        self.count = defaultdict(int)
        self.samples = defaultdict(list)
        self.callback_times = defaultdict(list)  # Pour callbacks avec détails

    def add(self, label: str, dt: float):
        self.t_tot[label] += dt
        self.count[label] += 1
        self.samples[label].append(dt)
    
    def add_callback(self, callback_type: str, dt: float, detail: str = ""):
        """Enregistre les temps des callbacks avec détails"""
        key = f"{callback_type}_{detail}" if detail else callback_type
        self.callback_times[key].append(dt)

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
        
        # Rapport des callbacks
        if self.callback_times:
            print("\n" + "=" * 70)
            print("⚡ ANALYSE CALLBACKS")
            cb_items = sorted(self.callback_times.items())
            for cb_type, times in cb_items:
                if times:
                    arr = np.array(times, dtype=np.float64)
                    total_cb = np.sum(arr)
                    n = len(arr)
                    avg = np.mean(arr)
                    mn = np.min(arr)
                    mx = np.max(arr)
                    p95 = np.percentile(arr, 95)
                    p99 = np.percentile(arr, 99)
                    print(f" - {cb_type:<32s} total={total_cb:8.3f}s  avg={avg:7.4f}s  "
                          f"min={mn:7.4f}s  p95={p95:7.4f}s  p99={p99:7.4f}s  max={mx:7.4f}s  (n={n})")
        
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
    z_range: Tuple[float, float] = (-1.0, 3.0)
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
        # Allocation avec marge
        # On alloue un buffer unique pour éviter la fragmentation et le vstack final
        self._capacity = int(self.target_points * 1.2)
        self._pts_buffer = np.empty((self._capacity, 3), dtype=np.float16)
        self._lbl_buffer = np.empty((self._capacity,), dtype=np.uint8)
        self._cursor = 0
        self.counts_by_tag = defaultdict(int)

    def reset(self, target_points=None):
        with self.lock:
            self._cursor = 0
            self.counts_by_tag.clear()
            
            if target_points is not None:
                new_target = int(target_points)
                if new_target != self.target_points:
                    self.target_points = new_target
                    # Realloc intelligente si besoin
                    needed = int(new_target * 1.2)
                    if needed > self._capacity: # Grow
                        # print(f"[Accumulator] Growing buffer: {self._capacity} -> {needed}")
                        self._capacity = needed
                        self._pts_buffer = np.empty((self._capacity, 3), dtype=np.float16)
                        self._lbl_buffer = np.empty((self._capacity,), dtype=np.uint8)
                    # On garde le buffer s'il est un peu trop grand (évite le churn)

    def get_tag_counts(self):
        with self.lock:
            return dict(self.counts_by_tag)
    
    def add(self, pts, lbls, tag=None):
        n = pts.shape[0]
        if n == 0:
            return

        with self.lock:
            # Check capacity overflow
            if self._cursor + n > self._capacity:
                # Double capacity
                new_cap = max(int(self._capacity * 1.5), self._cursor + n + 100000)
                # print(f"[Accumulator] Overflow resize: {self._capacity} -> {new_cap}")
                new_pts = np.empty((new_cap, 3), dtype=np.float16)
                new_lbl = np.empty((new_cap,), dtype=np.uint8)
                
                # Copy old content
                new_pts[:self._cursor] = self._pts_buffer[:self._cursor]
                new_lbl[:self._cursor] = self._lbl_buffer[:self._cursor]
                
                self._pts_buffer = new_pts
                self._lbl_buffer = new_lbl
                self._capacity = new_cap
            
            # Direct insert (Zero copy from `get` perspective later)
            self._pts_buffer[self._cursor : self._cursor + n] = np.asarray(pts, dtype=np.float16)
            self._lbl_buffer[self._cursor : self._cursor + n] = lbls
            self._cursor += n

            if tag is not None:
                self.counts_by_tag[int(tag)] += n

    def is_complete(self):
        with self.lock:
            return self._cursor >= self.target_points

    def get(self):
        with self.lock:
            if self._cursor == 0:
                return None, None
            
            # Return a COPY of the slice.
            # This is one big copy, much better than vstack(thousands of arrays).
            # We must copy because we reuse the internal buffer for the next frame.
            pts_out = self._pts_buffer[:self._cursor].copy()
            lbl_out = self._lbl_buffer[:self._cursor].copy()
            
            return pts_out, lbl_out


# ==========================
# METEO
# ==========================
class WeatherManager:
    def __init__(self, world, apply_settle_tick: bool = True):
        self.world = world
        self.apply_settle_tick = bool(apply_settle_tick)
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
        if self.apply_settle_tick:
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
        empty_points_per_hit: int = 1,
        window_back: int = 2,
        window_forward: int = 2,
        allow_extra_maintenance_ticks: bool = True,
        cam_supersample_factor: int = 4,
    ):
        self.world = world
        self.enable_blur = enable_blur
        self.sensor_ids = set()
        self.lidars = []
        self.cameras = []
        self.camera_callbacks = {}
        self.cameras_active = True
        self.sensors_created = False
        self.current_robot_transform = None
        self.reference_robot_transform = None
        # Matrice world->robot(T0) figée au moment où on fixe T0.
        # But: éviter toute dérive/oscillation si on reconstruit des Transform ailleurs.
        self.reference_robot_M_wr: Optional[np.ndarray] = None
        self.camera_data: Dict[str, np.ndarray] = {}
        self.camera_received: Dict[str, bool] = {}
        self.lock = threading.Lock()
        self.capture_points_target = int(capture_points_target)
        self.lidar_accumulator = LidarAccumulatorUntilTarget(self.capture_points_target)
        self.current_pose_j = None
        self.allow_extra_maintenance_ticks = bool(allow_extra_maintenance_ticks)

        # ✅ AJOUT : frame caméra qu’on veut garder (T0)
        self.target_cam_frame = None

        # ✅ AJOUT : frame LiDAR qu’on accepte (utile en mode 1 tick/pose strict)
        self.target_lidar_frame = None
        # Mode strict: si True, on ignore toute callback LiDAR tant que
        # target_lidar_frame n'est pas défini (évite les frames inter-poses).
        self.require_target_lidar_frame = False

        # Suivi des callbacks LiDAR reçus pour le frame attendu.
        # En mode 1 tick/pose, sans une courte attente (sans tick) on peut capturer
        # avant l'arrivée de toutes les callbacks => sorties (NPZ) quasi vides et oscillantes.
        self._lidar_seen_frame: Optional[int] = None
        self._lidar_seen_sensor_ids: Set[int] = set()

        # Cam config :
        # 🚀 OPTIM: cam_supersample_factor contrôle le ratio capture/sortie.
        # factor=4 → capture 2048×1536, 2× pyrDown (qualité max, tick CARLA lent)
        # factor=2 → capture 1024×768,  1× pyrDown (qualité très bonne, tick ~2-3× plus rapide)
        # factor=1 → capture directe 512×384 (pas de supersampling, tick le plus rapide)
        self.cam_supersample_factor = max(1, int(cam_supersample_factor))

        # Résolution finale (réseau)
        self.cam_out_w = 512
        self.cam_out_h = 384

        self.cam_capture_w = self.cam_out_w * self.cam_supersample_factor
        self.cam_capture_h = self.cam_out_h * self.cam_supersample_factor

        # Encodage JPEG caméra réalisé directement dans le callback.
        self.cam_encode_wait_timeout_s = 0.0

        # LiDAR: pipeline async (callback ultra-légère -> workers)
        self._lidar_q = Queue(maxsize=2048)
        self._lidar_workers = []
        # 🚀 BUGFIX: max(12, min(12, ...)) retourne toujours 12!
        # On utilise tous les cores disponibles (min 8, max 24)
        cpu_count = int(os.cpu_count() or 8)
        self._lidar_workers_n = min(24, max(8, cpu_count))
        for _ in range(self._lidar_workers_n):
            t = Thread(target=self._lidar_worker, daemon=True)
            t.start()
            self._lidar_workers.append(t)

        # Anti-doublon (capteur -> 1 message / frame)
        self._lidar_enqueued_by_frame: Dict[int, Set[int]] = {}

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


        self._lidar_tf_cache = {}  # (frame_id, sensor_id) -> carla.Transform
        # Transform attendue (sensor->world) après move_all_lidar_rigs().
        # Plus robuste que data.transform (qui peut être incohérent selon le backend/latence).
        self._lidar_expected_M_sw: Dict[int, np.ndarray] = {}  # sensor_id -> 4x4
        # Snapshot figé des transforms attendues par frame cible.
        # Évite d'utiliser une transform d'une pose suivante dans une callback retardée.
        self._lidar_expected_M_sw_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        self.camera_jpeg = {}          # cam_name -> bytes
        self.camera_raw_received = {cfg['name']: False for cfg in self.CAMERA_CONFIGS}
        # Empêche de ré-enqueue la même caméra tant que le worker n'a pas fini
        self.camera_pending = {cfg['name']: False for cfg in self.CAMERA_CONFIGS}
        self.camera_frame_id = {cfg['name']: -1 for cfg in self.CAMERA_CONFIGS}
        self.camera_jpeg_frame_id = {cfg['name']: -1 for cfg in self.CAMERA_CONFIGS}
        self.target_cam_frame = None

        self._accum_epoch = 0
        # Matrice world->robot(T0) figée pour l'accumulation courante.
        self._accum_reference_robot_M_wr: Optional[np.ndarray] = None
        self.cam_height_noise_pct = float(cam_height_noise_pct)
        self.cam_angle_noise_pct = float(cam_angle_noise_pct)

        self.empty_points_per_hit = max(int(empty_points_per_hit), 0)
        self._lidar_rotation_frequency_hz = 10.0
        self._expected_points_per_scan = max(
            1,
            int(float(self.params.get('pps', 0)) / max(float(self._lidar_rotation_frequency_hz), 1e-6))
        )

        # Debug transform: compare conventions (colonne vs ligne/transposée).
        # Objectif: voir si un mauvais transpose laisse les points en world.
        self.debug_lidar_transforms = bool(DEBUG_LIDAR_TRANSFORMS)
        self._dbg_tf_prints_left = 10
        self._dbg_tf_last_frame_printed = None

        # Convention d'application des matrices 4x4 (détectée une fois).
        # Possibles: "col" (M@p), "colT" (M.T@p), "row" (p@M), "rowT" (p@M.T)
        # Note: pour "row*", l'ordre de composition des matrices doit être inversé.
        self._tf_apply_mode: Optional[str] = None

        print("🎯 CAPTURE LiDAR/CAM — repère robot T0 + multi-poses")

    def settle_sensors_after_teleport(self, settle_ticks: int = 2, settle_sleep_s: float = 0.01) -> None:
        """Laisse CARLA appliquer les set_transform avant une nouvelle capture.

        Important: à appeler AVANT start_new_accumulation(), pour que les callbacks
        éventuellement reçues pendant ce settle ne polluent pas la frame cible.
        """
        n = max(0, int(settle_ticks))
        for _ in range(n):
            self.world.tick()
        if float(settle_sleep_s) > 0.0:
            time.sleep(float(settle_sleep_s))

    def verify_sensors_positioned(self, ref_position: dict, warn_threshold_m: float = 1.0) -> bool:
        """Vérifie que les capteurs (caméras + LiDARs) sont bien à leur position attendue.

        Compare la position actuelle (get_transform) de chaque capteur avec la position
        ego de référence. Si un capteur est trop loin, affiche un warning.

        Args:
            ref_position: dict avec 'location' et 'rotation' de l'ego.
            warn_threshold_m: seuil en mètres au-delà duquel on considère un capteur mal placé.

        Returns:
            True si tous les capteurs sont dans le seuil, False sinon.
        """
        ego_loc = ref_position['location']
        ex, ey, ez = float(ego_loc['x']), float(ego_loc['y']), float(ego_loc['z'])
        all_ok = True

        # Vérif caméras
        for cam, cfg in zip(self.cameras, self.CAMERA_CONFIGS):
            if not (cam and cam.is_alive):
                continue
            try:
                tf = cam.get_transform()
                # La position attendue de la caméra = ego + offset local
                expected_x = ex + cfg['dx']
                expected_y = ey + cfg['dy']
                expected_z = ez + cfg['dz']
                dx = tf.location.x - expected_x
                dy = tf.location.y - expected_y
                dz = tf.location.z - expected_z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist > warn_threshold_m:
                    print(
                        f"⚠️ CAM '{cfg['name']}' mal positionnée: "
                        f"dist={dist:.2f}m (attendu ~[{expected_x:.1f},{expected_y:.1f},{expected_z:.1f}], "
                        f"réel [{tf.location.x:.1f},{tf.location.y:.1f},{tf.location.z:.1f}])"
                    )
                    all_ok = False
            except Exception:
                pass

        # Vérif LiDARs (slot 0 = pose ego T0)
        rig_0 = self.lidar_rigs.get(0, [])
        for lidar in rig_0:
            if not (lidar and lidar.is_alive):
                continue
            try:
                tf = lidar.get_transform()
                dx = tf.location.x - ex
                dy = tf.location.y - ey
                dz = tf.location.z - ez
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                # Les LiDARs sont en z-stack, donc dz peut être grand (~2m),
                # on ne vérifie que xy.
                dist_xy = math.sqrt(dx*dx + dy*dy)
                if dist_xy > warn_threshold_m:
                    print(
                        f"⚠️ LIDAR (slot 0, id={lidar.id}) mal positionné: "
                        f"dist_xy={dist_xy:.2f}m (attendu ~[{ex:.1f},{ey:.1f}], "
                        f"réel [{tf.location.x:.1f},{tf.location.y:.1f}])"
                    )
                    all_ok = False
            except Exception:
                pass

        if all_ok:
            print("✅ Tous les capteurs sont bien positionnés")
        return all_ok

    def set_cameras_active(self, active: bool):
        """Active/désactive les caméras (stop/listen) pour réduire le coût de rendu.

        Utile pour un downsample temporel: on ne capture les images que 1 pose sur N,
        mais on garde la résolution intacte.
        """
        active = bool(active)
        if self.cameras_active == active:
            return

        for cam, cfg in zip(self.cameras, self.CAMERA_CONFIGS):
            if not (cam and cam.is_alive):
                continue
            name = cfg['name']
            try:
                if active:
                    cb = self.camera_callbacks.get(name, None)
                    if cb is not None:
                        cam.listen(cb)
                else:
                    cam.stop()
            except Exception:
                pass

        self.cameras_active = active

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

    def _expected_lidar_callbacks(self) -> int:
        # Nombre de capteurs LiDAR actifs = (slots) * (configs z-stack actives)
        n_cfg = sum(1 for x in self.active_lidar_mask if x)
        return int(n_cfg) * int(len(self.lidar_slot_ids))

    def cache_lidar_transforms_for_frame(self, frame_id: int) -> None:
        fid = int(frame_id)
        with self.lock:
            self._lidar_tf_cache.clear()
            for lidar in self.lidars:
                if lidar and lidar.is_alive:
                    self._lidar_tf_cache[(fid, int(lidar.id))] = lidar.get_transform()

    def snapshot_expected_lidar_matrices_for_frame(self, frame_id: int) -> None:
        """Fige la table sensor->world pour une frame cible donnée."""
        fid = int(frame_id)
        with self.lock:
            snap = {
                int(sensor_id): np.array(M_sw, dtype=np.float32, copy=True)
                for sensor_id, M_sw in self._lidar_expected_M_sw.items()
            }
            self._lidar_expected_M_sw_by_frame[fid] = snap

            # Garde une petite fenêtre pour limiter la mémoire.
            min_keep = fid - 4
            old_keys = [k for k in self._lidar_expected_M_sw_by_frame.keys() if int(k) < min_keep]
            for k in old_keys:
                self._lidar_expected_M_sw_by_frame.pop(k, None)

    def wait_lidar_workers_idle(self, timeout_s: float = 3.0, poll_s: float = 0.001) -> bool:
        """Attend que la queue LiDAR async soit totalement drainée."""
        t0 = time.perf_counter()
        while True:
            if int(getattr(self._lidar_q, "unfinished_tasks", 0)) == 0:
                return True
            if (time.perf_counter() - t0) >= float(timeout_s):
                return False
            time.sleep(float(poll_s))

    def calibrate_matrix_apply_mode_once(self) -> None:
        """Détermine une fois la convention d'application des matrices CARLA.

        On compare le résultat de carla.Transform.transform(Location) avec 4 façons
        d'appliquer get_matrix() via numpy. Ça élimine l'heuristique ROI%.
        """
        if self._tf_apply_mode is not None:
            return
        try:
            tf = carla.Transform(
                carla.Location(x=10.0, y=-5.0, z=2.0),
                carla.Rotation(pitch=12.0, yaw=37.0, roll=18.0),
            )
            M = np.array(tf.get_matrix(), dtype=np.float32)

            rng = np.random.default_rng(12345)
            pts = rng.normal(size=(64, 3)).astype(np.float32)
            pts *= np.array([5.0, 5.0, 2.0], dtype=np.float32)
            pts4 = np.ones((pts.shape[0], 4), dtype=np.float32)
            pts4[:, :3] = pts

            truth = []
            for x, y, z in pts:
                w = tf.transform(carla.Location(x=float(x), y=float(y), z=float(z)))
                truth.append((w.x, w.y, w.z))
            truth = np.array(truth, dtype=np.float32)

            def pred(mode: str) -> np.ndarray:
                if mode == "col":
                    return (M @ pts4.T).T[:, :3]
                if mode == "colT":
                    return (M.T @ pts4.T).T[:, :3]
                if mode == "row":
                    return (pts4 @ M)[:, :3]
                return (pts4 @ M.T)[:, :3]  # rowT

            modes = ["col", "colT", "row", "rowT"]
            errs = {}
            for m in modes:
                p = pred(m)
                errs[m] = float(np.mean((p - truth) ** 2))

            best = min(errs.items(), key=lambda kv: kv[1])[0]
            with self.lock:
                if self._tf_apply_mode is None:
                    self._tf_apply_mode = best

            try:
                print(f"[TF CALIB] apply_mode={self._tf_apply_mode} mse={errs[self._tf_apply_mode]:.3e} all={errs}")
            except Exception:
                pass
        except Exception:
            # Fallback (ne doit jamais casser la capture)
            with self.lock:
                if self._tf_apply_mode is None:
                    self._tf_apply_mode = "col"

    def reset_lidar_frame_tracking(self) -> None:
        with self.lock:
            self._lidar_seen_sensor_ids.clear()
            self._lidar_seen_frame = int(self.target_lidar_frame) if self.target_lidar_frame is not None else None
            self._lidar_enqueued_by_frame.clear()
            if self._lidar_seen_frame is not None:
                self._lidar_enqueued_by_frame[int(self._lidar_seen_frame)] = set()

    @staticmethod
    def _try_bytes(buf) -> bytes:
        try:
            if isinstance(buf, (bytes, bytearray)):
                return bytes(buf)
            return bytes(memoryview(buf))
        except Exception:
            return b""

    @staticmethod
    def _shallow_copy_pose(pose: Optional[dict]) -> Optional[dict]:
        if pose is None:
            return None
        try:
            return {
                "location": dict(pose.get("location", {})),
                "rotation": dict(pose.get("rotation", {})),
            }
        except Exception:
            return None

    def _drain_queue_nowait(self, q: Queue) -> int:
        n = 0
        try:
            while True:
                q.get_nowait()
                q.task_done()
                n += 1
        except Exception:
            pass
        return n

    def _enqueue_lidar_task(self, task) -> bool:
        try:
            # Petit timeout pour lisser les pics sans retomber trop vite
            # dans le fallback lourd du callback LiDAR.
            self._lidar_q.put(task, timeout=0.002)
            return True
        except Full:
            return False
        except Exception:
            return False

    def _mark_lidar_seen(self, sensor_id: int, frame_id: int) -> None:
        with self.lock:
            if self._lidar_seen_frame is None:
                return
            if int(frame_id) != int(self._lidar_seen_frame):
                return
            self._lidar_seen_sensor_ids.add(int(sensor_id))

    def wait_for_lidar_callbacks(self, timeout_s: float = 0.25, poll_s: float = 0.001) -> bool:
        tgt = self.target_lidar_frame
        if tgt is None:
            return True

        expected = self._expected_lidar_callbacks()
        if expected <= 0:
            return True

        t0 = time.perf_counter()
        while True:
            with self.lock:
                got = len(self._lidar_seen_sensor_ids) if self._lidar_seen_frame == int(tgt) else 0
            if got >= expected:
                return True
            if (time.perf_counter() - t0) >= float(timeout_s):
                return False
            time.sleep(float(poll_s))

    def start_new_accumulation(
        self,
        target_points: Optional[int] = None,
        target_lidar_frame: Optional[int] = None,
        target_cam_frame: Optional[int] = None,
    ):
        # ✅ epoch++ pour invalider callbacks retard
        with self.lock:
            self._accum_epoch += 1
            self._accum_reference_robot_M_wr = (
                None if self.reference_robot_M_wr is None
                else np.array(self.reference_robot_M_wr, dtype=np.float32, copy=True)
            )

        # Purge best-effort des tâches LiDAR en attente (ancien epoch)
        try:
            self._drain_queue_nowait(self._lidar_q)
        except Exception:
            pass

        with self.lock:
            self._lidar_enqueued_by_frame.clear()
            self.target_lidar_frame = None if target_lidar_frame is None else int(target_lidar_frame)
            self.target_cam_frame = None if target_cam_frame is None else int(target_cam_frame)

        self.lidar_accumulator.reset(target_points or self.capture_points_target)

        # reset cam
        for name in self.camera_raw_received:
            self.camera_raw_received[name] = False
            self.camera_pending[name] = False
            self.camera_frame_id[name] = -1
            self.camera_jpeg_frame_id[name] = -1
            self.camera_jpeg[name] = None

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


    def _to_robot_frame_cached(self,
                              pts_local: np.ndarray,
                              M_sw: np.ndarray,
                              M_wr: np.ndarray,
                              bank_offset_world: Tuple[float, float, float],
                              dbg_frame: int = -1,
                              out_dtype=np.float32) -> np.ndarray:
        if pts_local is None or pts_local.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        # world -> robot(T0) (déjà figé au moment où T0 est fixé)
        if M_wr is None:
            return np.zeros((0, 3), dtype=np.float32)

        pts4 = np.ones((pts_local.shape[0], 4), dtype=np.float32)
        pts4[:, :3] = pts_local.astype(np.float32, copy=False)

        # Applique une matrice 4x4 selon la convention CARLA calibrée.
        # On applique SEQUENTIELLEMENT:
        #   1) sensor -> world via M_sw
        #   2) undo offset world (soustraction directe, indépendante de la convention)
        #   3) world -> robot(T0) via M_wr
        with self.lock:
            mode = self._tf_apply_mode or "col"

        def _apply_mat(M: np.ndarray, pts4_in: np.ndarray) -> np.ndarray:
            if mode == "row":
                return pts4_in @ M
            if mode == "rowT":
                return pts4_in @ M.T
            if mode == "colT":
                return (M.T @ pts4_in.T).T
            return (M @ pts4_in.T).T  # col

        pts_world4 = _apply_mat(M_sw, pts4)
        # undo global bank offset in WORLD (simple et robuste)
        dxo, dyo, dzo = bank_offset_world
        if dxo or dyo or dzo:
            pts_world4[:, 0] -= float(dxo)
            pts_world4[:, 1] -= float(dyo)
            pts_world4[:, 2] -= float(dzo)

        pts_robot4 = _apply_mat(M_wr, pts_world4)
        pts_robot = pts_robot4[:, :3]

        # Debug convention : version worker (utilise dbg_frame capturé)
        if self.debug_lidar_transforms and self._dbg_tf_prints_left > 0:
            try:
                start_f = int(DEBUG_LIDAR_TRANSFORMS_START_FRAME)
                cur_f = int(dbg_frame)
                if cur_f != -1 and cur_f < start_f:
                    return pts_robot

                if self._dbg_tf_last_frame_printed is None or self._dbg_tf_last_frame_printed != int(cur_f):
                    self._dbg_tf_last_frame_printed = int(cur_f)
                    self._dbg_tf_prints_left -= 1

                    n = int(pts4.shape[0])
                    m = min(n, 2048)
                    if n > m:
                        idx = np.arange(m, dtype=np.int64)
                        c_loc = pts_local[idx]
                        pts4_s = pts4[idx]
                    else:
                        c_loc = pts_local
                        pts4_s = pts4

                    x_min_d, x_max_d = -16.0, 16.0
                    y_min_d, y_max_d = -16.0, 16.0
                    z_min_d, z_max_d = -1.0, 3.0

                    def _roi_ratio(p3: np.ndarray) -> float:
                        if p3.size == 0:
                            return 0.0
                        x = p3[:, 0]; y = p3[:, 1]; z = p3[:, 2]
                        return float(((x >= x_min_d) & (x <= x_max_d) & (y >= y_min_d) & (y <= y_max_d) & (z >= z_min_d) & (z <= z_max_d)).mean())

                    print("\n" + "=" * 70)
                    print("[TF DBG] Robot-frame transform (sequential) [WORKER]")
                    print(f"[TF DBG] frame={cur_f} sample={m}/{n} bank_offset_world={bank_offset_world} apply_mode={self._tf_apply_mode}")

                    def _summ(name: str, p3: np.ndarray, roi: float):
                        if p3.size == 0:
                            print(f"[TF DBG] {name}: empty")
                            return
                        mn = p3.min(axis=0)
                        mx = p3.max(axis=0)
                        mu = p3.mean(axis=0)
                        print(f"[TF DBG] {name}: roi={roi*100:.3f}% mean=({mu[0]:.2f},{mu[1]:.2f},{mu[2]:.2f}) x[{mn[0]:.1f},{mx[0]:.1f}] y[{mn[1]:.1f},{mx[1]:.1f}] z[{mn[2]:.1f},{mx[2]:.1f}]")

                    # Stats uniquement sur le repère final (robot)
                    c_robot = pts_robot[:m] if pts_robot.shape[0] >= m else pts_robot
                    r_robot = _roi_ratio(c_robot)
                    _summ("local", c_loc, 0.0)
                    _summ("robot", c_robot, r_robot)
                    print("=" * 70 + "\n")
            except Exception:
                pass

        if out_dtype is not None and out_dtype != np.float32:
            return pts_robot.astype(out_dtype, copy=False)
        return pts_robot

    def _lidar_worker(self):
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32),
            ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
        ])
        while True:
            task = self._lidar_q.get()
            try:
                (raw_bytes, fid, sensor_id, slot_id, cfg_index,
                 epoch_task, M_sw, M_wr, bank_off, empty_k, max_range_m_task) = task

                with self.lock:
                    if int(epoch_task) != int(self._accum_epoch):
                        continue

                if not raw_bytes:
                    self._mark_lidar_seen(sensor_id, fid)
                    continue

                arr = np.frombuffer(raw_bytes, dtype=dtype)
                n = arr.shape[0]
                if n == 0:
                    self._mark_lidar_seen(sensor_id, fid)
                    continue

                if M_wr is None:
                    self._mark_lidar_seen(sensor_id, fid)
                    continue

                # Optim: alloc direct
                pts_local = np.empty((n, 3), dtype=np.float32)
                pts_local[:, 0] = arr['x']
                pts_local[:, 1] = arr['y']
                pts_local[:, 2] = arr['z']
                
                lbl_hits = arr['ObjTag'].astype(np.uint8, copy=False)

                pts_robot_hits = self._to_robot_frame_cached(
                    pts_local, M_sw, M_wr, bank_off, dbg_frame=int(fid), out_dtype=np.float16
                )

                # 🚀 OPTIMISATION EMPTY: Réduire drastiquement les calculs
                pts_robot_empty = np.zeros((0, 3), dtype=np.float32)
                lbl_empty = np.zeros((0,), dtype=np.uint8)
                k = int(empty_k)
                
                if k > 0 and n > 0:
                   # Limite: max 20% des points, ou 10k points max par callback
                   k_actual = min(k, max(1, int(20000 / n)))
                   if k_actual > 0:
                       max_range_m = 24.0
                       d = np.linalg.norm(pts_local, axis=1).astype(np.float32)
                       s_max = np.minimum(0.98, max_range_m / (d + 1e-6)).astype(np.float32)
                       
                       # Générer moins de points EMPTY
                       r = np.random.rand(n, k_actual).astype(np.float32)
                       t = r * s_max[:, None]
                       
                       pts_empty_local = (pts_local[:, None, :] * t[..., None]).reshape(-1, 3)
                       pts_robot_empty = self._to_robot_frame_cached(
                           pts_empty_local, M_sw, M_wr, bank_off, dbg_frame=int(fid), out_dtype=np.float16
                       )
                       lbl_empty = np.full((len(pts_robot_empty),), LIDAR_EMPTY_SENTINEL, dtype=np.uint8)
                       
                       # Libérer mémoire immédiatement
                       del pts_empty_local, r, t, d, s_max


                # UNKNOWN désactivé — on ne génère que Hits + Empty

                pts_concat = np.vstack([pts_robot_hits, pts_robot_empty])
                lbl_concat = np.hstack([lbl_hits, lbl_empty])

                if self.debug_lidar_transforms and self._dbg_tf_prints_left > 0:
                    try:
                        print(
                            f"[TF DBG] lidar_worker frame={fid} sensor={int(sensor_id)} slot={slot_id} cfg={cfg_index} "
                            f"hits={len(pts_robot_hits):,} empty={len(pts_robot_empty):,}"
                        )
                    except Exception:
                        pass

                with self.lock:
                    if int(epoch_task) != int(self._accum_epoch):
                        continue

                self.lidar_accumulator.add(pts_concat, lbl_concat, tag=slot_id)
                self._mark_lidar_seen(sensor_id, fid)
                
                # 🚀 Libération mémoire immédiate après ajout
                del pts_concat, lbl_concat, pts_robot_hits, pts_robot_empty
                del lbl_hits, lbl_empty, pts_local, arr

            except Exception:
                try:
                    print("Erreur LiDAR worker:")
                    traceback.print_exc()
                except Exception:
                    pass
            finally:
                try:
                    self._lidar_q.task_done()
                except Exception:
                    pass




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

        # Homogeneous points
        pts4 = np.ones((pts_local.shape[0], 4), dtype=np.float32)
        pts4[:, :3] = pts_local.astype(np.float32, copy=False)

        # Deux conventions possibles selon la convention des matrices retournées par CARLA.
        # 1) colonne: p' = M * p
        pts_robot4_col = (M_sr @ pts4.T).T
        # 2) ligne:   p' = p * M   => équivalent numpy: pts4 @ M.T
        pts_robot4_row = (pts4 @ M_sr.T)

        # Debug ponctuel (capé) pour comprendre un éventuel problème de transpose
        if self.debug_lidar_transforms and self._dbg_tf_prints_left > 0:
            try:
                start_f = int(DEBUG_LIDAR_TRANSFORMS_START_FRAME)
                cur_f = int(self.target_lidar_frame or -1)
                if cur_f != -1 and cur_f < start_f:
                    return pts_robot4_col[:, :3]

                # évite de spammer plusieurs fois le même frame
                if self._dbg_tf_last_frame_printed is None or self._dbg_tf_last_frame_printed != int(self.target_lidar_frame or -1):
                    self._dbg_tf_last_frame_printed = int(self.target_lidar_frame or -1)
                    self._dbg_tf_prints_left -= 1

                    # petit échantillon pour stats rapides
                    n = int(pts4.shape[0])
                    m = min(n, 2048)
                    if n > m:
                        idx = np.random.choice(n, size=m, replace=False)
                        c_col = pts_robot4_col[idx, :3]
                        c_row = pts_robot4_row[idx, :3]
                        c_loc = pts_local[idx]
                    else:
                        c_col = pts_robot4_col[:, :3]
                        c_row = pts_robot4_row[:, :3]
                        c_loc = pts_local

                    # ROI attendue (celle du dataset implicite)
                    x_min_d, x_max_d = -16.0, 16.0
                    y_min_d, y_max_d = -16.0, 16.0
                    z_min_d, z_max_d = -1.0, 3.0

                    def _roi_ratio(p3: np.ndarray) -> float:
                        if p3.size == 0:
                            return 0.0
                        x = p3[:, 0]; y = p3[:, 1]; z = p3[:, 2]
                        return float(((x >= x_min_d) & (x <= x_max_d) & (y >= y_min_d) & (y <= y_max_d) & (z >= z_min_d) & (z <= z_max_d)).mean())

                    r_col = _roi_ratio(c_col)
                    r_row = _roi_ratio(c_row)

                    st_loc = getattr(sensor_transform, 'location', None)
                    st_rot = getattr(sensor_transform, 'rotation', None)

                    print("\n" + "=" * 70)
                    print("[TF DBG] Compare conventions (col vs row/transposée)")
                    print(f"[TF DBG] target_lidar_frame={self.target_lidar_frame} sample={m}/{n} bank_offset_world={self.bank_global_offset_world}")
                    if st_loc is not None and st_rot is not None:
                        print(f"[TF DBG] sensor_tf loc=({st_loc.x:.3f},{st_loc.y:.3f},{st_loc.z:.3f}) rot(p,y,r)=({st_rot.pitch:.2f},{st_rot.yaw:.2f},{st_rot.roll:.2f})")
                    try:
                        rr = self.reference_robot_transform
                        print(f"[TF DBG] robot(T0) loc=({rr['location']['x']:.3f},{rr['location']['y']:.3f},{rr['location']['z']:.3f}) rot(p,y,r)=({rr['rotation']['pitch']:.2f},{rr['rotation']['yaw']:.2f},{rr['rotation']['roll']:.2f})")
                    except Exception:
                        pass

                    def _summ(name: str, p3: np.ndarray, roi: float):
                        if p3.size == 0:
                            print(f"[TF DBG] {name}: empty")
                            return
                        mn = p3.min(axis=0)
                        mx = p3.max(axis=0)
                        mu = p3.mean(axis=0)
                        print(f"[TF DBG] {name}: roi={roi*100:.3f}% mean=({mu[0]:.2f},{mu[1]:.2f},{mu[2]:.2f}) x[{mn[0]:.1f},{mx[0]:.1f}] y[{mn[1]:.1f},{mx[1]:.1f}] z[{mn[2]:.1f},{mx[2]:.1f}]")

                    _summ("local", c_loc, 0.0)
                    _summ("robot_col(M@p)", c_col, r_col)
                    _summ("robot_row(p@M.T)", c_row, r_row)
                    print("=" * 70 + "\n")
            except Exception:
                # debug doit jamais casser la capture
                pass

        # Convention utilisée (actuelle): colonne.
        return pts_robot4_col[:, :3]



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
        try:
            robot_tf = carla.Transform(
                carla.Location(**position["location"]),
                carla.Rotation(**position["rotation"])
            )
            self.reference_robot_M_wr = np.array(robot_tf.get_inverse_matrix(), dtype=np.float32)
        except Exception:
            self.reference_robot_M_wr = None

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
                if self.allow_extra_maintenance_ticks:
                    self.world.tick()
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

                    for s in self.lidar_slot_ids:
                        rig_list = []

                        for i, cfg in enumerate(self.LIDAR_CONFIGS):
                            lidar_bp = bp_library.find('sensor.lidar.ray_cast_semantic')
                            lidar_bp.set_attribute('channels', str(cfg['channels']))
                            lidar_bp.set_attribute('points_per_second', str(cfg['pps']))
                            lidar_bp.set_attribute('rotation_frequency', '10')
                            lidar_bp.set_attribute('range', str(cfg['range']))
                            lidar_bp.set_attribute('upper_fov', str(cfg['upper_fov']))
                            lidar_bp.set_attribute('lower_fov', str(cfg['lower_fov']))
                            lidar_bp.set_attribute('horizontal_fov', str(cfg['horizontal_fov']))
                            try:
                                lidar_bp.set_attribute('sensor_tick', '0.0')
                            except Exception:
                                pass
                            try:
                                lidar_bp.set_attribute('role_name', 'virtual_sensor')
                            except Exception:
                                pass

                            tf = carla.Transform(
                                start_transform.location + carla.Location(x=cfg['dx'], y=cfg['dy'], z=cfg['dz']),
                                carla.Rotation()
                            )
                            lidar = self.world.spawn_actor(lidar_bp, tf)

                            self.sensor_ids.add(lidar.id)
                            self.lidars.append(lidar)
                            rig_list.append(lidar)
                            self.lidar_actor_to_slot[lidar.id] = int(s)

                            # ✅ CALLBACK PATCHÉ : transform cache + epoch guard
                            def make_cb(sensor, cfg_index=i, slot_id=int(s)):
                                def _cb(data):
                                    if not self.active_lidar_mask[cfg_index]:
                                        return

                                    try:
                                        fid = int(getattr(data, 'frame', -1))
                                        tgt_f = self.target_lidar_frame

                                        # Mode strict: n'accepte aucune frame hors fenêtre cible.
                                        if self.require_target_lidar_frame and tgt_f is None:
                                            return

                                        # filtre frame strict si demandé
                                        if tgt_f is not None and fid != int(tgt_f):
                                            return

                                        # ✅ epoch snapshot + anti-doublon (1 message / capteur / frame)
                                        with self.lock:
                                            epoch_now = int(self._accum_epoch)
                                            fid_i = int(fid)
                                            sid_i = int(sensor.id)

                                            old_frames = [k for k in self._lidar_enqueued_by_frame.keys() if int(k) < (fid_i - 2)]
                                            for k in old_frames:
                                                self._lidar_enqueued_by_frame.pop(k, None)

                                            per_frame_seen = self._lidar_enqueued_by_frame.setdefault(fid_i, set())
                                            if sid_i in per_frame_seen:
                                                return
                                            per_frame_seen.add(sid_i)

                                            # Snapshot du repère robot(T0) au moment où on accepte cette frame.
                                            M_wr = None
                                            if self._accum_reference_robot_M_wr is not None:
                                                M_wr = np.array(self._accum_reference_robot_M_wr, dtype=np.float32, copy=True)
                                            elif self.reference_robot_M_wr is not None:
                                                M_wr = np.array(self.reference_robot_M_wr, dtype=np.float32, copy=True)
                                            bank_off = tuple(self.bank_global_offset_world)
                                            empty_k = int(self.empty_points_per_hit)

                                        if M_wr is None:
                                            with self.lock:
                                                self._lidar_enqueued_by_frame.setdefault(int(fid), set()).discard(int(sensor.id))
                                            return

                                        # ✅ TRANSFORM ROBUSTE : on utilise la transform attendue (celle qu'on a set_transform)
                                        # pour éliminer les oscillations dues à data.transform parfois stale.
                                        with self.lock:
                                            M_sw = None
                                            per_frame = self._lidar_expected_M_sw_by_frame.get(int(fid), None)
                                            if per_frame is not None:
                                                M_sw = per_frame.get(int(sensor.id), None)
                                            if M_sw is None:
                                                M_sw = self._lidar_expected_M_sw.get(int(sensor.id), None)
                                        if M_sw is None:
                                            # fallback (au cas où) : data.transform, puis cache.
                                            sensor_tf = getattr(data, 'transform', None)
                                            if sensor_tf is None:
                                                with self.lock:
                                                    if epoch_now != int(self._accum_epoch):
                                                        return
                                                    sensor_tf = self._lidar_tf_cache.get((fid, int(sensor.id)), None)
                                            if sensor_tf is None:
                                                with self.lock:
                                                    self._lidar_enqueued_by_frame.setdefault(int(fid), set()).discard(int(sensor.id))
                                                return
                                            try:
                                                M_sw = np.array(sensor_tf.get_matrix(), dtype=np.float32)
                                            except Exception:
                                                with self.lock:
                                                    self._lidar_enqueued_by_frame.setdefault(int(fid), set()).discard(int(sensor.id))
                                                return

                                        raw_bytes = self._try_bytes(getattr(data, 'raw_data', b""))
                                        if not raw_bytes:
                                            self._mark_lidar_seen(int(sensor.id), int(fid))
                                            return

                                        max_range_m = float(self.LIDAR_CONFIGS[int(cfg_index)].get('range', self.params.get('range', 24.0)))
                                        max_range_m = max(1.0, max_range_m)

                                        # Fast path: callback ultra-légère, traitement complet dans les workers.
                                        task = (
                                            raw_bytes,
                                            int(fid),
                                            int(sensor.id),
                                            int(slot_id),
                                            int(cfg_index),
                                            int(epoch_now),
                                            np.array(M_sw, dtype=np.float32, copy=True),
                                            (None if M_wr is None else np.array(M_wr, dtype=np.float32, copy=True)),
                                            tuple(bank_off),
                                            int(empty_k),
                                            float(max_range_m),
                                        )
                                        if self._enqueue_lidar_task(task):
                                            return

                                        dtype = np.dtype([
                                            ('x', np.float32), ('y', np.float32), ('z', np.float32),
                                            ('CosAngle', np.float32),
                                            ('ObjIdx', np.uint32), ('ObjTag', np.uint32)
                                        ])
                                        arr = np.frombuffer(raw_bytes, dtype=dtype)
                                        n = int(arr.shape[0])
                                        if n <= 0:
                                            self._mark_lidar_seen(int(sensor.id), int(fid))
                                            return

                                        pts_local = np.empty((n, 3), dtype=np.float32)
                                        pts_local[:, 0] = arr['x']
                                        pts_local[:, 1] = arr['y']
                                        pts_local[:, 2] = arr['z']
                                        lbl_hits = arr['ObjTag'].astype(np.uint8, copy=False)

                                        d = np.linalg.norm(pts_local, axis=1).astype(np.float32)
                                        valid_hit = np.isfinite(d) & (d > 1e-3) & (d <= (max_range_m + 1e-3))
                                        if not np.any(valid_hit):
                                            self._mark_lidar_seen(int(sensor.id), int(fid))
                                            return

                                        pts_local_valid = pts_local[valid_hit]
                                        lbl_hits_valid = lbl_hits[valid_hit]

                                        pts_robot_hits = self._to_robot_frame_cached(
                                            pts_local_valid, M_sw, M_wr, bank_off, dbg_frame=int(fid), out_dtype=np.float16
                                        )

                                        pts_robot_empty = np.zeros((0, 3), dtype=np.float16)
                                        lbl_empty = np.zeros((0,), dtype=np.uint8)
                                        k = int(empty_k)
                                        if k > 0 and pts_local_valid.shape[0] > 0:
                                            n_valid = int(pts_local_valid.shape[0])
                                            k_actual = min(k, max(1, int(20000 / max(n_valid, 1))))
                                            d_valid = np.linalg.norm(pts_local_valid, axis=1).astype(np.float32)
                                            s_max = np.minimum(0.98, max_range_m / (d_valid + 1e-6)).astype(np.float32)
                                            r = np.random.rand(n_valid, k_actual).astype(np.float32)
                                            t = r * s_max[:, None]
                                            pts_empty_local = (pts_local_valid[:, None, :] * t[..., None]).reshape(-1, 3)
                                            pts_robot_empty = self._to_robot_frame_cached(
                                                pts_empty_local, M_sw, M_wr, bank_off, dbg_frame=int(fid), out_dtype=np.float16
                                            )
                                            lbl_empty = np.full((len(pts_robot_empty),), LIDAR_EMPTY_SENTINEL, dtype=np.uint8)

                                        # UNKNOWN désactivé — on ne génère que Hits + Empty

                                        pts_concat = np.vstack([pts_robot_hits, pts_robot_empty])
                                        lbl_concat = np.hstack([lbl_hits_valid, lbl_empty])

                                        with self.lock:
                                            epoch_ok = int(epoch_now) == int(self._accum_epoch)
                                        if not epoch_ok:
                                            self._mark_lidar_seen(int(sensor.id), int(fid))
                                            return

                                        self.lidar_accumulator.add(pts_concat, lbl_concat, tag=slot_id)
                                        self._mark_lidar_seen(int(sensor.id), int(fid))
                                    except Exception:
                                        try:
                                            self._mark_lidar_seen(int(sensor.id), int(getattr(data, 'frame', -1)))
                                        except Exception:
                                            pass
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
                        # cam_bp.set_attribute('shutter_speed', '500')   # 1/10s

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
                                t_cam_cb = time.perf_counter()
                                
                                try:
                                    
                                    tgt = self.target_cam_frame
                                    # print(f"[Debug] Camera TGT: {tgt}")
                                    if tgt is None:
                                        return
                                    
                                    fid = int(image.frame)
                                    
                                    # ✅ règle robuste : on accepte la 1ère image ">= tgt" (pas besoin de tolérance ±1)
                                    # et on n’accepte qu’une seule fois par caméra pour cette pose
                                    with self.lock:
                                        if self.camera_raw_received.get(name, False):
                                            return
                                        if self.camera_pending.get(name, False):
                                            return
                                        # Tolérance d'1-2 frames en mode one-tick pour éviter les ratés
                                        # quand la callback caméra arrive légèrement en retard.
                                        min_fid = int(tgt) - 2
                                        if fid < min_fid:
                                            return
                                        # Marque "in-flight" pour éviter de ré-enqueue à chaque tick
                                        self.camera_pending[name] = True

                                    # Conversion + downsample + encodage JPEG directement dans callback.
                                    image.convert(carla.ColorConverter.Raw)
                                    # print(f"[Debug] Camera image size before processing: {image.width}x{image.height} (frame={fid})")
                                    t0_proc = time.perf_counter()

                                    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                                    arr = arr.reshape((image.height, image.width, 4))
                                    bgr = arr[:, :, :3]

                                    t0_resize = time.perf_counter()
                                    # 🚀 OPTIM: downsample adaptatif selon le facteur de supersampling
                                    ssf = self.cam_supersample_factor
                                    if ssf == 4 and image.width == self.cam_out_w * 4 and image.height == self.cam_out_h * 4:
                                        small = cv2.pyrDown(cv2.pyrDown(bgr))
                                    elif ssf == 2 and image.width == self.cam_out_w * 2 and image.height == self.cam_out_h * 2:
                                        small = cv2.pyrDown(bgr)
                                    elif ssf == 1 and image.width == self.cam_out_w and image.height == self.cam_out_h:
                                        small = bgr  # Pas de resize nécessaire
                                    else:
                                        small = cv2.resize(
                                            bgr,
                                            (self.cam_out_w, self.cam_out_h),
                                            interpolation=cv2.INTER_AREA,
                                        )
                                    self.perf.add_callback('camera_resize', time.perf_counter() - t0_resize, name)

                                    t0_encode = time.perf_counter()
                                    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                    self.perf.add_callback('camera_jpeg_encode', time.perf_counter() - t0_encode, name)
                                    # print("[Debug] JPEG encode ok: " + str(ok))
                                    with self.lock:
                                        if ok and buf is not None:
                                            self.camera_jpeg[name] = buf.tobytes()
                                            self.camera_jpeg_frame_id[name] = fid
                                            self.camera_frame_id[name] = fid
                                            self.camera_raw_received[name] = True
                                            # print(f"[Debug] Camera uf : buffer size={len(self.camera_jpeg[name])} bytes")
                                        self.camera_pending[name] = False
                                        

                                    self.perf.add_callback('camera_process_total', time.perf_counter() - t0_proc, name)
                                    
                                    dt_total_cam = time.perf_counter() - t_cam_cb
                                    self.perf.add_callback('camera_callback_total', dt_total_cam, name)

                                except Exception:
                                    print(f"Erreur dans camera callback (name={name}):")
                                    traceback.print_exc()
                                    with self.lock:
                                        if name in self.camera_pending:
                                            self.camera_pending[name] = False
                                    pass
                            return _cb

                        cb = make_cam_cb(cfg['name'])
                        self.camera_callbacks[cfg['name']] = cb
                        cam.listen(cb)

                self.sensors_created = True
                # Calibre une fois la convention d'application des matrices CARLA.
                self.calibrate_matrix_apply_mode_once()
                print(f"✅ {len(self.lidars)} LiDARs et {len(self.cameras)} caméras créés")
                if self.allow_extra_maintenance_ticks:
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
                new_tf = carla.Transform(sensor_loc_world, sensor_rot_world)
                lidar.set_transform(new_tf)

                # Cache la matrice attendue (sensor->world) pour ce capteur.
                # Utilisée ensuite côté callback pour éviter les incohérences.
                try:
                    M_sw = np.array(new_tf.get_matrix(), dtype=np.float32)
                    with self.lock:
                        self._lidar_expected_M_sw[int(lidar.id)] = M_sw
                except Exception:
                    pass

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
        yaw_jitter=(-45.0, 45.0),
        roll_jitter=(-5.0, 5.0),
        channels_range=(128, 256),
        upper_fov=(40.0, 60.0),
        lower_fov=(-60.0, -40.0),
        pps_range=(500_000, 2_100_000),
        rotation_frequency='10',
    ):
        for lidar in self.lidars:
            if not (lidar and lidar.is_alive):
                continue

            # Jitter en LOCAL (plus logique si on applique un offset global anticollision après)
            # Sinon on mélange deux systèmes de coordonnées.
            
            # On récupère transform initiale (ou courante)
            # Pour faire propre: on applique le jitter sur la pose Relative au rig, pas world.
            # MAIS carla.Actor.set_transform est en WORLD.
            # Si on veut jitter "autour" de la position actuelle, on le fait ici.
            
            # 1. Get current transform (Includes potentially global offset if already moved)
            # BUT wait, this function is called usually AFTER move_all_lidar_rigs_final(..., offset)
            # If so, lidar.get_transform() is in WORLD with offset.
            # If we simply add jitter to world x,y,z, we are fine regarding physics, 
            # BUT `_to_robot_frame_cached` does: `pts_world - bank_offset`.
            # If we jitter world: `true_pos = base + offset + jitter`.
            # `decoder = (true_pos - offset) = base + jitter`.
            # So the jitter IS preserved in the robot frame reconstruction.
            #
            # The user criticism was: "si le jitter est voulu (tu veux le jitter réel)... Mais ce qui devient incohérent c’est l’objectif de l’offset global... Si derrière tu jitters en world, tu recrées potentiellement des collisions".
            #
            # Solution: Apply jitter BEFORE global offset.
            # But the sensors are ALREADY at `base + offset`.
            # To fix cleanly: `randomize` should be called BEFORE `move_all_lidar_rigs_final` or
            # `move_all_lidar_rigs_final` should handle jitter.
            #
            # As a quick fix respecting the request: "ne randomize pas en WORLD après ou randomize en LOCAL".
            # I will assume this function is called when sensors are at their nominal position (or I just jitter logically).
            #
            # Let's apply jitter to the Transform, but be careful.
            
            tf = lidar.get_transform()
            loc = tf.location
            rot = tf.rotation

            # Apply small local perturbations
            loc.x += random.uniform(*loc_jitter_x)
            loc.y += random.uniform(*loc_jitter_y)
            loc.z += random.uniform(*loc_jitter_z)
            
            pass # Keep world jitter for now but reduce it if needed, or better:
            # The critique says: "Si l’offset global est virtuel... ne randomize pas en WORLD après".
            # The user code calls `move_all_lidar_rigs_final` THEN `randomize`.
            # I will move the call order in `generate` and here just apply jitter.
            
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

            new_tf = carla.Transform(loc, rot)
            lidar.set_transform(new_tf)
            try:
                M_sw = np.array(new_tf.get_matrix(), dtype=np.float32)
                with self.lock:
                    self._lidar_expected_M_sw[int(lidar.id)] = M_sw
            except Exception:
                pass

    # def start_new_accumulation(self, target_points: Optional[int] = None):
    #     self.lidar_accumulator.reset(target_points or self.capture_points_target)
    #     # Par défaut on accepte toutes les frames LiDAR, le générateur peut fixer un target précis.
    #     self.target_lidar_frame = None
    #     for name in self.camera_received:
    #         self.camera_received[name] = False
    #         self.camera_pending[name] = False
    #         self.camera_frame_id[name] = -1
    #         self.camera_jpeg_frame_id[name] = -1
    #         self.camera_jpeg[name] = None

    def capture_current_frame(self, weather_preset=None):
        with SectionTimer(self.perf, "accumulate_lidar_get"):
            points, labels = self.lidar_accumulator.get()
        if points is None or len(points) == 0:
            print(" ❌ Pas de points LiDAR")
            return None
        # print(f"[Debug] ✅ {len(points):,} points LiDAR (hits + empty) accumulés (repère ROBOT)")

        # On ne bloque PAS ici sur l'encodage JPEG.
        # Les JPEG seront attendus côté thread de sauvegarde (AsyncWriter), ce qui
        # permet à la boucle principale de continuer à tick/accumuler.
        with self.lock:
            images = {}
            if self.target_cam_frame is not None:
                for name in self.camera_raw_received.keys():
                    jpg = self.camera_jpeg.get(name, None)
                    # print(f"[Debug] Camera {name}: JPEG {'ready' if jpg else 'missing'}")
                    if jpg:
                        images[name] = jpg

            if self.target_cam_frame is None:
                expected_cam_frames = {}
            else:
                expected_cam_frames = {
                    name: int(self.target_cam_frame)
                    for name in self.camera_raw_received.keys()
                }
        # print(f"[Debug] ✅ {len(images)} images prêtes (JPEG)")
        return {
            'points': points,
            'labels': labels,
            'images': images,
            'expected_cam_frames': expected_cam_frames,
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
# NUMBA-ACCELERATED RAYCASTING (voxelisation)
# ==========================
def _numba_raycast_occlusion_python(centers, ego_vx, ego_vy, ego_vz, occ_flat, nx, ny, nz):
    """Fallback pur-Python si Numba n'est pas disponible (ne devrait pas \u00eatre appel\u00e9)."""
    raise NotImplementedError("Should not be called when HAS_NUMBA is False")

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _numba_raycast_occlusion(centers, ego_vx, ego_vy, ego_vz, occ_flat, nx, ny, nz):
        """
        Ray-marching param\u00e9trique Numba-JIT pour classifier les voxels non-observ\u00e9s.
        Utilise un raycasting 3D-DDA (Amanatides & Woo) pour chaque voxel,
        beaucoup plus rapide que le sampling param\u00e9trique numpy.

        Args:
            centers:  (N, 3) float32 - centres des voxels non-observ\u00e9s en coordonn\u00e9es voxel
            ego_vx/vy/vz: position ego en coordonn\u00e9es voxel
            occ_flat: (nx*ny*nz,) uint8 - grille d'occupation aplatie (iz * nx*ny + iy * nx + ix)
            nx, ny, nz: dimensions de la grille

        Returns:
            is_occluded: (N,) bool
        """
        N = centers.shape[0]
        is_occluded = np.zeros(N, dtype=np.bool_)
        nxy = nx * ny

        for idx in prange(N):
            cx = centers[idx, 0]
            cy = centers[idx, 1]
            cz = centers[idx, 2]

            # Direction ego -> centre voxel
            dx = cx - ego_vx
            dy = cy - ego_vy
            dz = cz - ego_vz

            # Longueur du rayon
            ray_len = math.sqrt(dx * dx + dy * dy + dz * dz)
            if ray_len < 1e-6:
                continue

            inv_len = 1.0 / ray_len
            dx *= inv_len
            dy *= inv_len
            dz *= inv_len

            # Position de d\u00e9part (ego)
            px = ego_vx
            py = ego_vy
            pz = ego_vz

            # Voxel courant
            vx_i = int(math.floor(px))
            vy_i = int(math.floor(py))
            vz_i = int(math.floor(pz))

            # Voxel cible
            tx_i = int(math.floor(cx))
            ty_i = int(math.floor(cy))
            tz_i = int(math.floor(cz))

            # Direction des pas
            step_x = 1 if dx >= 0 else -1
            step_y = 1 if dy >= 0 else -1
            step_z = 1 if dz >= 0 else -1

            # t_max: distance param\u00e9trique jusqu'au prochain plan de voxel
            if abs(dx) > 1e-10:
                if dx > 0:
                    t_max_x = (float(vx_i + 1) - px) / dx
                else:
                    t_max_x = (float(vx_i) - px) / dx
                t_delta_x = abs(1.0 / dx)
            else:
                t_max_x = 1e30
                t_delta_x = 1e30

            if abs(dy) > 1e-10:
                if dy > 0:
                    t_max_y = (float(vy_i + 1) - py) / dy
                else:
                    t_max_y = (float(vy_i) - py) / dy
                t_delta_y = abs(1.0 / dy)
            else:
                t_max_y = 1e30
                t_delta_y = 1e30

            if abs(dz) > 1e-10:
                if dz > 0:
                    t_max_z = (float(vz_i + 1) - pz) / dz
                else:
                    t_max_z = (float(vz_i) - pz) / dz
                t_delta_z = abs(1.0 / dz)
            else:
                t_max_z = 1e30
                t_delta_z = 1e30

            # Traverser les voxels avec DDA jusqu'au voxel cible
            max_steps = nx + ny + nz  # Borne sup\u00e9rieure du nombre de pas
            for _ in range(max_steps):
                # V\u00e9rifier si on a atteint le voxel cible
                if vx_i == tx_i and vy_i == ty_i and vz_i == tz_i:
                    break

                # Avancer dans la direction qui a le plus petit t_max
                if t_max_x < t_max_y:
                    if t_max_x < t_max_z:
                        vx_i += step_x
                        t_max_x += t_delta_x
                    else:
                        vz_i += step_z
                        t_max_z += t_delta_z
                else:
                    if t_max_y < t_max_z:
                        vy_i += step_y
                        t_max_y += t_delta_y
                    else:
                        vz_i += step_z
                        t_max_z += t_delta_z

                # Hors grille?
                if vx_i < 0 or vx_i >= nx or vy_i < 0 or vy_i >= ny or vz_i < 0 or vz_i >= nz:
                    break

                # V\u00e9rifier si ce voxel est occup\u00e9
                flat_id = vz_i * nxy + vy_i * nx + vx_i
                if occ_flat[flat_id]:
                    is_occluded[idx] = True
                    break

        return is_occluded


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
        cam_supersample_factor: int = 4,
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
        lidar_empty_points_per_hit: int = 1,
        camera_stride: int = 1,
        fixed_delta_seconds: float = 0.1,
        carla_substepping: bool = False,
        carla_max_substep_delta_time: float = 0.02,
        carla_max_substeps: int = 1,
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
            z_range=(-1.0, 3.0),
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
        self.cam_supersample_factor = max(1, int(cam_supersample_factor))

        self.window_back = int(window_back)
        self.window_forward = int(window_forward)
        self.proximity_radius = float(proximity_radius)
        self.lidar_layout_if_clear = lidar_layout_if_clear
        self.allowed_semantic_tags = allowed_semantic_tags
        self.max_ticks_per_pose = int(max_ticks_per_pose)
        self.randomize_clear_poses = bool(randomize_clear_poses)

        # Capture caméra à chaque pose.
        self.camera_stride = max(1, int(camera_stride))

        # Mode forcé: toujours 1 tick par pose.
        self.one_cam_tick_per_pose = True

        self.fixed_delta_seconds = max(0.01, float(fixed_delta_seconds))
        self.carla_substepping = bool(carla_substepping)
        self.carla_max_substep_delta_time = max(0.001, float(carla_max_substep_delta_time))
        self.carla_max_substeps = max(1, int(carla_max_substeps))
        # En mode 1 tick forcé, pas de ticks de maintenance supplémentaires.
        self.allow_extra_maintenance_ticks = False

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

                # Résolution automatique du nom de map (accepte noms courts comme "Town10")
                available_maps = self.client.get_available_maps()
                match = [m for m in available_maps if self.map_name in m]
                if not match:
                    print(f"❌ Map '{self.map_name}' introuvable. Maps disponibles :")
                    for m in sorted(available_maps):
                        print(f"   - {m}")
                    return False
                resolved_map = match[0]
                print(f"🗺️  Map résolue : {self.map_name} → {resolved_map}")

                with SectionTimer(self.perf, "load_world"):
                    self.world = self.client.load_world(resolved_map)
                print("✅ Monde CARLA prêt")

                self._deep_cleanup_world()

                with SectionTimer(self.perf, "apply_world_settings"):
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    # Utilise le paramètre CLI/constructeur (évite le hardcode).
                    settings.fixed_delta_seconds = float(self.fixed_delta_seconds)
                    # ⚠️ GARDER FALSE: no_rendering_mode=True casse les LiDARs!
                    settings.no_rendering_mode = False
                    
                    # Réglages CARLA pour accélérer world.tick() sans changer les capteurs.
                    settings.substepping = bool(self.carla_substepping)
                    settings.max_substep_delta_time = float(self.carla_max_substep_delta_time)
                    settings.max_substeps = int(self.carla_max_substeps)
                    
                    self.world.apply_settings(settings)
                    print(
                        f"⚙️ World settings: fixed_delta_seconds={settings.fixed_delta_seconds:.3f}, "
                        f"substep={settings.substepping}, "
                        f"max_substep_dt={settings.max_substep_delta_time:.4f}, "
                        f"max_substeps={settings.max_substeps}, "
                        f"no_rendering={settings.no_rendering_mode}"
                    )

                    
                with SectionTimer(self.perf, "apply_weather"):
                    self.weather_manager = WeatherManager(
                        self.world,
                        apply_settle_tick=self.allow_extra_maintenance_ticks,
                    )
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
            if self.allow_extra_maintenance_ticks:
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
        # gc.collect()

    # ---------- logique ego -----------

    # ---------- NPC SPAWNER (other_actors du JSON) ----------

    def _init_npc_state(self):
        """Initialise le state pour le spawn/move des NPCs."""
        self._npc_actors: Dict[int, carla.Actor] = {}  # actor_id_json -> carla.Actor
        self._npc_controllers: Dict[int, carla.Actor] = {}  # actor_id_json -> walker controller
        self._bp_cache: Dict[str, Optional[object]] = {}  # type_id -> blueprint (ou None si introuvable)

    def _get_blueprint(self, type_id: str):
        """Cherche un blueprint CARLA par type_id, avec cache et fallback."""
        if type_id in self._bp_cache:
            return self._bp_cache[type_id]

        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find(type_id) if type_id else None

        if bp is None:
            # Fallback : chercher par sous-chaîne
            # ex: "vehicle.mitsubishi.fusorosa" -> chercher dans la lib
            candidates = bp_library.filter(type_id)
            if candidates:
                bp = candidates[0]

        self._bp_cache[type_id] = bp
        return bp

    def _spawn_or_move_npcs(self, position_data: dict):
        """
        Spawn ou téléporte les other_actors pour la frame courante.
        - Les acteurs déjà spawnés sont téléportés.
        - Les nouveaux sont spawnés.
        - Les acteurs absents de cette frame sont détruits.
        """
        other_actors = position_data.get('other_actors', [])
        if not other_actors and not self._npc_actors:
            return

        current_ids = set()

        for actor_info in other_actors:
            json_id = int(actor_info['actor_id'])
            current_ids.add(json_id)

            loc = actor_info['location']
            rot = actor_info['rotation']
            tf = carla.Transform(
                carla.Location(x=float(loc['x']), y=float(loc['y']), z=float(loc['z'])),
                carla.Rotation(
                    yaw=float(rot.get('yaw', 0)),
                    pitch=float(rot.get('pitch', 0)),
                    roll=float(rot.get('roll', 0)),
                )
            )

            # Acteur déjà spawné → téléporter
            if json_id in self._npc_actors:
                actor = self._npc_actors[json_id]
                try:
                    if actor.is_alive:
                        actor.set_transform(tf)
                        # Figer la vélocité (pas de physique entre les ticks)
                        vel = actor_info.get('velocity', {})
                        actor.set_target_velocity(carla.Vector3D(
                            x=float(vel.get('x', 0)),
                            y=float(vel.get('y', 0)),
                            z=float(vel.get('z', 0)),
                        ))
                        continue
                    else:
                        # Acteur mort → le retirer et re-spawner
                        self._npc_actors.pop(json_id, None)
                        self._npc_controllers.pop(json_id, None)
                except Exception:
                    self._npc_actors.pop(json_id, None)
                    self._npc_controllers.pop(json_id, None)

            # Nouveau spawn
            type_id = actor_info.get('type', '')
            bp = self._get_blueprint(type_id)
            if bp is None:
                continue

            # Désactiver la physique si possible (on contrôle la position)
            is_vehicle = 'vehicle.' in type_id
            is_walker = 'walker.' in type_id

            try:
                if is_vehicle:
                    # Relever un peu le z pour éviter le clip dans le sol
                    tf.location.z += 0.5
                    actor = self.world.try_spawn_actor(bp, tf)
                    if actor is not None:
                        actor.set_simulate_physics(False)
                        self._npc_actors[json_id] = actor
                elif is_walker:
                    actor = self.world.try_spawn_actor(bp, tf)
                    if actor is not None:
                        actor.set_simulate_physics(False)
                        self._npc_actors[json_id] = actor
                else:
                    # Autre type (props, etc.)
                    actor = self.world.try_spawn_actor(bp, tf)
                    if actor is not None:
                        self._npc_actors[json_id] = actor

            except Exception as e:
                # Spawn échoué (collision, blueprint invalide, etc.) → skip
                pass

        # Détruire les acteurs qui ne sont plus dans la scène
        stale_ids = set(self._npc_actors.keys()) - current_ids
        if stale_ids:
            batch = []
            for sid in stale_ids:
                ctrl = self._npc_controllers.pop(sid, None)
                actor = self._npc_actors.pop(sid, None)
                if ctrl is not None:
                    try:
                        batch.append(carla.command.DestroyActor(ctrl))
                    except Exception:
                        pass
                if actor is not None:
                    try:
                        batch.append(carla.command.DestroyActor(actor))
                    except Exception:
                        pass
            if batch:
                try:
                    self.client.apply_batch_sync(batch, True)
                except Exception:
                    pass

    def _cleanup_all_npcs(self):
        """Détruit tous les NPCs spawnés."""
        if not hasattr(self, '_npc_actors'):
            return
        batch = []
        for ctrl in self._npc_controllers.values():
            try:
                batch.append(carla.command.DestroyActor(ctrl))
            except Exception:
                pass
        for actor in self._npc_actors.values():
            try:
                batch.append(carla.command.DestroyActor(actor))
            except Exception:
                pass
        if batch:
            try:
                self.client.apply_batch_sync(batch, True)
            except Exception:
                pass
        self._npc_actors.clear()
        self._npc_controllers.clear()
        print(f"🧹 NPCs nettoyés ({len(batch)} acteurs détruits)")

    # ---------- (fin NPC spawner) ----------

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

    # ================================================================
    #  CLASSIFICATION DES VOXELS NON-OBSERVÉS (Empty visible / Unknown occluté)
    #  Raycasting vectorisé depuis la position ego (0,0,0 en repère robot)
    # ================================================================

    def _classify_unobserved_voxels(
        self,
        occ_set_flat: np.ndarray,
        observed_set_flat: np.ndarray,
        vx: 'VoxelConfig',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pour chaque voxel de la grille qui n'a AUCUNE observation (ni hit, ni empty, ni unknown),
        détermine si il est :
          - VISIBLE depuis l'ego (0,0,0)  → Empty
          - OCCULTÉ par un voxel occupé   → Unknown

        Version accélérée :
          - Numba JIT (si disponible) pour le ray-marching → ~10-30× plus rapide
          - Sinon fallback numpy avec batch plus gros (16384)

        Args:
            occ_set_flat:      array 1-D (int64) des flat-IDs des voxels occupés (hits)
            observed_set_flat: array 1-D (int64) des flat-IDs de TOUS les voxels observés
                               (hits ∪ empty ∪ unknown)
            vx:                VoxelConfig

        Returns:
            (empty_flat_ids, unknown_flat_ids)  – deux arrays int64
        """
        nx, ny, nz = vx.grid_shape
        vs = float(vx.voxel_size)
        x_min = float(vx.x_range[0])
        y_min = float(vx.y_range[0])
        z_min = float(vx.z_range[0])
        total_voxels = nx * ny * nz

        # --- grille booléenne "observé" ---
        observed_mask = np.zeros(total_voxels, dtype=bool)
        if observed_set_flat.size:
            valid = observed_set_flat[(observed_set_flat >= 0) & (observed_set_flat < total_voxels)]
            observed_mask[valid] = True

        unobs_flat = np.where(~observed_mask)[0].astype(np.int64)
        if unobs_flat.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # --- grille 3-D "occupé" pour le lookup rapide (flat bool) ---
        occ_flat = np.zeros(total_voxels, dtype=np.uint8)
        if occ_set_flat.size:
            occ_v = occ_set_flat[(occ_set_flat >= 0) & (occ_set_flat < total_voxels)]
            occ_flat[occ_v] = 1

        # --- position ego en coordonnées voxel ---
        ego_vx = (0.0 - x_min) / vs
        ego_vy = (0.0 - y_min) / vs
        ego_vz = (0.0 - z_min) / vs

        # --- centres des voxels non-observés (en coord voxel) ---
        nxy = nx * ny
        iz_u = unobs_flat // nxy
        rem_u = unobs_flat % nxy
        iy_u = rem_u // nx
        ix_u = rem_u % nx
        centers = np.stack([ix_u, iy_u, iz_u], axis=1).astype(np.float32) + 0.5

        # --- Choix de la méthode ---
        if HAS_NUMBA:
            is_occluded = _numba_raycast_occlusion(
                centers, ego_vx, ego_vy, ego_vz,
                occ_flat, nx, ny, nz,
            )
        else:
            # Fallback numpy vectorisé, batch plus gros
            ego_v = np.array([ego_vx, ego_vy, ego_vz], dtype=np.float32)
            dirs = centers - ego_v[None, :]
            dists = np.linalg.norm(dirs, axis=1)
            max_dist = float(dists.max()) if dists.size else 0.0
            n_steps = max(int(np.ceil(max_dist * 2.0)), 1)
            t_values = np.linspace(0.0, 1.0, n_steps + 2, dtype=np.float32)[1:-1]

            # Grille 3D pour le lookup
            occ_3d = occ_flat.reshape((nz, ny, nx)).astype(bool)

            BATCH = 16384  # 4× plus gros qu'avant
            is_occluded = np.zeros(unobs_flat.size, dtype=bool)

            for b0 in range(0, unobs_flat.size, BATCH):
                b1 = min(b0 + BATCH, unobs_flat.size)
                b_dirs = dirs[b0:b1]
                samples = ego_v[None, None, :] + t_values[None, :, None] * b_dirs[:, None, :]
                vi = np.empty_like(samples, dtype=np.int32)
                vi[..., 0] = np.clip(samples[..., 0].astype(np.int32), 0, nx - 1)
                vi[..., 1] = np.clip(samples[..., 1].astype(np.int32), 0, ny - 1)
                vi[..., 2] = np.clip(samples[..., 2].astype(np.int32), 0, nz - 1)
                occ_check = occ_3d[vi[..., 2], vi[..., 1], vi[..., 0]]
                is_occluded[b0:b1] = np.any(occ_check, axis=1)

        empty_ids   = unobs_flat[~is_occluded]
        unknown_ids = unobs_flat[is_occluded]
        return empty_ids, unknown_ids

    def _generate_points_in_voxels(
        self,
        voxel_flat_ids: np.ndarray,
        label: int,
        vx: 'VoxelConfig',
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Génère des points aléatoires uniformément répartis dans les voxels indiqués.

        Args:
            voxel_flat_ids: flat IDs des voxels
            label:          classe à affecter (EMPTY_LABEL ou UNKNOWN_LABEL)
            vx:             VoxelConfig
            rng:            numpy Generator

        Returns:
            (pts, lbls) – float16 / uint8
        """
        if voxel_flat_ids.size == 0:
            return np.zeros((0, 3), dtype=np.float16), np.zeros((0,), dtype=np.uint8)

        nx, ny, nz = vx.grid_shape
        vs = float(vx.voxel_size)
        x_min = float(vx.x_range[0])
        y_min = float(vx.y_range[0])
        z_min = float(vx.z_range[0])

        n_vox = voxel_flat_ids.size
        n_pts_per = rng.integers(
            self.points_per_voxel_min,
            self.points_per_voxel_max + 1,
            size=n_vox, dtype=np.int32,
        )
        total = int(n_pts_per.sum())
        if total == 0:
            return np.zeros((0, 3), dtype=np.float16), np.zeros((0,), dtype=np.uint8)

        # décodage flat → ix, iy, iz
        iz = voxel_flat_ids // (nx * ny)
        rem = voxel_flat_ids % (nx * ny)
        iy = rem // nx
        ix = rem % nx

        # origines monde de chaque voxel
        x0 = ix.astype(np.float32) * vs + x_min
        y0 = iy.astype(np.float32) * vs + y_min
        z0 = iz.astype(np.float32) * vs + z_min

        # repeat pour chaque point
        vidx = np.repeat(np.arange(n_vox, dtype=np.int32), n_pts_per)

        # offsets aléatoires uniformes dans [0, vs)
        off = rng.random((total, 3), dtype=np.float32) * vs

        pts = np.empty((total, 3), dtype=np.float16)
        pts[:, 0] = (x0[vidx] + off[:, 0]).astype(np.float16)
        pts[:, 1] = (y0[vidx] + off[:, 1]).astype(np.float16)
        pts[:, 2] = (z0[vidx] + off[:, 2]).astype(np.float16)

        lbls = np.full(total, label, dtype=np.uint8)
        return pts, lbls

    def _build_implicit_from_points(
        self,
        pts_robot: np.ndarray,
        lbl_raw: np.ndarray,
        target_total_points: int,
        debug_tag: str = "",
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
            prefix = f"[{debug_tag}] " if debug_tag else ""
            print(f"{prefix}[TIME] {name:<22s}: {dt:.4f}s")

        prefix = f"[{debug_tag}] " if debug_tag else ""

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
        # Diagnostic léger pour comprendre les frames "quasi vides".
        # (évite les gros prints quand tout va bien)
        in_ratio = float(mask_in.mean())
        if in_ratio < 0.05:
            try:
                print(
                    f"{prefix}   [ROI dbg] in={in_ratio*100:.2f}% | "
                    f"x[{x.min():.1f},{x.max():.1f}] y[{y.min():.1f},{y.max():.1f}] z[{z.min():.1f},{z.max():.1f}]"
                )
            except Exception:
                pass
        if not np.any(mask_in):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint8)

        pts = pts_robot[mask_in]
        lbl_raw = lbl_raw[mask_in]
        toc(t0, "ROI filtering")

        # ---------------- SPLIT ----------------
        t0 = tic()
        mask_empty = (lbl_raw == LIDAR_EMPTY_SENTINEL)
        # If no explicit empty points from worker, use implicit logic?
        # The user critique said: "Générer les empty après voxelisation... marque des voxels free-space"
        # Since I disabled empty generation in worker, mask_empty will be all False.
        # This effectively disables "Empty" voxels unless we add logic here.
        # For this refactor request (focus on perf/logic structure), disabling expensive/wrong generation is better than keeping it.
        # Ideally, we would raycast here, but that is a new feature.
        # We will proceed with existing points (Hits only).

        # Unknown désactivé : on ignore tout point avec label UNKNOWN_SENTINEL
        mask_hit = ~mask_empty

        pts_hits = pts[mask_hit]
        lbl_hits_raw = lbl_raw[mask_hit].astype(np.uint8, copy=False)

        pts_empty = pts[mask_empty]
        pts_unknown_ray = np.zeros((0, 3), dtype=np.float32)  # plus de points Unknown

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
        toc(t0, "voxelization")

        rng = np.random.default_rng()

        # ================================================================
        # CLASSIFICATION DES VOXELS NON-OBSERVÉS  (Empty visible / Unknown occluté)
        # Raycasting vectorisé depuis ego (0,0,0) à travers la grille occupée.
        # On génère ensuite des points synthétiques dans ces voxels.
        # ================================================================
        t0_unobs = tic()

        # Ensemble de tous les voxels qui ont AU MOINS une observation (hit ∪ empty)
        occ_set_flat   = np.unique(flat_hit) if flat_hit.size else np.array([], dtype=np.int64)
        _parts = [p for p in (flat_hit, flat_empty) if p.size > 0]
        observed_flat  = np.unique(np.concatenate(_parts)) if _parts else np.array([], dtype=np.int64)

        new_empty_ids, new_unknown_ids = self._classify_unobserved_voxels(
            occ_set_flat, observed_flat, vx,
        )

        # Unknown désactivé : les voxels occultés sont fusionnés en Empty
        new_empty_ids = np.concatenate([new_empty_ids, new_unknown_ids]) if new_unknown_ids.size else new_empty_ids
        new_unknown_ids = np.array([], dtype=np.int64)

        # Sous-échantillonnage avec les ratios de conservation
        if new_empty_ids.size and self.voxel_keep_ratio_empty < 1.0:
            keep_n = max(1, int(new_empty_ids.size * self.voxel_keep_ratio_empty))
            new_empty_ids = rng.choice(new_empty_ids, size=keep_n, replace=False)

        # Génération de points synthétiques dans les voxels classifiés (Empty uniquement)
        synth_empty_pts, synth_empty_lbl = self._generate_points_in_voxels(
            new_empty_ids, EMPTY_LABEL, vx, rng,
        )
        # Fusion avec les points existants issus du LiDAR
        if synth_empty_pts.shape[0] > 0:
            if pts_empty.shape[0] > 0:
                pts_empty = np.vstack([pts_empty.astype(np.float32), synth_empty_pts.astype(np.float32)])
            else:
                pts_empty = synth_empty_pts.astype(np.float32)
            flat_empty = np.concatenate([flat_empty, voxel_flat_ids(synth_empty_pts.astype(np.float32))])

        print(
            f"{prefix}   [unobs] classifiés: {new_empty_ids.size} voxels empty (Unknown fusionné → Empty) "
            f"→ +{synth_empty_pts.shape[0]} pts empty"
        )
        toc(t0_unobs, "unobs voxel classify+gen")

        # ---------------- OCC LABEL MAP (compressé) ----------------
        t0 = tic()
        uniq_hit = np.zeros((0,), dtype=np.int64)
        hit_labels_per_uniq = np.zeros((0,), dtype=np.uint8)
        if flat_hit.size:
            uniq_hit, first_idx = np.unique(flat_hit, return_index=True)
            hit_labels_per_uniq = lbl_hits_id[first_idx]
        toc(t0, "occ label map")

        # ================= OCC (vectorisé, sans boucle Python) =================
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
                has_upsample = R > 0.0 and p_up > 0.0 and hasattr(self, "_occ_offset_bank")

                # --- BASE: sampling vectorisé sans boucle Python ---
                base_total = int(n_pts_per_vox.sum())

                # Pour chaque point de sortie, on sait de quel voxel il vient
                voxel_id_per_out = np.repeat(np.arange(n_vox, dtype=np.int32), n_pts_per_vox)

                # Nombre de hits par voxel, répété pour chaque point de sortie
                counts_rep = counts[voxel_id_per_out].astype(np.int64, copy=False)

                # Tirage d'un index local aléatoire dans [0, count_voxel)
                local_idx = rng.integers(0, counts_rep, size=base_total, dtype=np.int64)

                # Position globale dans l'array trié : start + local
                global_pos = start_idx[voxel_id_per_out].astype(np.int64, copy=False) + local_idx

                # Gather des points
                chosen_global = order[global_pos]
                out_pts_base = pts_hits[chosen_global].astype(np.float32, copy=False)

                # Labels OCC via mapping compressé vectorisé
                lbl_per_voxel = np.empty(n_vox, dtype=np.uint8)
                if uniq_hit.size:
                    vox_pos = np.searchsorted(uniq_hit, uniq_v)
                    # Clip in case of edge values
                    np.clip(vox_pos, 0, uniq_hit.size - 1, out=vox_pos)
                    lbl_per_voxel = hit_labels_per_uniq[vox_pos]
                else:
                    lbl_per_voxel[:] = UNKNOWN_LABEL

                out_lbl_base = lbl_per_voxel[voxel_id_per_out]

                # --- UPSAMPLE OCC (vectorisé) ---
                if has_upsample:
                    K = self._occ_offset_bank.shape[0]
                    if p_up >= 1.0:
                        n_add_per_vox = n_pts_per_vox.copy()
                    else:
                        n_add_per_vox = rng.binomial(n_pts_per_vox.astype(np.int64), p_up).astype(np.int32)

                    total_add = int(n_add_per_vox.sum())
                    if total_add > 0:
                        # Indices de base pour chaque point upsamplé
                        up_voxel_id = np.repeat(np.arange(n_vox, dtype=np.int32), n_add_per_vox)

                        # Offsets cumulés pour savoir où piocher dans out_pts_base
                        base_offsets = np.empty(n_vox + 1, dtype=np.int64)
                        base_offsets[0] = 0
                        np.cumsum(n_pts_per_vox, out=base_offsets[1:])

                        # Pour chaque point upsamplé, indice local dans le voxel
                        n_pts_rep = n_pts_per_vox[up_voxel_id].astype(np.int64)
                        local_up = rng.integers(0, n_pts_rep, size=total_add, dtype=np.int64)
                        center_idx = base_offsets[up_voxel_id].astype(np.int64) + local_up

                        centers_up = out_pts_base[center_idx]
                        off_idx = rng.integers(0, K, size=total_add, dtype=np.int32)
                        new_pts = centers_up + self._occ_offset_bank[off_idx]

                        np.clip(new_pts[:, 0], x_min, x_max, out=new_pts[:, 0])
                        np.clip(new_pts[:, 1], y_min, y_max, out=new_pts[:, 1])
                        np.clip(new_pts[:, 2], z_min, z_max, out=new_pts[:, 2])

                        up_lbl = lbl_per_voxel[up_voxel_id]

                        out_pts_base = np.vstack([out_pts_base, new_pts])
                        out_lbl_base = np.concatenate([out_lbl_base, up_lbl])

                points_occ = out_pts_base
                labels_occ = out_lbl_base

        toc(t0_occ, "OCC vectorized")

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

        # UNKNOWN désactivé — pas de génération de points Unknown

        n_occ, n_emp = points_occ.shape[0], points_empty.shape[0]
        print(f"{prefix}   → Pools: occ={n_occ} pts, empty={n_emp} pts (avant ratios)")

        # ---------------- ratios / target ----------------
        t0 = tic()
        if target_total_points <= 0:
            pts_final = np.vstack([points_occ, points_empty]).astype(np.float32, copy=False)
            lbl_final = np.hstack([labels_occ, labels_empty]).astype(np.uint8, copy=False)
            toc(t0, "final stack (no target)")
            toc(t_global, "TOTAL")
            return pts_final, lbl_final

        # Ratios: occ + empty = 100%
        r_occ = self.ratio_occ
        r_emp = self.ratio_empty + self.ratio_unknown  # budget unknown redistribué à empty
        s = max(r_occ + r_emp, 1e-6)
        n_occ_target = int(target_total_points * (r_occ / s))
        n_emp_target = int(target_total_points - n_occ_target)

        def sample_class(pts_c: np.ndarray, lbl_c: np.ndarray, target: int):
            if pts_c.shape[0] == 0 or target <= 0:
                return np.zeros((0, 3), dtype=np.float16), np.zeros((0,), dtype=np.uint8)
            if pts_c.shape[0] <= target:
                return pts_c, lbl_c
            idx = rng.choice(pts_c.shape[0], size=target, replace=False)
            return pts_c[idx], lbl_c[idx]

        pts_occ_s, lbl_occ_s = sample_class(points_occ, labels_occ, n_occ_target)
        pts_emp_s, lbl_emp_s = sample_class(points_empty, labels_empty, n_emp_target)

        pts_final = np.vstack([pts_occ_s, pts_emp_s]).astype(np.float16, copy=False)
        lbl_final = np.hstack([lbl_occ_s, lbl_emp_s]).astype(np.uint8, copy=False)

        toc(t0, "final sampling+stack")

        print(
            f"{prefix}   → Points finals (occ/empty): "
            f"{pts_occ_s.shape[0]}/{pts_emp_s.shape[0]} "
            f"(total={pts_final.shape[0]:,} cible={target_total_points:,})"
        )

        toc(t_global, "TOTAL")
        return pts_final, lbl_final









    # ---------- SAVE FRAME ----------

    def save_frame(self, frame_data, frame_id, ref_pos):
            fmt_id = f"{frame_id:06d}"
            pts, lbl = frame_data['points'], frame_data['labels']
            
            # --- LIDAR/OCC (CPU Principal) ---
            target_n = random.randint(self.points_min_saved, self.points_max_saved)
            with SectionTimer(self.perf, "build_implicit_total"):
                pts_f, lbl_f = self._build_implicit_from_points(pts, lbl, target_n, debug_tag=f"frame_{fmt_id}")

            # Sauvegarde points
            np.savez(os.path.join(self.output_dir, "points", f"frame_{fmt_id}.npz"), points=pts_f.astype(np.float16), labels=lbl_f)

            # --- IMAGES ---
            # Les JPEG sont préparés dans le callback caméra, on écrit directement ici.
            # print(" ***** Frame data ***** "+ str(frame_data.keys()))
            # print(" ***** Frame data ***** "+ str(frame_data["images"]))
            # 1) Écrit immédiatement les JPEG déjà présents dans frame_data.
            images_now = frame_data.get('images', {}) or {}
            for cam_name, jpg_bytes in images_now.items():
                if not jpg_bytes:
                    continue
                path = os.path.join(self.output_dir, "images", f"frame_{fmt_id}_{cam_name}.jpg")
                with open(path, "wb") as f:
                    f.write(jpg_bytes)

            expected = frame_data.get('expected_cam_frames', {})
            if expected:
                # 🚀 OPTIM: timeout réduit (2s suffit) + poll rapide (5ms au lieu de 100ms)
                cam_wait_timeout_s = 2.0
                cam_poll_s = 0.005

                for cam_name, expected_fid in expected.items():
                    deadline = time.perf_counter() + cam_wait_timeout_s
                    # Déjà écrit depuis frame_data['images']
                    if cam_name in images_now:
                        continue
                    
                    if expected_fid < 0:
                        continue

                    jpg_bytes = None
                    while time.perf_counter() < deadline:
                        with self.sensor_manager.lock:
                            current_fid = self.sensor_manager.camera_jpeg_frame_id.get(cam_name, -1)
                            current_jpg = self.sensor_manager.camera_jpeg.get(cam_name, None)

                        if current_fid >= expected_fid and current_jpg:
                            jpg_bytes = current_jpg
                            break

                        time.sleep(cam_poll_s)

                    # Fallback robuste: prend la dernière image dispo même si expected_fid non atteint.
                    if not jpg_bytes:
                        with self.sensor_manager.lock:
                            last_jpg = self.sensor_manager.camera_jpeg.get(cam_name, None)
                        if last_jpg:
                            jpg_bytes = last_jpg

                    if not jpg_bytes:
                        continue

                    path = os.path.join(self.output_dir, "images", f"frame_{fmt_id}_{cam_name}.jpg")
                    with open(path, "wb") as f:
                        f.write(jpg_bytes)

            # Metadata
            with open(os.path.join(self.output_dir, "metadata", f"frame_{fmt_id}.json"), 'w') as f:
                json.dump({'frame_id': frame_id, 'pos': ref_pos}, f)


    # ---------- MAIN LOOP ----------

    # =========================
    # 3) FastDatasetGenerator.generate : MÉTHODE COMPLÈTE À MODIFIER
    #    -> Patch : target frame + cache transform + reset tracking
    # =========================

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
            window_back=self.window_back,
            window_forward=self.window_forward,
            allow_extra_maintenance_ticks=self.allow_extra_maintenance_ticks,
            cam_supersample_factor=self.cam_supersample_factor,
        )
        # In one_cam_tick mode, we might want strict lidar frame matching only for the first tick?
        # Actually, if we accumulate, we might want multiple frames.
        # But if the argument means "fast capture", we usually want to align everything.
        # Let's keep strictness logic as originally intended: strict frame matching.
        self.sensor_manager.require_target_lidar_frame = bool(self.one_cam_tick_per_pose)

        max_frames = min(max_frames, len(self.positions))

        # 🚀 OPTIM: Pré-filtrage des positions dupliquées dans la trajectoire.
        # Si plusieurs positions consécutives sont au même endroit (< 0.01m),
        # on ne garde que la première. Évite de rendre la même scène N fois.
        DEDUP_DIST_M = 0.05  # seuil de déduplication (5 cm)
        unique_indices = []
        prev_loc = None
        for idx in range(max_frames):
            loc = self.positions[idx]['ego_location']
            x, y, z = float(loc['x']), float(loc['y']), float(loc['z'])
            if prev_loc is not None:
                dx = x - prev_loc[0]
                dy = y - prev_loc[1]
                dz = z - prev_loc[2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < DEDUP_DIST_M:
                    continue  # skip duplicate
            unique_indices.append(idx)
            prev_loc = (x, y, z)

        n_skipped = max_frames - len(unique_indices)
        print("\n🚀 GÉNÉRATION DATASET OCCUPANCY IMPLICITE (BANK LiDAR multi-poses)")
        print(f"  Frames max: {max_frames}")
        if n_skipped > 0:
            print(f"  ⚡ {n_skipped} positions dupliquées détectées et ignorées "
                  f"({len(unique_indices)} positions uniques)")
        print(f"  cam_supersample_factor: {self.cam_supersample_factor} "
              f"(capture {self.cam_supersample_factor * 512}×{self.cam_supersample_factor * 384})\n")

        start_time = time.time()
        position_count = 0

        # Init NPC spawner state
        self._init_npc_state()

        try:
            for loop_idx, pos_idx in enumerate(unique_indices):
                # Barrière stricte: on ne commence pas une nouvelle pose tant que
                # la sauvegarde précédente n'est pas totalement terminée.
                if loop_idx > 0:
                    with SectionTimer(self.perf, "pose_barrier_wait_save", silent=True):
                        if self.writer.queue.qsize() > 8:
                            self.writer.queue.join()

                position_count += 1
                base_pos = self.positions[pos_idx]

                ref_position = {
                    'location': base_pos['ego_location'],
                    'rotation': base_pos['ego_rotation'],
                    'timestamp_sim': base_pos['timestamp_sim']
                }

                # 👥 Spawn / téléporter les NPCs (other_actors du JSON)
                with SectionTimer(self.perf, "spawn_move_npcs"):
                    self._spawn_or_move_npcs(base_pos)
                n_npcs = len(self._npc_actors)

                print(
                    f"\n📍 Position {position_count}/{len(unique_indices)} "
                    f"(traj[{pos_idx}]) "
                    f"x={ref_position['location']['x']:.2f}, "
                    f"y={ref_position['location']['y']:.2f} "
                    f"| NPCs: {n_npcs}"
                )

                # create sensors
                if loop_idx == 0:
                    start_transform = carla.Transform(
                        carla.Location(**ref_position['location']),
                        carla.Rotation(**ref_position['rotation'])
                    )
                    if not self.sensor_manager.create_sensors_once(start_transform):
                        print("❌ Impossible de créer les capteurs")
                        return

                # robot frame T0
                with SectionTimer(self.perf, "move_cameras_to_position"):
                    self.sensor_manager.set_reference_robot(ref_position)
                    self.sensor_manager.move_cameras_to_position(ref_position)

                do_capture_cam = True
                # IMPORTANT: en mode one-tick, on garde les caméras actives en continu
                # pour éviter de perdre la frame juste après un stop/listen.
                self.sensor_manager.set_cameras_active(True)

                # build slot_to_pose
                pitch0 = float(ref_position["rotation"].get("pitch", 0.0))
                roll0  = float(ref_position["rotation"].get("roll", 0.0))

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
                    slot_to_pose[int(s)] = pose_j

                print(f"🧭 rigs actifs: {len(slot_to_pose)}")

                # 1) Positionner d'abord les rigs "purs" (sans offset global)
                with SectionTimer(self.perf, "move_all_lidar_rigs_initial"):
                    self.sensor_manager.move_all_lidar_rigs(slot_to_pose, global_offset_world=(0.0, 0.0, 0.0))

                # 2) Randomize (Jitter) : SUPPRIMÉ ici car il serait écrasé par move_all_lidar_rigs (offset global).
                # On ne conserve que le jitter final (après offset) mais avec des bornes réduites pour ne pas causer de collisions.
                # if self.randomize_clear_poses and (not self.one_cam_tick_per_pose):
                #    with SectionTimer(self.perf, "randomize_all_lidars_before_offset"):
                #        self.sensor_manager.randomize_all_lidars_params()

                # 3) Calculer l'offset global d'évitement (sur la base des positions perturbées ou non)
                if self.one_cam_tick_per_pose:
                    dx, dy, dz = (0.0, 0.0, 0.0)
                else:
                    with SectionTimer(self.perf, "find_safe_global_offset"):
                        dx, dy, dz = self.sensor_manager.find_safe_global_offset(
                            slot_to_pose, min_dist=1.0, max_tries=15, xy_radius=2.0, z_offset=0.5
                        )

                if abs(dx) + abs(dy) + abs(dz) > 0.0:
                    print(f"↔️ global transpose lidar bank: dx={dx:.2f} dy={dy:.2f} dz={dz:.2f}")

                # 4) Appliquer l'offset global (déplace tout le monde)
                # Note: move_all_lidar_rigs ré-applique la pose de base + offset.
                with SectionTimer(self.perf, "move_all_lidar_rigs_final"):
                    self.sensor_manager.move_all_lidar_rigs(slot_to_pose, global_offset_world=(dx, dy, dz))

                # On applique le Jitter MAINTENANT (sur la pose finale avec offset)
                # On réduit fortement les bornes (2cm, 1-2 deg) pour garantir que l'offset de sécurité reste valide.
                if self.randomize_clear_poses and (not self.one_cam_tick_per_pose):
                     with SectionTimer(self.perf, "randomize_all_lidars_final"):
                        self.sensor_manager.randomize_all_lidars_params(
                            loc_jitter_x=(-0.2, 0.2),
                            loc_jitter_y=(-0.2, 0.2),
                            loc_jitter_z=(-0.2, 0.2),
                            pitch_jitter=(-1.0, 1.0),
                            yaw_jitter=(-45.0, 45.0), # On garde un peu plus de yaw car moins critique pour collisions latérales
                            roll_jitter=(-1.0, 1.0)
                        )

                # Barrière "capteurs bien déplacés":
                # TOUJOURS au moins 1 tick de settle après set_transform() pour que
                # CARLA applique les nouvelles positions AVANT le tick de capture.
                # Sans ça, les capteurs (cam + lidar) peuvent encore être à l'ancienne
                # position quand le tick de capture arrive.
                # En mode one_cam_tick: 1 tick settle + petit sleep pour laisser le
                # moteur physique se stabiliser.
                # En mode normal: 1 tick + 50ms.
                settle_ticks = 1
                settle_sleep_s = 0.02 if self.one_cam_tick_per_pose else 0.05
                
                with SectionTimer(self.perf, "sensors_settle_after_teleport", silent=True):
                    self.sensor_manager.settle_sensors_after_teleport(
                        settle_ticks=settle_ticks,
                        settle_sleep_s=settle_sleep_s,
                    )

                # Vérification que les capteurs sont bien à la bonne position
                with SectionTimer(self.perf, "verify_sensors_positioned", silent=True):
                    self.sensor_manager.verify_sensors_positioned(ref_position, warn_threshold_m=1.0)

                # ✅ On vise la prochaine frame
                try:
                    snap = self.world.get_snapshot()
                    fid_next = int(snap.frame) + 1
                except Exception:
                    fid_next = None

                # Si one_cam_tick: on veut capturer Lidar sur plusieurs ticks si besoin?
                # La critique dit: "Garder target_cam_frame = fid_next uniquement sur le 1er tick... Puis désactiver cam... et continuer à tick jusqu’au quota".
                
                target_lidar_frame = fid_next if (self.one_cam_tick_per_pose and fid_next is not None) else None
                target_cam_frame = fid_next if (do_capture_cam and fid_next is not None) else None

                # ✅ start new accumulation (epoch++) + targets atomiques
                self.sensor_manager.start_new_accumulation(
                    self.capture_points_target,
                    target_lidar_frame=target_lidar_frame,
                    target_cam_frame=target_cam_frame,
                )

                self.sensor_manager.reset_lidar_frame_tracking()

                # ✅ cache des transforms pour la frame cible (CRITIQUE)
                if self.sensor_manager.target_lidar_frame is not None:
                    self.sensor_manager.snapshot_expected_lidar_matrices_for_frame(self.sensor_manager.target_lidar_frame)
                    # En mode strict, on évite d'utiliser le cache "get_transform" qui peut être stale.
                    # On se fie à snapshot_expected_lidar_matrices_for_frame qui vient de nos calculs.
                    # self.sensor_manager.cache_lidar_transforms_for_frame(...) 

                ticks_done = 0

                with SectionTimer(self.perf, "lidar_accumulation_ticks"):
                    # 🚀 MODE OPTIMISÉ: 1 tick par pose (comme V2)
                    # if self.one_cam_tick_per_pose:
                        # UN SEUL TICK - on ignore is_complete()
                    self.world.tick()
                    # self.world.tick()
                    ticks_done = 1
                    # else:
                    #     # Mode accumulation classique: boucle jusqu'à quota
                    #     while (not self.sensor_manager.lidar_accumulator.is_complete()
                    #         and ticks_done < self.max_ticks_per_pose):
                    #         self.world.tick()
                    #         ticks_done += 1

                # ✅ wait callbacks lidar (important)

                ok_lidar = True
                if self.sensor_manager.target_lidar_frame is not None:
                    with SectionTimer(self.perf, "wait_lidar_callbacks"):
                        ok_lidar = self.sensor_manager.wait_for_lidar_callbacks(timeout_s=0.5, poll_s=0.0005)
                    if not ok_lidar:
                        exp = self.sensor_manager._expected_lidar_callbacks()
                        with self.sensor_manager.lock:
                            got = len(self.sensor_manager._lidar_seen_sensor_ids)
                            tgt = self.sensor_manager._lidar_seen_frame
                        print(f"⚠️ LiDAR callbacks incomplètes pour frame={tgt} ({got}/{exp})")

                # Barrière compute: s'assure que la queue workers LiDAR est vide
                with SectionTimer(self.perf, "wait_lidar_workers"):
                    ok_workers = self.sensor_manager.wait_lidar_workers_idle(timeout_s=0.5, poll_s=0.0005)
                if not ok_workers:
                    print("⚠️ Timeout: workers LiDAR encore occupés avant capture")

                # camera wait (inchangé)
                if do_capture_cam:
                    # 🚀 OPTIM: poll à 2ms au lieu de 100ms (économise 0.1-0.8s/frame)
                    with SectionTimer(self.perf, "camera_wait_all"):
                        t0_cam_wait = time.perf_counter()
                        while True:
                            with self.sensor_manager.lock:
                                ok_all = all(self.sensor_manager.camera_raw_received.values())
                            if ok_all:
                                break
                            if (time.perf_counter() - t0_cam_wait) > 0.5:
                                break
                            time.sleep(0.002)

                print(f"⏱ ticks accumulation: {ticks_done}")

                # Barrière stricte en mode 1 tick/pose : si on n'a pas toutes les callbacks
                # du frame cible, on SKIP la frame (sinon dataset instable / ROI quasi vide).
                # On ne retick pas : on reste conforme à 1 tick par pose.
                if self.one_cam_tick_per_pose and (self.sensor_manager.target_lidar_frame is not None) and ((not ok_lidar) or (not ok_workers)):
                    print("⚠️ Skip frame: callbacks/workers LiDAR incomplets (mode 1 tick/pose)")
                    continue

                with SectionTimer(self.perf, "capture_current_frame"):
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

                    # Mode strict demandé: on force la fin de sauvegarde avant de passer
                    # à la pose suivante (évite tout chevauchement compute/IO inter-poses).
                    # with SectionTimer(self.perf, "save_frame_barrier", silent=True):
                    #     self.writer.queue.join()

                    self.global_frame_counter += 1
                    del frame_data
                    # gc.collect()
                else:
                    print("⚠️ Pas de points LiDAR pour cette frame")

                if position_count % 5 == 0:
                    elapsed = time.time() - start_time
                    fps = self.global_frame_counter / elapsed if elapsed > 0 else 0
                    remaining = len(unique_indices) - position_count
                    eta = remaining / fps if fps > 0 else 0
                    print(
                        f"{'='*50}"
                        f" PROGRESSION {position_count}/{len(unique_indices)}"
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
            with SectionTimer(self.perf, "writer_queue_drain", silent=True):
                try:
                    self.writer.queue.join()
                except Exception:
                    pass
            print("🧹 Nettoyage final...")
            self._cleanup_all_npcs()
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
    parser.add_argument('--v-upper', type=float, default=30.0, help='FOV vertical haut (deg)')
    parser.add_argument('--v-lower', type=float, default=-70.0, help='FOV vertical bas (deg)')
    parser.add_argument('--lidar-channels', type=int, default=1024, help='Canaux LiDAR')
    parser.add_argument('--lidar-pps', type=int, default=1_500_000, help='Points/seconde LiDAR')
    parser.add_argument('--lidar-range', type=float, default=100, help='Portée LiDAR (m)')

    parser.add_argument('--map', type=str, default='Town10HD_Opt', help='Carte CARLA')
    parser.add_argument('--trajectory-json', type=str,
                        default="carla_trajectories/Town10HD_Opt_fast_20251010_194958_veh25.json",
                        help='Trajectoire à rejouer')
    parser.add_argument('--weather-id', type=int, default=0,
                        help='0 clear_noon | 1 overcast_morning | ...')
    parser.add_argument('--profile', action='store_true', default=True, help='Activer le profiling')
    parser.add_argument('--capture-points', type=int, default=2_000_000,
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
    parser.add_argument('--window-back', type=int, default=3, help='Nombre de poses passées de l’ego à charger')
    parser.add_argument('--window-forward', type=int, default=3, help='Nombre de poses futures de l’ego à charger')
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

    # Cam / ticks perf
    parser.add_argument('--camera-stride', type=int, default=1,
                        help='Capture cam 1 pose sur 2 (par défaut)')

    # occupancy implicite
    parser.add_argument('--implicit-voxel-size', type=float, default=0.25,
                        help="Taille des voxels occupancy implicite (m)")
    parser.add_argument('--implicit-points-per-voxel-min', type=int, default=1,
                        help="Nb min de points par voxel")
    parser.add_argument('--implicit-points-per-voxel-max', type=int, default=1,
                        help="Nb max de points par voxel")
    parser.add_argument('--implicit-ratio-occ', type=float, default=0.6,
                        help="Ratio approx de points occupés dans le dataset implicite")
    parser.add_argument('--implicit-ratio-empty', type=float, default=0.4,
                        help="Ratio approx de points empty")
    parser.add_argument('--implicit-ratio-unknown', type=float, default=0.0,
                        help="Ratio approx de points unknown (désactivé)")
    parser.add_argument('--voxel-keep-ratio-empty', type=float, default=0.1,
                        help="Proportion de voxels empty gardés")
    parser.add_argument('--voxel-keep-ratio-unknown', type=float, default=0.1,
                        help="Proportion de voxels unknown gardés")
    parser.add_argument('--implicit-empty-points-per-hit', type=int, default=1,
                        help="Nb de points empty à tirer par hit LiDAR dans les callbacks")
    parser.add_argument('--cam-supersample-factor', type=int, default=3,
                        help='Facteur de supersampling caméra: 4=2048x1536 (meilleure qualité, tick lent), '
                             '2=1024x768 (bonne qualité, tick ~2-3x plus rapide), '
                             '1=512x384 (pas de supersampling, tick le plus rapide)')
    parser.add_argument('--fixed-delta-seconds', type=float, default=0.1,
                        help='Pas de simulation CARLA en mode synchrone (ex: 0.1 pour accélérer)')
    parser.add_argument('--carla-substepping', action='store_true',
                        help='Active le substepping CARLA (désactivé par défaut pour accélérer le tick)')
    parser.add_argument('--carla-max-substep-dt', type=float, default=0.02,
                        help='Delta max d\'un substep CARLA')
    parser.add_argument('--carla-max-substeps', type=int, default=1,
                        help='Nombre max de substeps CARLA')

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
    print(f" cam_supersample_factor={args.cam_supersample_factor}")
    print("\n💡 TIPS PERFORMANCE CARLA:")
    print("  - Lancer CARLA avec: CarlaUE4.exe -quality-level=Low  (tick ~2-3x plus rapide)")
    print("  - Utiliser --cam-supersample-factor 2  (réduit pixels rendus de 75%)")
    print("  - GPU sous-utilisé => normal (CARLA séquentiel dans le game thread UE4)")
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
        camera_stride=args.camera_stride,
        cam_supersample_factor=args.cam_supersample_factor,
        fixed_delta_seconds=args.fixed_delta_seconds,
        carla_substepping=args.carla_substepping,
        carla_max_substep_delta_time=args.carla_max_substep_dt,
        carla_max_substeps=args.carla_max_substeps,
    )
    generator.generate(max_frames=args.frames)


if __name__ == "__main__":
    main()
