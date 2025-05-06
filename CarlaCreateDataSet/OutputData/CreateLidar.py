# lidar_simulator.py
import numpy as np
import time

# =============================================================================
# === SECTION 1 : Dictionnaire des Tags Sémantiques ==========================
# =============================================================================
# Ce dictionnaire est utilisé par le simulateur pour assigner des tags
# et par le voxelizer/visualiseur pour filtrer/colorer.

SEMANTIC_TAGS = {
    0: {"name": "unlabeled", "color": (0, 0, 0)},
    1: {"name": "road", "color": (128, 64, 128)},
    2: {"name": "sidewalk", "color": (244, 35, 232)},
    3: {"name": "building", "color": (70, 70, 70)},
    4: {"name": "wall", "color": (102, 102, 156)},
    5: {"name": "fence", "color": (190, 153, 153)},
    6: {"name": "pole", "color": (153, 153, 153)},
    7: {"name": "traffic light", "color": (250, 170, 30)},
    8: {"name": "traffic sign", "color": (220, 220, 0)},
    9: {"name": "vegetation", "color": (107, 142, 35)},
    10: {"name": "terrain", "color": (152, 251, 152)},
    11: {"name": "sky", "color": (70, 130, 180)},
    12: {"name": "pedestrian", "color": (220, 20, 60)},
    13: {"name": "rider", "color": (255, 0, 0)},
    14: {"name": "car", "color": (0, 0, 142)},
    15: {"name": "truck", "color": (0, 0, 70)},
    16: {"name": "bus", "color": (0, 60, 100)},
    17: {"name": "train", "color": (0, 80, 100)},
    18: {"name": "motorcycle", "color": (0, 0, 230)},
    19: {"name": "bicycle", "color": (119, 11, 32)},
    20: {"name": "static", "color": (110, 190, 160)},
    21: {"name": "dynamic", "color": (170, 120, 50)},
    22: {"name": "other", "color": (55, 90, 80)},
    23: {"name": "water", "color": (45, 60, 150)},
    24: {"name": "road line", "color": (157, 234, 50)},
    25: {"name": "ground", "color": (81, 0, 81)},
    26: {"name": "bridge", "color": (150, 100, 100)},
    27: {"name": "rail track", "color": (230, 150, 140)},
    28: {"name": "guard rail", "color": (180, 165, 180)}
}

# =============================================================================
# === SECTION 2 : Simulateur Lidar ==========================================
# =============================================================================

class LidarSimulator:
    """ Simule un capteur Lidar générant des nuages de points sémantiques. """
    def __init__(self, x=0.0, y=0.0, z=1.7, # Position du Lidar (en mètres)
                 points_per_scan=100000, # Nombre de points par scan (~ densité)
                 fps=10, # Scans par seconde
                 max_range=100.0): # Portée maximale
        self.position = np.array([x, y, z])
        self.points_per_scan = points_per_scan
        self.scan_time = 1.0 / fps if fps > 0 else 0
        self.max_range = max_range
        self.last_scan_time = -1 # Pour simuler le timing

    def generate_scan(self):
        """
        Génère UN scan Lidar simulé.
        !!! CETTE FONCTION EST UN PLACEHOLDER BASIQUE !!!
        Remplacez cette logique par le chargement de vraies données ou un simulateur réaliste.

        Retourne:
            np.ndarray: Tableau de shape (N, 5) où N <= points_per_scan.
                        Colonnes: [x, y, z, intensité, tag_sémantique]
                        Coordonnées relatives au Lidar.
        """
        current_time = time.time()
        if self.scan_time > 0 and self.last_scan_time > 0:
            time_since_last = current_time - self.last_scan_time
            if time_since_last < self.scan_time:
                time.sleep(self.scan_time - time_since_last)
        self.last_scan_time = time.time()

        # --- Logique de Simulation Simplifiée ---
        num_points = self.points_per_scan
        radius = np.random.rand(num_points) * self.max_range
        theta = np.random.rand(num_points) * 2 * np.pi
        phi = np.arccos(2 * np.random.rand(num_points) - 1)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        intensity = np.random.rand(num_points) * 255

        semantic_tag = np.full(num_points, 0, dtype=int) # Default: unlabeled (tag 0)
        # Simuler sol
        ground_mask = (z > -self.position[2] - 0.3) & (z < -self.position[2] + 0.3) & (radius < self.max_range * 0.8)
        semantic_tag[ground_mask & (np.abs(y) < 2.5)] = 1 # road proche
        semantic_tag[ground_mask & (np.abs(y) >= 2.5)] = 2 # sidewalk plus loin
        # Simuler "murs" = building
        building_mask = (np.abs(x) > self.max_range * 0.7) | (np.abs(y) > self.max_range * 0.7) & (z > -1.0)
        semantic_tag[building_mask] = 3
        # Simuler "voitures"
        car_mask = (radius < 20) & (x > 5) & (np.abs(y) < 5) & (z > -1.5) & (z < 0.5)
        semantic_tag[car_mask] = 14
        # Simuler vegetation haute
        veg_mask = (radius > 10) & (z > 0.5) & ~building_mask & ~car_mask
        semantic_tag[veg_mask] = 9

        scan_data = np.stack([x, y, z, intensity, semantic_tag], axis=-1)
        # --- Fin Simulation ---

        return scan_data

# Petit test si le fichier est exécuté seul
if __name__ == '__main__':
    print("Test LidarSimulator...")
    lidar = LidarSimulator(points_per_scan=5000, fps=1)
    scan1 = lidar.generate_scan()
    print(f"Scan 1 shape: {scan1.shape}, Tags unique: {np.unique(scan1[:, 4])}")
    time.sleep(0.5) # Moins que le scan_time (1s)
    scan2 = lidar.generate_scan() # Devrait attendre ~0.5s
    print(f"Scan 2 shape: {scan2.shape}, Tags unique: {np.unique(scan2[:, 4])}")
    print("Test LidarSimulator terminé.")