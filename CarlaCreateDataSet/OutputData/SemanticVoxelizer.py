# semantic_voxelizer.py
import numpy as np
import warnings
# Importer les définitions des tags depuis le simulateur
from CreateLidar import SEMANTIC_TAGS

class SemanticVoxelizer:
    """ Crée une grille de voxels sémantiques à partir d'un nuage de points tagués. """
    def __init__(self,
                 voxel_size=(0.5, 0.5, 0.5),       # Taille (dx, dy, dz) en mètres
                 point_cloud_range=(-50, -50, -5, 50, 50, 3), # Limites [xmin, ymin, zmin, xmax, ymax, zmax]
                 allowed_tags=None):              # Liste des tags sémantiques à inclure
        """
        Args:
            voxel_size (tuple): Taille (dx, dy, dz) de chaque voxel en mètres.
            point_cloud_range (list/tuple): Limites [xmin, ymin, zmin, xmax, ymax, zmax].
            allowed_tags (list or None): Liste des IDs de tags sémantiques à conserver.
                                         Si None, tous les tags valides sont conservés.
        """
        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        self.allowed_tags = set(allowed_tags) if allowed_tags is not None else None

        # Calculer l'origine et la taille de la grille
        self.grid_origin = self.point_cloud_range[:3]
        grid_dims_float = (self.point_cloud_range[3:] - self.grid_origin) / self.voxel_size
        # Arrondir et s'assurer que la taille est au moins 1 dans chaque dimension
        self.grid_size = np.maximum(1, np.round(grid_dims_float)).astype(np.int32)

        print(f"Voxelizer initialized:")
        print(f"  Voxel Size: {self.voxel_size}")
        print(f"  PC Range: {self.point_cloud_range}")
        print(f"  Grid Origin: {self.grid_origin}")
        print(f"  Grid Size (voxels XYZ): {self.grid_size}") # Ordre: X, Y, Z
        if self.allowed_tags is not None:
             allowed_names = [SEMANTIC_TAGS.get(t, {}).get('name', 'Inconnu') for t in sorted(list(self.allowed_tags))]
             print(f"  Allowed Tags IDs: {sorted(list(self.allowed_tags))}")
             print(f"  Allowed Tags Names: {allowed_names}")
        else:
             print("  Allowed Tags: All (except potentially invalid tags)")

        self.empty_voxel_tag = -1 # Tag pour voxel vide

    def voxelize(self, point_cloud):
        """ Convertit un nuage de points en grille de voxels sémantiques. """
        if point_cloud.shape[1] < 5:
            raise ValueError("Point cloud must have at least 5 columns (x,y,z,i,tag)")

        points = point_cloud[:, :3]
        tags = point_cloud[:, 4].astype(int)

        voxel_grid = np.full(self.grid_size, self.empty_voxel_tag, dtype=np.int32)

        # 1. Filtrer par tag sémantique (si spécifié)
        if self.allowed_tags is not None:
            mask_allowed = np.isin(tags, list(self.allowed_tags))
            points = points[mask_allowed]
            tags = tags[mask_allowed]
            if points.shape[0] == 0:
                 # warnings.warn("No points remaining after tag filtering.") # Peut être verbeux
                 return voxel_grid

        # 2. Filtrer les points hors limites
        min_bound = self.grid_origin
        max_bound = self.point_cloud_range[3:]
        mask_in_range = np.all((points >= min_bound) & (points < max_bound), axis=1)
        points = points[mask_in_range]
        tags = tags[mask_in_range]
        if points.shape[0] == 0:
             # warnings.warn("No points remaining after range filtering.") # Peut être verbeux
             return voxel_grid

        # 3. Calculer les indices des voxels
        voxel_indices = ((points - self.grid_origin) / self.voxel_size).astype(np.int32)
        # Assurer que les indices sont bien dans les limites de la grille (peut arriver à cause des flottants)
        voxel_indices = np.minimum(voxel_indices, self.grid_size - 1)
        voxel_indices = np.maximum(voxel_indices, 0)


        # 4. Assigner les tags aux voxels (le dernier point écrase les précédents)
        # Utiliser les indices comme clés : (ix, iy, iz)
        voxel_grid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = tags

        return voxel_grid

# Petit test si le fichier est exécuté seul
if __name__ == '__main__':
     print("\nTesting SemanticVoxelizer...")
     # Créer un nuage de points factice
     # Colonnes: x, y, z, intensité, tag
     points = np.array([
         [1.1, 2.1, 0.1, 100, 1],   # road dans voxel (0, 0, 0) si size=2, range=0-10
         [1.5, 2.5, 0.5, 100, 1],   # road aussi dans voxel (0, 1, 0)
         [3.1, 4.1, 1.1, 100, 3],   # building dans voxel (1, 2, 0)
         [3.5, 4.5, 1.5, 100, 14],  # car dans voxel (1, 2, 0) -> va écraser building
         [9.9, 9.9, 4.9, 100, 9],   # vegetation dans voxel (4, 4, 2)
         [10.1, 2.0, 1.0, 100, 4],  # hors limite xmax
         [1.0, -0.1, 1.0, 100, 5],  # hors limite ymin
         [1.0, 1.0, -0.1, 100, 6],  # hors limite zmin
         [1.0, 8.0, 6.0, 100, 12], # hors limite zmax
         [2.0, 3.0, 1.0, 100, 7],   # traffic light (sera filtré si non allowed)
     ])
     voxelizer_test = SemanticVoxelizer(
         voxel_size=(2.0, 2.0, 2.0),
         point_cloud_range=(0, 0, 0, 10, 10, 5), # Grid 5x5x3 (index 0-4, 0-4, 0-2)
         allowed_tags=[1, 2, 3, 9, 14, 25] # Exclure tag 7 (traffic light)
     )
     grid = voxelizer_test.voxelize(points)
     print("\nTest Voxel Grid Shape:", grid.shape)
     print("Test Voxel Grid non-empty values:")
     non_empty_indices = np.argwhere(grid != voxelizer_test.empty_voxel_tag)
     for idx in non_empty_indices:
         print(f"  Index {tuple(idx)} -> Tag {grid[tuple(idx)]} ({SEMANTIC_TAGS.get(grid[tuple(idx)], {}).get('name', 'Inconnu')})")

     # Vérifications attendues:
     # Voxel (0,1,0) devrait avoir tag 1 (road)
     # Voxel (1,2,0) devrait avoir tag 14 (car), écrasant 3 (building)
     # Voxel (4,4,2) devrait avoir tag 9 (vegetation)
     # Voxel avec tag 7 ne devrait pas apparaître
     print("\nTest SemanticVoxelizer terminé.")