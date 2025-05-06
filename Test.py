import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
import open3d as o3d

# Paramètres configurables
VOXEL_SIZE = 0.5  # Taille des voxels en mètres
RANGE = 50.0       # Portée maximale en mètres (cubique autour du véhicule)

# Chargement du dataset nuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='C:/Users/FJO25096/Downloads', verbose=True)

# Sélection d'un échantillon
sample_token = nusc.sample[0]['token']
sample = nusc.get('sample', sample_token)

# Initialisation de la voxel map
voxel_grid = np.zeros((
    int(2 * RANGE / VOXEL_SIZE), 
    int(2 * RANGE / VOXEL_SIZE),
    int(2 * RANGE / VOXEL_SIZE)
), dtype=int)

# Récupération des données des 5 radars
for radar_channel in ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                     'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
    
    # Chargement des points radar
    radar_data = nusc.get_sample_data(sample['data'][radar_channel])[0]
    points = RadarPointCloud(radar_data).points.T  # (x, y, z, vx, vy, ...)
    
    # Conversion en coordonnées voxel
    x_idx = ((points[:, 0] + RANGE) / VOXEL_SIZE).astype(int)
    y_idx = ((points[:, 1] + RANGE) / VOXEL_SIZE).astype(int)
    z_idx = ((points[:, 2] + RANGE) / VOXEL_SIZE).astype(int)
    
    # Filtrage des indices valides
    valid = (x_idx >= 0) & (x_idx < voxel_grid.shape[0]) & \
            (y_idx >= 0) & (y_idx < voxel_grid.shape[1]) & \
            (z_idx >= 0) & (z_idx < voxel_grid.shape[2])
    
    # Mise à jour de la voxel map
    np.add.at(voxel_grid, (x_idx[valid], y_idx[valid], z_idx[valid]), 1)

# Visualisation avec Open3D
mesh = o3d.geometry.VoxelGrid.create_from_voxel_array(
    voxels=o3d.core.Tensor(voxel_grid),
    voxel_size=VOXEL_SIZE,
    origin=(-RANGE, -RANGE, -RANGE)
)

# Création de la fenêtre
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)

# Paramètres de visualisation
view_ctl = vis.get_view_control()
view_ctl.set_front([0, 0, -1])
view_ctl.set_up([0, -1, 0])
view_ctl.set_zoom(0.1)

vis.run()
vis.destroy_window()