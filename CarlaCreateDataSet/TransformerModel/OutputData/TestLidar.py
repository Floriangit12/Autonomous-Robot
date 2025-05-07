# test_lidar_voxel.py
import time
import numpy as np
import warnings

# Importer les classes définies dans les autres fichiers
try:
    from CreateLidar import LidarSimulator, SEMANTIC_TAGS
    from SemanticVoxelizer import SemanticVoxelizer
except ImportError as e:
    print(f"ERREUR: Assurez-vous que lidar_simulator.py et semantic_voxelizer.py sont dans le même dossier ou PYTHONPATH.")
    print(e)
    exit()

# Optionnel pour la visualisation temps réel
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    # Créer un mapping couleur pour la visualisation Open3D (valeurs entre 0 et 1)
    COLOR_MAP_O3D = {tag: np.array(info["color"]) / 255.0 for tag, info in SEMANTIC_TAGS.items()}
    DEFAULT_COLOR = np.array([0.5, 0.5, 0.5]) # Gris pour tags inconnus/filtrés
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Avertissement: open3d non trouvé. La visualisation sera désactivée. `pip install open3d`")


# =============================================================================
# === SECTION : Test de Visualisation "Temps Réel" ==========================
# =============================================================================

if __name__ == "__main__":

    print("Lancement du test de simulation Lidar + Voxelisation...")

    # --- Configuration ---
    LIDAR_FPS = 5
    LIDAR_POINTS = 50000
    LIDAR_RANGE = 50.0

    VOXEL_SIZE = (0.5, 0.5, 0.5)
    GRID_RANGE = [-40, -40, -3, 40, 40, 5]

    ALLOWED_TAGS_FOR_VOXELIZATION = [
        1, 2, 3, 4, 5, 6, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 28
        # Commentez/décommentez pour filtrer
    ]
    print(f"Tags autorisés pour voxelisation: {ALLOWED_TAGS_FOR_VOXELIZATION}")
    # --- Fin Configuration ---

    # --- Initialisation ---
    lidar_sim = LidarSimulator(fps=LIDAR_FPS, points_per_scan=LIDAR_POINTS, max_range=LIDAR_RANGE)
    voxelizer = SemanticVoxelizer(voxel_size=VOXEL_SIZE, point_cloud_range=GRID_RANGE, allowed_tags=ALLOWED_TAGS_FOR_VOXELIZATION)

    # --- Préparer la visualisation Open3D ---
    vis = None
    o3d_voxel_grid = None
    escape_is_pressed = False # Définir le flag AVANT la fonction callback

    if OPEN3D_AVAILABLE:
        print("\nInitialisation de la fenêtre Open3D...")
        print("Appuyez sur 'Q' dans la fenêtre pour quitter.")
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name='Voxel Grid Sémantique Temps Réel', width=1024, height=768)
        o3d_voxel_grid = o3d.geometry.VoxelGrid()
        view_control = vis.get_view_control()
        view_control.set_front([0.5, -0.8, -0.3]); view_control.set_lookat([5, 0, 0]);
        view_control.set_up([0, 0, 1]); view_control.set_zoom(0.1)
        vis.add_geometry(o3d_voxel_grid, reset_bounding_box=False)

        # Callback pour quitter avec la touche Q
        def request_exit(vis):
            # nonlocal escape_is_pressed # Incorrect
            global escape_is_pressed  # *** CORRECTION ICI ***
            escape_is_pressed = True
            print(" Demande de sortie reçue (touche Q)...")
        vis.register_key_callback(ord("Q"), request_exit) # Touche 'Q' majuscule

    else:
        print("\nVisualisation Open3D désactivée.")

    # --- Boucle Principale ---
    print("\nDébut de la simulation (Ctrl+C pour arrêter)...")
    frame_count = 0
    try:
        while True:
            if vis and escape_is_pressed: break # Vérifier le flag

            # print(f"\n--- Frame {frame_count} ---") # Optionnel : décommenter pour voir chaque frame

            # 1. Générer un scan Lidar simulé
            t_start_scan = time.time()
            point_cloud_data = lidar_sim.generate_scan()
            t_end_scan = time.time()
            # print(f"Scan généré ({point_cloud_data.shape[0]} pts) en {(t_end_scan-t_start_scan)*1000:.1f} ms")

            # 2. Voxeliser le nuage de points
            t_start_voxel = time.time()
            semantic_voxel_grid_np = voxelizer.voxelize(point_cloud_data)
            t_end_voxel = time.time()
            # print(f"Voxelisation (shape={semantic_voxel_grid_np.shape}) en {(t_end_voxel-t_start_voxel)*1000:.1f} ms")

            # 3. Mettre à jour la visualisation Open3D
            if vis and o3d_voxel_grid is not None:
                t_start_vis = time.time()
                # Conversion Numpy Grid -> Open3D VoxelGrid
                non_empty_indices = np.argwhere(semantic_voxel_grid_np != voxelizer.empty_voxel_tag)
                if non_empty_indices.shape[0] > 0:
                    centers = (non_empty_indices.astype(np.float32) + 0.5) * voxelizer.voxel_size + voxelizer.grid_origin
                    tags_for_colors = semantic_voxel_grid_np[non_empty_indices[:, 0], non_empty_indices[:, 1], non_empty_indices[:, 2]]
                    colors = np.array([COLOR_MAP_O3D.get(tag, DEFAULT_COLOR) for tag in tags_for_colors])

                    temp_pcd = o3d.geometry.PointCloud()
                    temp_pcd.points = o3d.utility.Vector3dVector(centers)
                    temp_pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d_voxel_grid_new = o3d.geometry.VoxelGrid.create_from_point_cloud(temp_pcd, voxel_size=voxelizer.voxel_size[0])

                    o3d_voxel_grid.voxels = o3d_voxel_grid_new.voxels
                    o3d_voxel_grid.colors = o3d_voxel_grid_new.colors
                    o3d_voxel_grid.origin = o3d_voxel_grid_new.origin
                    o3d_voxel_grid.voxel_size = o3d_voxel_grid_new.voxel_size

                    vis.update_geometry(o3d_voxel_grid)
                else:
                    if len(o3d_voxel_grid.get_voxels()) > 0:
                         o3d_voxel_grid.clear()
                         vis.update_geometry(o3d_voxel_grid)

                # Mettre à jour la fenêtre
                if not vis.poll_events(): break
                vis.update_renderer()
                # t_end_vis = time.time()
                # print(f"Visualisation màj ({len(o3d_voxel_grid.get_voxels())} voxels) en {(t_end_vis-t_start_vis)*1000:.1f} ms") # Verbeux

            else:
                # Pause si pas de visualisation
                elapsed_total = time.time() - t_start_scan
                sleep_time = max(0, lidar_sim.scan_time - elapsed_total)
                time.sleep(sleep_time)

            frame_count += 1

    except KeyboardInterrupt:
        print("\nArrêt de la simulation (Ctrl+C).")
    finally:
        if vis:
            vis.destroy_window()
            print("Fenêtre Open3D fermée.")
