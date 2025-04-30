import numpy as np
import open3d as o3d
import os
import argparse
import yaml # Utiliser PyYAML directement pour charger la config

# --- Colormap SemanticKITTI (Recopié pour autonomie du script) ---
# Vous pouvez aussi le charger depuis le fichier yaml si vous préférez
SEMANTIC_KITTI_COLOR_MAP_BGR = {
    0: [0, 0, 0], 1: [245, 150, 100], 10: [245, 230, 100], 11: [250, 80, 100],
    13: [200, 40, 250], 15: [30, 30, 250], 16: [40, 200, 250], 18: [255, 0, 0],
    20: [255, 200, 0], 30: [80, 240, 150], 31: [150, 240, 255], 32: [0, 0, 255],
    40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75], 49: [75, 0, 175],
    50: [0, 200, 255], 51: [50, 120, 255], 52: [0, 175, 0], 60: [0, 60, 135],
    70: [80, 240, 255], 71: [50, 80, 0], 72: [255, 255, 0], 80: [220, 220, 0],
    81: [100, 100, 255], 99: [255, 150, 50], 252: [245, 150, 100], 253: [150, 240, 255],
    254: [80, 240, 150], 255: [0, 0, 255], 256: [40, 200, 250], 257: [200, 40, 250],
    258: [255, 0, 0], 259: [255, 200, 0]
}
# Convertir en RGB normalisé pour Open3D
color_map_rgb_norm = {
    k: [v[2]/255.0, v[1]/255.0, v[0]/255.0]
    for k, v in SEMANTIC_KITTI_COLOR_MAP_BGR.items()
}


def visualize_voxel_grid(dataset_root, sequence, scan_id, config_rel_path):
    """
    Charge et visualise une grille de voxels sémantiques pré-calculée.
    """
    print("--- Configuration ---")
    # --- !!! ATTENTION : PARAMÈTRES À VÉRIFIER ET ADAPTER !!! ---
    # Ces valeurs DOIVENT correspondre à la grille utilisée pour générer vos fichiers.
    GRID_DIMS_VOXELS = (256, 256, 32) # Exemple: XYZ - Vérifiez l'ordre de stockage!
    VOXEL_SIZE = 0.2 # Taille physique en mètres
    GRID_ORIGIN_METERS = np.array([0.0, -25.6, -2.0]) # Exemple: Coin minimum [X,Y,Z]
    STORAGE_ORDER = 'ZYX' # Ordre des dimensions lors du reshape (souvent ZYX pour C-order)
    # --- Fin des paramètres à vérifier ---

    print(f"Dataset Root: {dataset_root}")
    print(f"Sequence: {sequence}, Scan Index: {scan_id}")
    print(f"Grid Dimensions (Logical XYZ): {GRID_DIMS_VOXELS}")
    print(f"Voxel Size: {VOXEL_SIZE}m")
    print(f"Grid Origin (XYZ): {GRID_ORIGIN_METERS}m")
    print(f"Storage Order Assumed for Reshape: {STORAGE_ORDER}")
    print("--- IMPORTANT: Vérifiez que ces paramètres sont corrects pour VOTRE dataset! ---")


    # --- Construire les chemins ---
    # Note: L'API semantic-kitti est surtout pour les points. On charge le fichier voxel direct.
    voxel_dir = os.path.join(dataset_root, "sequences", sequence, "voxels")
    label_file_path = os.path.join(voxel_dir, f"{scan_id:06d}.label")
    # Le fichier config n'est utilisé ici que pour le colormap, on pourrait aussi utiliser celui codé en dur.
    # config_abs_path = os.path.join(dataset_root, config_rel_path)

    if not os.path.exists(label_file_path):
        print(f"\nERREUR: Fichier label non trouvé: {label_file_path}")
        print("Vérifiez le chemin vers le dataset et la présence du sous-dossier 'voxels'.")
        return

    # --- Charger les labels voxels ---
    print(f"\nChargement de : {label_file_path}")
    try:
        label_data_flat = np.fromfile(label_file_path, dtype=np.uint16)
        print(f"Chargé {label_data_flat.size} labels.")
    except Exception as e:
        print(f"ERREUR lors du chargement du fichier .label: {e}")
        return

    # --- Remodeler la grille ---
    expected_size = np.prod(GRID_DIMS_VOXELS)
    if label_data_flat.size != expected_size:
        print(f"\nERREUR: Taille fichier ({label_data_flat.size}) != Taille grille attendue ({expected_size})")
        print(f"Vérifiez GRID_DIMS_VOXELS {GRID_DIMS_VOXELS}.")
        return

    print(f"Remodelage en grille {GRID_DIMS_VOXELS} (Stockage supposé: {STORAGE_ORDER})...")
    try:
        if STORAGE_ORDER.upper() == 'ZYX':
            voxel_semantic_labels = label_data_flat.reshape(GRID_DIMS_VOXELS[::-1]) # Z, Y, X
        elif STORAGE_ORDER.upper() == 'XYZ':
             voxel_semantic_labels = label_data_flat.reshape(GRID_DIMS_VOXELS) # X, Y, Z
        else:
            print(f"ERREUR: Ordre de stockage '{STORAGE_ORDER}' non supporté (utilisez XYZ ou ZYX).")
            return
        print(f"Remodelage réussi. Shape obtenu: {voxel_semantic_labels.shape}")
    except ValueError as e:
        print(f"ERREUR de remodelage: {e}. Vérifiez GRID_DIMS_VOXELS et STORAGE_ORDER.")
        return

    # --- Préparer la visualisation (PointCloud des centres) ---
    print("\nPréparation de la visualisation Open3D...")
    voxel_centers_list = []
    voxel_colors_list = []

    # Itérer selon l'ordre de stockage pour correspondre au reshape
    if STORAGE_ORDER.upper() == 'ZYX':
        occupied_indices = np.argwhere(voxel_semantic_labels > 0) # Renvoie (iz, iy, ix)
        indices_are_zyx = True
    else: # XYZ
        occupied_indices = np.argwhere(voxel_semantic_labels > 0) # Renvoie (ix, iy, iz)
        indices_are_zyx = False

    print(f"Extraction de {len(occupied_indices)} voxels occupés (label > 0)...")

    for idx in occupied_indices:
        if indices_are_zyx:
            iz, iy, ix = idx
            label = voxel_semantic_labels[iz, iy, ix]
            voxel_index_xyz = np.array([ix, iy, iz])
        else: # XYZ
            ix, iy, iz = idx
            label = voxel_semantic_labels[ix, iy, iz]
            voxel_index_xyz = np.array([ix, iy, iz])

        color = color_map_rgb_norm.get(label, [0.5, 0.5, 0.5]) # Gris si inconnu
        center_coordinate = (voxel_index_xyz + 0.5) * VOXEL_SIZE + GRID_ORIGIN_METERS
        voxel_centers_list.append(center_coordinate)
        voxel_colors_list.append(color)

    if not voxel_centers_list:
        print("Aucun voxel occupé trouvé pour visualiser.")
        return

    # --- Créer et Afficher le PointCloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_centers_list))
    pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors_list))
    print("Nuage de points des centres de voxels créé.")

    print("\nLancement de la fenêtre de visualisation Open3D...")
    print("Fermez la fenêtre pour terminer le script.")
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    grid_origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=GRID_ORIGIN_METERS)

    o3d.visualization.draw_geometries([pcd, world_axes, grid_origin_axes],
                                      window_name=f"Voxel Grid - Seq {sequence} Scan {scan_id:06d}",
                                      width=1280, height=720)
    print("Visualiseur fermé.")


# --- Point d'entrée principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise une grille de voxels sémantiques pré-calculée type SemanticKITTI SSC.")

    parser.add_argument("dataset_root", type=str,
                        help="Chemin vers le dossier racine du dataset (contenant 'sequences').")

    parser.add_argument("-s", "--sequence", type=str, default="01",
                        help="ID de la séquence à visualiser (ex: '01', basé sur votre image). Défaut: 01")

    parser.add_argument("-i", "--scan_id", type=int, default=0,
                        help="Index du scan/fichier voxel dans la séquence (ex: 0 pour 000000.label). Défaut: 0")

    # Le chemin vers config n'est plus utilisé pour charger les données ici, seulement pour info potentiel.
    # parser.add_argument("-c", "--config", type=str, default="config/semantic-kitti.yaml",
    #                     help="Chemin relatif vers le fichier config YAML (pour colormap si besoin).")

    args = parser.parse_args()

    # Vérifier si le dossier du dataset existe
    if not os.path.isdir(args.dataset_root):
        print(f"ERREUR: Le dossier du dataset '{args.dataset_root}' n'existe pas.")
    else:
        # Appeler la fonction principale de visualisation
        visualize_voxel_grid(args.dataset_root, args.sequence, args.scan_id, "") # On passe une chaine vide pour config path