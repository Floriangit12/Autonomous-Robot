#!/usr/bin/env python

import carla
import numpy as np
import math
import time
import os
import queue
import argparse
from datetime import datetime
import pytz
import logging

# Importer les fonctions de gestion de l'environnement
import carla_env_manager

# Configuration du logging (peut être partagée ou spécifique)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# -- Constantes et Configuration (Dataset) -----------------------------------
# ==============================================================================

# --- Liste des points de vue (x, y) à capturer ---
# Coordonnées spécifiques à la carte CARLA utilisée (ex: Town03)
VIEWPOINTS_TOWN03 = [
    (130.0, 50.0),   # Un carrefour
    (0.0, 0.0),      # Proche de l'origine
    (200.0, 150.0),  # Une autre zone
    (-50.0, 100.0),  # Zone différente
    (80.0, -100.0),  # Encore ailleurs
]
# VIEWPOINTS_TOWN05 = [...] # Définir pour d'autres cartes si besoin

# Sélectionner la liste de points de vue
TARGET_VIEWPOINTS = VIEWPOINTS_TOWN03

# --- Dossier principal de sauvegarde ---
BASE_SAVE_DIR = "_multi_bev_dataset"

# --- Paramètres CARLA et Capteurs (peuvent être surchargés via argparse) ---
HOST = 'localhost'
PORT = 2000
TIMEOUT = 30.0 # Augmenté pour setup + capture
DELTA_SECONDS = 0.1

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
CAMERA_FOV = 90.0
DEFAULT_SENSOR_Z = 5.0
DEFAULT_SENSOR_PITCH = -30.0
DEFAULT_SENSOR_YAW = 0.0

# --- Paramètres Voxel/BEV Grid ---
VOXEL_SIZE = 0.1
GRID_RANGE_X = 32.0
GRID_RANGE_Y = 32.0
GRID_RANGE_Z = 6.0
EMPTY_VOXEL_VALUE = -1
BEV_EMPTY_VALUE = -1

# --- Paramètres Environnement PNJ (passés à carla_env_manager) ---
NUM_VEHICLES = 40
NUM_PEDESTRIANS = 80
FOG_DENSITY = 0.05 # Un peu de brouillard léger

# --- Métadonnées ---
TIMEZONE = 'Europe/Paris'
LOCATION_METADATA = 'Montbonnot-Saint-Martin, Auvergne-Rhône-Alpes, France'


# ==============================================================================
# -- Fonctions BEV (Adaptées pour être appelées dans une boucle) --------------
# ==============================================================================

# Reprise des fonctions utilitaires (peuvent aussi être dans un fichier séparé)
def get_camera_intrinsics(width, height, fov):
    f = width / (2 * math.tan(math.radians(fov / 2)))
    cx = width / 2
    cy = height / 2
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def create_voxel_grid(grid_dims):
    return np.full(grid_dims, fill_value=EMPTY_VOXEL_VALUE, dtype=np.int16)

def process_depth_and_segmentation_world(depth_img, seg_img, K, sensor_world_transform_matrix):
    depth_data = np.frombuffer(depth_img.raw_data, dtype=np.dtype("float32"))
    depth_meters = np.reshape(depth_data, (img_height, img_width)) # Utiliser les variables globales/passées
    seg_data = np.frombuffer(seg_img.raw_data, dtype=np.dtype("uint8"))
    seg_labels = np.reshape(seg_data, (img_height, img_width, 4))[:, :, 2]

    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    u_coords, v_coords = np.meshgrid(np.arange(img_width), np.arange(img_height))

    x_cam = (u_coords - cx) * depth_meters / f
    y_cam = (v_coords - cy) * depth_meters / f
    z_cam = depth_meters
    valid_mask = (depth_meters > 0.1) & (depth_meters < 150.0)

    x_cam = x_cam[valid_mask]
    y_cam = y_cam[valid_mask]
    z_cam = z_cam[valid_mask]
    labels = seg_labels[valid_mask]
    points_camera_frame = np.stack((x_cam, y_cam, z_cam), axis=-1)

    points_h = np.hstack((points_camera_frame, np.ones((points_camera_frame.shape[0], 1))))
    points_world_frame_h = np.dot(sensor_world_transform_matrix, points_h.T).T
    points_world_frame = points_world_frame_h[:, :3]

    return points_world_frame, labels

def world_to_voxel_idx(point_world, grid_origin_world, voxel_size, grid_dims):
    relative_pos = point_world - grid_origin_world
    vx = int(relative_pos[0] / voxel_size)
    vy = int(relative_pos[1] / voxel_size)
    vz = int(relative_pos[2] / voxel_size)
    if 0 <= vx < grid_dims[0] and 0 <= vy < grid_dims[1] and 0 <= vz < grid_dims[2]:
        return vx, vy, vz
    else:
        return None

def populate_voxel_grid_world(voxel_grid, points_world, labels, grid_origin_world, voxel_size):
    grid_dims = voxel_grid.shape
    for i in range(len(points_world)):
        voxel_indices = world_to_voxel_idx(points_world[i], grid_origin_world, voxel_size, grid_dims)
        if voxel_indices is not None:
            vx, vy, vz = voxel_indices
            voxel_grid[vx, vy, vz] = labels[i]
    return voxel_grid

def project_voxel_to_bev(voxel_grid):
    grid_dims = voxel_grid.shape
    bev_grid = np.full((grid_dims[0], grid_dims[1]), fill_value=BEV_EMPTY_VALUE, dtype=voxel_grid.dtype)
    for vx in range(grid_dims[0]):
        for vy in range(grid_dims[1]):
            z_column = voxel_grid[vx, vy, :]
            occupied_voxels = z_column[z_column != EMPTY_VOXEL_VALUE]
            if occupied_voxels.size > 0:
                highest_occupied_idx = np.max(np.where(z_column != EMPTY_VOXEL_VALUE))
                bev_grid[vx, vy] = z_column[highest_occupied_idx]
    return bev_grid


def generate_single_bev(
    world, # Passer le monde en argument
    viewpoint_xy,
    save_dir,
    filename_prefix,
    # Reprendre les paramètres nécessaires
    sensor_z=DEFAULT_SENSOR_Z,
    sensor_pitch=DEFAULT_SENSOR_PITCH,
    sensor_yaw=DEFAULT_SENSOR_YAW,
    img_width=IMAGE_WIDTH,
    img_height=IMAGE_HEIGHT,
    cam_fov=CAMERA_FOV,
    voxel_sz=VOXEL_SIZE,
    grid_x_range=GRID_RANGE_X,
    grid_y_range=GRID_RANGE_Y,
    grid_z_range=GRID_RANGE_Z,
):
    """
    Génère et sauvegarde UN SEUL BEV pour un point de vue donné.
    Assume que le monde est déjà en mode synchrone.
    Gère la création et destruction de SES capteurs.
    """
    sensor_list = []
    sensor_queue = queue.Queue() # Queue locale pour cette fonction
    blueprint_library = world.get_blueprint_library() # Obtenir depuis le monde passé

    # --- Informations Date/Heure/Lieu ---
    utc_now = datetime.now(pytz.utc)
    local_tz = pytz.timezone(TIMEZONE)
    local_time = utc_now.astimezone(local_tz)
    timestamp_str = local_time.strftime("%Y%m%d_%H%M%S")
    logging.info(f"Génération BEV pour {viewpoint_xy} at {timestamp_str}")

    try:
        # --- Placement et Création des Capteurs ---
        sensor_location = carla.Location(x=viewpoint_xy[0], y=viewpoint_xy[1], z=sensor_z)
        sensor_rotation = carla.Rotation(pitch=sensor_pitch, yaw=sensor_yaw, roll=0.0)
        sensor_world_transform = carla.Transform(sensor_location, sensor_rotation)
        sensor_tf_matrix = np.array(sensor_world_transform.get_matrix())

        # Créer les capteurs requis
        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(img_width))
        depth_bp.set_attribute('image_size_y', str(img_height))
        depth_bp.set_attribute('fov', str(cam_fov))
        depth_sensor = world.spawn_actor(depth_bp, sensor_world_transform)
        depth_sensor.listen(lambda image: sensor_queue.put((image.frame, 'depth', image)))
        sensor_list.append(depth_sensor)

        seg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(img_width))
        seg_bp.set_attribute('image_size_y', str(img_height))
        seg_bp.set_attribute('fov', str(cam_fov))
        seg_sensor = world.spawn_actor(seg_bp, sensor_world_transform)
        seg_sensor.listen(lambda image: sensor_queue.put((image.frame, 'seg', image)))
        sensor_list.append(seg_sensor)
        logging.debug(f"Capteurs {depth_sensor.id}, {seg_sensor.id} créés pour {viewpoint_xy}")

        # --- Calculs Préliminaires Grid ---
        K = get_camera_intrinsics(img_width, img_height, cam_fov)
        grid_origin_world = np.array([
            viewpoint_xy[0] - grid_x_range / 2.0,
            viewpoint_xy[1] - grid_y_range / 2.0,
            sensor_z - grid_z_range / 2.0 # Ajuster Z si nécessaire
        ])
        voxel_grid_dims = (
            int(grid_x_range / voxel_sz),
            int(grid_y_range / voxel_sz),
            int(grid_z_range / voxel_sz)
        )

        # --- Capture ---
        # Le tick est géré par la boucle externe, on attend juste les données
        # Il faut un tick APRES le spawn pour que listen fonctionne
        current_frame = world.tick() # Tick pour enregistrer les listeners et obtenir le frame actuel
        logging.debug(f"Attente des données pour frame {current_frame}")

        sensor_data = {'depth': None, 'seg': None}
        try:
            for _ in range(len(sensor_list)):
                 s_frame, s_type, s_data = sensor_queue.get(timeout=5.0)
                 if s_frame == current_frame:
                     sensor_data[s_type] = s_data
                     logging.debug(f"Données reçues pour {s_type} frame {s_frame}")
                 # else: logging.warning(...) # Gérer frames en retard si ça arrive

        except queue.Empty:
            logging.error(f"Timeout en attendant les données capteur pour frame {current_frame} à {viewpoint_xy}")
            return None, None # Échec pour ce point de vue

        if not sensor_data['depth'] or not sensor_data['seg']:
            logging.error(f"Données capteur manquantes pour frame {current_frame} à {viewpoint_xy}")
            return None, None # Échec

        # --- Traitement et Sauvegarde ---
        logging.debug("Traitement des données et voxelisation...")
        voxel_grid_3d = create_voxel_grid(voxel_grid_dims)
        points_world, labels = process_depth_and_segmentation_world(
            sensor_data['depth'], sensor_data['seg'], K, sensor_tf_matrix
        )
        voxel_grid_3d = populate_voxel_grid_world(
            voxel_grid_3d, points_world, labels, grid_origin_world, voxel_sz
        )
        logging.debug("Projection BEV...")
        bev_grid_2d = project_voxel_to_bev(voxel_grid_3d)

        logging.debug(f"Sauvegarde dans {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
        filename_base = f"{filename_prefix}_x{viewpoint_xy[0]:.1f}_y{viewpoint_xy[1]:.1f}_{timestamp_str}"
        bev_filepath = os.path.join(save_dir, f"{filename_base}_bev.npy")
        meta_filepath = os.path.join(save_dir, f"{filename_base}_meta.npy")

        np.save(bev_filepath, bev_grid_2d)
        metadata = {
            'timestamp_utc': utc_now.isoformat(), 'timestamp_local': local_time.isoformat(),
            'timezone': TIMEZONE, 'location_context': LOCATION_METADATA,
            'viewpoint_xy': viewpoint_xy,
            'sensor_transform': {'location': (sensor_location.x, sensor_location.y, sensor_location.z),
                                 'rotation': (sensor_rotation.pitch, sensor_rotation.yaw, sensor_rotation.roll)},
            'voxel_size': voxel_sz, 'voxel_grid_origin_world': grid_origin_world.tolist(),
            'voxel_grid_dimensions_xyz': voxel_grid_dims, 'bev_grid_shape_xy': bev_grid_2d.shape,
            'carla_map': world.get_map().name, 'simulation_frame': current_frame
        }
        np.save(meta_filepath, metadata)
        logging.info(f"BEV sauvegardé: {bev_filepath}")
        return bev_filepath, meta_filepath

    except Exception as e:
        logging.error(f"Erreur dans generate_single_bev pour {viewpoint_xy}: {e}", exc_info=True)
        return None, None
    finally:
        # --- Nettoyage des capteurs de CETTE fonction ---
        logging.debug(f"Nettoyage des capteurs pour {viewpoint_xy}")
        # Utiliser client.destroy_actors() est plus sûr en cas de crash partiel
        if sensor_list:
             client = world.get_client() # Obtenir le client depuis le monde
             ids_to_destroy = [s.id for s in sensor_list if s and s.is_alive]
             if ids_to_destroy:
                  client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in ids_to_destroy], True)
                  logging.debug(f"{len(ids_to_destroy)} capteurs détruits pour {viewpoint_xy}")
        # Ne pas toucher aux settings du monde ici

# ==============================================================================
# -- Script Principal (Dataset Multi-Vues) ------------------------------------
# ==============================================================================

def main(args):
    client = None
    world = None
    original_settings = None
    vehicle_ids, walker_ids, controller_ids = [], [], [] # Pour cleanup final

    try:
        # --- Connexion Initiale ---
        logging.info(f"Connexion au serveur CARLA sur {args.host}:{args.port}")
        client = carla.Client(args.host, args.port)
        client.set_timeout(TIMEOUT)
        world = client.get_world()
        original_settings = world.get_settings() # Sauvegarder les settings initiaux
        logging.info(f"Connecté à la carte: {world.get_map().name}")

        # --- Configuration Globale du Monde (Mode Synchrone) ---
        logging.info("Activation du mode synchrone...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DELTA_SECONDS
        world.apply_settings(settings)

        # --- Configuration de l'Environnement PNJ via le module séparé ---
        logging.info("Configuration de l'environnement PNJ...")
        vehicle_ids, walker_ids, controller_ids = carla_env_manager.setup_carla_environment(
            client,
            world,
            num_vehicles=args.num_vehicles,
            num_pedestrians=args.num_peds,
            fog_density=args.fog
            # Ajouter d'autres paramètres météo ici si besoin
        )
        if not vehicle_ids and not walker_ids:
             logging.warning("Aucun acteur PNJ n'a pu être spawné. Le dataset sera généré dans un monde vide.")

        # --- Boucle sur les Points de Vue ---
        logging.info(f"Début de la génération BEV pour {len(TARGET_VIEWPOINTS)} points de vue...")
        successful_captures = 0
        for i, viewpoint in enumerate(TARGET_VIEWPOINTS):
            logging.info(f"--- Traitement Point de Vue {i+1}/{len(TARGET_VIEWPOINTS)}: {viewpoint} ---")

            # Définir un sous-dossier pour ce point de vue
            view_save_dir = os.path.join(args.save_dir, f"view_{i:04d}")
            filename_prefix = f"vp_{i:04d}" # Préfixe spécifique à la vue

            # Appeler la fonction de génération pour ce point de vue
            # Passe le 'world' configuré
            bev_file, meta_file = generate_single_bev(
                world=world, # Passe l'objet world
                viewpoint_xy=viewpoint,
                save_dir=view_save_dir,
                filename_prefix=filename_prefix,
                # Passer d'autres args si nécessaire (img_width, etc.)
                img_width=args.width,
                img_height=args.height,
                cam_fov=args.fov,
                sensor_z=args.sensor_z,
                sensor_pitch=args.sensor_pitch,
                voxel_sz=args.voxel_size,
                grid_x_range=args.grid_x,
                grid_y_range=args.grid_y,
                grid_z_range=args.grid_z
            )

            if bev_file and meta_file:
                successful_captures += 1
            else:
                logging.warning(f"Échec de la génération BEV pour le point de vue {viewpoint}. Passage au suivant.")

            # Optionnel: petit délai entre les captures
            # time.sleep(0.5)
            # Il n'est pas forcément nécessaire de faire un tick ici si l'environnement est statique
            # world.tick() # Faire un tick si on veut que les PNJ bougent entre les captures

        logging.info(f"Génération terminée. {successful_captures}/{len(TARGET_VIEWPOINTS)} points de vue capturés avec succès.")

    except Exception as e:
        logging.error(f"Une erreur majeure est survenue dans le script principal: {e}", exc_info=True)
    finally:
        # --- Nettoyage Global ---
        logging.info("Début du nettoyage global...")
        if client and world: # Vérifier si la connexion a réussi
             # Nettoyer les PNJ spawnés par carla_env_manager
             if vehicle_ids or walker_ids or controller_ids:
                  carla_env_manager.cleanup_carla_environment(client, vehicle_ids, walker_ids, controller_ids)

             # Rétablir les settings originaux du monde (désactiver mode synchrone, etc.)
             if original_settings:
                  logging.info("Rétablissement des paramètres originaux du monde.")
                  world.apply_settings(original_settings)
        else:
             logging.warning("Client ou Monde non initialisé, nettoyage PNJ et settings ignoré.")

        logging.info("Script principal terminé.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Génère un dataset BEV sémantique pour plusieurs points de vue dans CARLA.')
    parser.add_argument('--host', default=HOST, help='IP du serveur CARLA')
    parser.add_argument('--port', default=PORT, type=int, help='Port TCP du serveur CARLA')
    parser.add_argument('--save_dir', default=BASE_SAVE_DIR, help='Dossier racine pour sauvegarder le dataset')
    parser.add_argument('--num_vehicles', default=NUM_VEHICLES, type=int, help='Nombre de véhicules PNJ')
    parser.add_argument('--num_peds', default=NUM_PEDESTRIANS, type=int, help='Nombre de piétons PNJ')
    parser.add_argument('--fog', default=FOG_DENSITY, type=float, help='Densité du brouillard (0-1)')
    # Ajouter d'autres arguments pour configurer capteurs, grille, etc.
    parser.add_argument('--width', default=IMAGE_WIDTH, type=int, help='Largeur image capteur')
    parser.add_argument('--height', default=IMAGE_HEIGHT, type=int, help='Hauteur image capteur')
    parser.add_argument('--fov', default=CAMERA_FOV, type=float, help='Champ de vision caméra')
    parser.add_argument('--sensor_z', default=DEFAULT_SENSOR_Z, type=float, help='Hauteur capteur BEV')
    parser.add_argument('--sensor_pitch', default=DEFAULT_SENSOR_PITCH, type=float, help='Inclinaison capteur BEV')
    parser.add_argument('--voxel_size', default=VOXEL_SIZE, type=float, help='Taille Voxel (m)')
    parser.add_argument('--grid_x', default=GRID_RANGE_X, type=float, help='Portée grille X (m)')
    parser.add_argument('--grid_y', default=GRID_RANGE_Y, type=float, help='Portée grille Y (m)')
    parser.add_argument('--grid_z', default=GRID_RANGE_Z, type=float, help='Portée grille Z (m)')

    args = parser.parse_args()

    # Créer le dossier de base si nécessaire
    os.makedirs(args.save_dir, exist_ok=True)

    main(args)