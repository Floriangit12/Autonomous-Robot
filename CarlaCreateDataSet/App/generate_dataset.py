# generate_dataset_from_scenario.py
import numpy as np
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import time
import sys
import glob # Pour trouver l'egg carla

# --- Import CARLA ---
try:
    carla_egg_path = glob.glob(os.path.join('C:', 'path', 'to', 'your', 'CARLA_xxx', 'PythonAPI', 'carla', 'dist', 'carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor, 'win-amd64' if os.name == 'nt' else 'linux-x86_64' )))[0]
    if not carla_egg_path: raise FileNotFoundError
    sys.path.append(carla_egg_path)
    import carla
    CARLA_AVAILABLE = True
    print(f"Bibliothèque CARLA importée depuis: {carla_egg_path}")
except Exception as e:
    print(f"ERREUR: Impossible d'importer CARLA: {e}")
    CARLA_AVAILABLE = False
    carla = None # Définir pour éviter erreurs
# --- Fin Import CARLA ---

# --- Importer les classes locales ---
try:
    from lidar_simulator import LidarSimulator, SEMANTIC_TAGS
    from semantic_voxelizer import SemanticVoxelizer
    # --- MODIFICATION IMPORT ---
    from Camera import CameraSensor # Importer votre VRAIE classe
    from robot_camera import RobotCamera # Importer depuis le fichier dédié
    # --- FIN MODIFICATION ---
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"ERREUR: Impossible d'importer les modules locaux nécessaires: {e}")
    MODULES_AVAILABLE = False
# --- Fin Imports Locaux ---

# =============================================================================
# === Fonctions Utilitaires ==================================================
# =============================================================================

def find_weather_presets():
    """Retourne un dictionnaire des presets météo CARLA."""
    if not CARLA_AVAILABLE: return {}
    # Utiliser dir() et getattr() pour trouver les presets dynamiquement
    preset_dict = {}
    for attr_name in dir(carla.WeatherParameters):
        if attr_name.isupper(): # Les presets sont généralement en majuscules
            try:
                preset_obj = getattr(carla.WeatherParameters, attr_name)
                # Vérifier si c'est bien un preset (instance de WeatherParameters)
                if isinstance(preset_obj, carla.WeatherParameters):
                    preset_dict[attr_name] = preset_obj
            except AttributeError:
                continue # Ignorer les autres attributs
    # Ajouter le preset par défaut s'il n'est pas déjà là (nom peut varier)
    if 'Default' not in preset_dict and hasattr(carla.WeatherParameters, 'Default'):
         preset_dict['Default'] = carla.WeatherParameters.Default
    return preset_dict

def parse_scenario_file(filepath):
    """Lit un fichier scénario avec commandes POSE, WEATHER, TIME, LIDARVOXEL, CAMERAS."""
    commands = []
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue # Ignorer lignes vides et commentaires

                parts = line.split(':')
                if len(parts) < 2:
                    print(f"Warn: Ligne {line_num+1} ignorée (format invalide): {line}")
                    continue

                command_type = parts[0].strip().upper()
                command_value = ':'.join(parts[1:]).strip() # Rejoindre si ':' dans la valeur (ex: heure)
                values = command_value.split()

                cmd_data = {'type': command_type}
                try:
                    if command_type == 'POSE':
                        if len(values) < 3: raise ValueError("POSE nécessite au moins X Y Z")
                        cmd_data['location'] = [float(v) for v in values[:3]]
                        # Yaw, Pitch, Roll optionnels (défaut 0)
                        rot = [0.0, 0.0, 0.0] # Ordre P, Y, R pour carla.Rotation
                        if len(values) >= 4: rot[1] = float(values[3]) # Yaw
                        if len(values) >= 5: rot[0] = float(values[4]) # Pitch
                        if len(values) >= 6: rot[2] = float(values[5]) # Roll
                        cmd_data['rotation'] = rot
                        commands.append(cmd_data)
                    elif command_type == 'WEATHER':
                        if not values: raise ValueError("WEATHER nécessite un nom/ID de preset")
                        cmd_data['preset'] = values[0] # Garder comme string/int pour l'instant
                        commands.append(cmd_data)
                    elif command_type == 'TIME':
                        if not values: raise ValueError("TIME nécessite une heure (ex: HH:MM)")
                        time_parts = values[0].split(':')
                        if len(time_parts) != 2: raise ValueError("Format heure invalide (HH:MM)")
                        cmd_data['hour'] = int(time_parts[0])
                        cmd_data['minute'] = int(time_parts[1])
                        commands.append(cmd_data)
                    elif command_type == 'LIDARVOXEL':
                         cmd_data['generate'] = values[0].lower() == 'true' if values else False
                         # Ajouter à la dernière commande POSE si possible, sinon ignorer ?
                         # Pour la simplicité, on l'ajoute comme commande séparée
                         # Mais il serait mieux de l'associer à une POSE
                         # On va l'associer implicitement à la POSE *précédente* ou *suivante*
                         # Simplification: On l'ajoute à la liste, la boucle principale décidera
                         commands.append(cmd_data)
                    elif command_type == 'CAMERAS':
                         cmd_data['generate'] = values[0].lower() == 'true' if values else False
                         commands.append(cmd_data)
                    else:
                        print(f"Warn: Ligne {line_num+1} ignorée (commande inconnue): {command_type}")

                except ValueError as e:
                    print(f"Warn: Ligne {line_num+1} ignorée (erreur valeur '{line}'): {e}")

    except FileNotFoundError:
        print(f"ERREUR: Fichier scénario non trouvé: {filepath}")
        return None
    return commands

def set_carla_time(world, hour, minute):
    """ Tente de régler l'heure en ajustant la position du soleil. APPROXIMATIF. """
    if not CARLA_AVAILABLE: return
    # Convertir HH:MM en angle solaire (très approx)
    # 06:00 -> 0 deg, 12:00 -> 90 deg, 18:00 -> 0 deg, 00:00 -> -90 deg
    total_minutes = hour * 60 + minute
    # Normaliser entre 0 et 1 pour un cycle jour/nuit
    time_fraction = total_minutes / (24 * 60)
    # Mapper sur un cycle sinusoïdal pour l'angle solaire (-90 à +90)
    sun_angle = 90 * math.sin(time_fraction * math.pi * 2 - math.pi / 2) # Décalé pour midi=90

    try:
        weather = world.get_weather()
        weather.sun_altitude_angle = sun_angle
        world.set_weather(weather)
        print(f"Heure CARLA réglée approximativement à {hour:02d}:{minute:02d} (Angle soleil: {sun_angle:.1f} deg)")
    except Exception as e:
        print(f"Erreur lors du réglage de l'heure CARLA: {e}")

# =============================================================================
# === Fonction Principale de Génération =======================================
# =============================================================================

def generate(args):
    """Fonction principale pour générer le dataset."""
    if not CARLA_AVAILABLE: print("ERREUR: Bibliothèque CARLA non chargée."); return
    if not MODULES_AVAILABLE: print("ERREUR: Modules locaux (Lidar, Voxelizer, Camera) non chargés."); return

    client = None
    robot_camera_instance = None # Renommer pour éviter conflit avec classe
    lidar_sim = None
    voxelizer = None
    world = None

    try:
        # 1. Connexion à CARLA
        print(f"Connexion à CARLA sur {args.carla_host}:{args.carla_port}...")
        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"Connecté à CARLA. Map: {os.path.basename(world.get_map().name)}")
        weather_presets = find_weather_presets()
        print(f"Presets météo disponibles: {list(weather_presets.keys())}")

        # 2. Lire le fichier scénario
        commands = parse_scenario_file(args.scenario_file)
        if commands is None or len(commands) == 0: print("Aucune commande valide trouvée."); return
        print(f"{len(commands)} commandes lues depuis {args.scenario_file}")

        # 3. Initialiser les capteurs et le robot
        # Trouver la première pose pour l'initialisation
        initial_transform = carla.Transform() # Défaut à l'origine
        first_pose_found = False
        for cmd in commands:
            if cmd['type'] == 'POSE':
                loc = cmd['location']; rot = cmd['rotation']
                initial_transform = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                                  carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
                first_pose_found = True
                break
        if not first_pose_found: print("Avertissement: Aucune commande POSE trouvée pour la position initiale.");

        lidar_sim = LidarSimulator(points_per_scan=args.lidar_points, fps=args.lidar_fps, max_range=args.lidar_range)
        voxelizer = SemanticVoxelizer(voxel_size=tuple(args.voxel_size), point_cloud_range=args.grid_range, allowed_tags=args.allowed_tags)
        cam_resolution = [args.img_width, args.img_height] # W, H pour CARLA
        # Noms des caméras utilisés par RobotCamera et pour le mapping de sortie
        robot_cam_ids = ['front', 'back', 'left', 'right']
        output_cam_names = ['Top', 'Bottom', 'Left', 'Right'] # Noms pour sauvegarder
        if len(robot_cam_ids) != len(output_cam_names): raise ValueError("Mismatch camera names")
        cam_name_mapping = dict(zip(robot_cam_ids, output_cam_names))

        robot_camera_instance = RobotCamera(world=world, transform=initial_transform, name="gen_robot", common_res=cam_resolution)

        # 4. Créer le dossier de sortie
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Sauvegarde du dataset dans: {args.output_dir}")

        # 5. Boucle de génération basée sur les commandes
        frame_counter = 0 # Compteur pour les dossiers de sortie
        current_pose = initial_transform # Garder la pose actuelle
        generate_lidar_next = False
        generate_cameras_next = False

        for cmd_idx, command in enumerate(tqdm(commands, desc="Processing Commands")):
            cmd_type = command['type']

            if cmd_type == 'POSE':
                # Si des captures étaient prévues pour la POSE précédente, on les fait MAINTENANT
                # avant de bouger le robot.
                if generate_lidar_next or generate_cameras_next:
                    frame_counter += 1
                    frame_dir = os.path.join(args.output_dir, str(frame_counter))
                    os.makedirs(frame_dir, exist_ok=True)
                    print(f"\n--- Génération Frame {frame_counter} @ Pose: {current_pose.location} ---")

                    # Générer Lidar/Voxel si demandé
                    if generate_lidar_next:
                        try:
                            lidar_sim.position = np.array([current_pose.location.x, current_pose.location.y, current_pose.location.z])
                            scan_points = lidar_sim.generate_scan()
                            voxel_grid = voxelizer.voxelize(scan_points)
                            voxel_save_path = os.path.join(frame_dir, "voxel.npy")
                            np.save(voxel_save_path, voxel_grid)
                            print(f"  Voxel grid saved ({voxel_grid.shape}).")
                        except Exception as e: print(f"  ERREUR Lidar/Voxel: {e}")

                    # Capturer/Sauver Caméras si demandé
                    if generate_cameras_next:
                        try:
                            cam_snapshots_data = robot_camera_instance.get_snapshots()
                            saved_cams = []
                            for cam_key, img_data in cam_snapshots_data.items():
                                output_cam_name = cam_name_mapping.get(cam_key)
                                if img_data is not None and output_cam_name:
                                    if img_data.shape[2] == 4: img_data_rgb = img_data[:,:,:3][:,:,::-1] # BGRA -> RGB
                                    elif img_data.shape[2] == 3: img_data_rgb = img_data
                                    else: warnings.warn(f"Format image inattendu: {img_data.shape}"); continue
                                    img_pil = Image.fromarray(img_data_rgb)
                                    img_save_path = os.path.join(frame_dir, f"{output_cam_name}.png")
                                    img_pil.save(img_save_path)
                                    saved_cams.append(output_cam_name)
                            print(f"  Images sauvegardées: {saved_cams}")
                        except Exception as e: print(f"  ERREUR Caméras: {e}")

                # Réinitialiser les flags pour la prochaine frame
                generate_lidar_next = False
                generate_cameras_next = False

                # Mettre à jour la pose pour la prochaine itération
                loc = command['location']; rot = command['rotation']
                current_pose = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                             carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]))
                robot_camera_instance.set_transform(current_pose)
                #print(f"Pose mise à jour: {current_pose.location}")

            elif cmd_type == 'WEATHER':
                preset_name = command['preset']
                if preset_name in weather_presets:
                    world.set_weather(weather_presets[preset_name])
                    print(f"Météo changée en: {preset_name}")
                else:
                    print(f"Avertissement: Preset météo '{preset_name}' inconnu.")

            elif cmd_type == 'TIME':
                set_carla_time(world, command['hour'], command['minute'])

            elif cmd_type == 'LIDARVOXEL':
                generate_lidar_next = command['generate']

            elif cmd_type == 'CAMERAS':
                generate_cameras_next = command['generate']

            # Faire avancer la simulation d'un pas après chaque commande (surtout après POSE)
            world.tick()

        # --- Fin de la boucle ---

        # Générer la dernière frame si nécessaire (si la dernière commande était POSE + flags True)
        if generate_lidar_next or generate_cameras_next:
            frame_counter += 1
            frame_dir = os.path.join(args.output_dir, str(frame_counter))
            os.makedirs(frame_dir, exist_ok=True)
            print(f"\n--- Génération Frame {frame_counter} @ Pose: {current_pose.location} ---")
            if generate_lidar_next:
                 try: lidar_sim.position = np.array([current_pose.location.x, current_pose.location.y, current_pose.location.z]); scan_points = lidar_sim.generate_scan(); voxel_grid = voxelizer.voxelize(scan_points); np.save(os.path.join(frame_dir, "voxel.npy"), voxel_grid); print(f"  Voxel grid saved.")
                 except Exception as e: print(f"  ERREUR Lidar/Voxel: {e}")
            if generate_cameras_next:
                 try:
                     cam_snapshots_data = robot_camera_instance.get_snapshots(); saved_cams = []
                     for cam_key, img_data in cam_snapshots_data.items():
                         output_cam_name = cam_name_mapping.get(cam_key)
                         if img_data is not None and output_cam_name:
                             if img_data.shape[2] == 4: img_data_rgb = img_data[:,:,:3][:,:,::-1]
                             elif img_data.shape[2] == 3: img_data_rgb = img_data
                             else: continue
                             img_pil = Image.fromarray(img_data_rgb); img_save_path = os.path.join(frame_dir, f"{output_cam_name}.png"); img_pil.save(img_save_path); saved_cams.append(output_cam_name)
                     print(f"  Images sauvegardées: {saved_cams}")
                 except Exception as e: print(f"  ERREUR Caméras: {e}")

        print(f"\nDataset généré avec {frame_counter} frames dans {args.output_dir}")

    except Exception as e:
        print(f"\n--- ERREUR SCRIPT ---"); import traceback; traceback.print_exc(); print(f"Error: {e}")
    finally:
        # --- Nettoyage ---
        print("Nettoyage des acteurs CARLA...")
        if robot_camera_instance:
            robot_camera_instance.destroy()
        # TODO: Revenir en mode asynchrone si changé
        print("Nettoyage terminé.")


# =============================================================================
# === Point d'Entrée Principal ==============================================
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Générer un dataset simulé Lidar(Voxel)/Camera avec CARLA via fichier scénario.')

    # Arguments Chemin
    parser.add_argument('--scenario_file', type=str, required=True, help='Chemin vers le fichier texte décrivant le scénario (poses, météo, temps, captures).')
    parser.add_argument('--output_dir', type=str, default='./generated_carla_dataset', help='Dossier où sauvegarder le dataset généré.')
    parser.add_argument('--carla-host', type=str, default='localhost', help='Adresse IP du serveur CARLA.')
    parser.add_argument('--carla-port', type=int, default=2000, help='Port TCP du serveur CARLA.')

    # Arguments Simulation/Capteurs
    parser.add_argument('--num_frames', type=int, default=-1, help='Nombre max de frames (poses) à générer (-1 pour toutes).')
    parser.add_argument('--lidar_points', type=int, default=80000, help='Nb points par scan Lidar simulé.')
    parser.add_argument('--lidar_fps', type=int, default=10, help='FPS simulé Lidar (affecte aussi caméras et tick).')
    parser.add_argument('--lidar_range', type=float, default=80.0, help='Portée max Lidar (m).')
    # Note: Camera names/FOV sont définis dans RobotCamera, mais résolution peut être passée
    parser.add_argument('--img_height', type=int, default=480, help='Hauteur des images caméra (commune).')
    parser.add_argument('--img_width', type=int, default=720, help='Largeur des images caméra (commune).')

    # Arguments Voxelisation
    parser.add_argument('--voxel_size', type=float, nargs=3, default=[0.4, 0.4, 0.4], help='Taille Voxel (dx dy dz) en mètres.')
    parser.add_argument('--grid_range', type=float, nargs=6, default=[-40, -40, -3, 40, 40, 5], help='Limites grille [xmin ymin zmin xmax ymax zmax] relatives au Lidar.')
    parser.add_argument('--allowed_tags', type=int, nargs='*', default=None, help='Optionnel: Liste IDs de tags sémantiques à inclure.')

    args = parser.parse_args()

    # Lancer la génération
    if CARLA_AVAILABLE and MODULES_AVAILABLE:
        generate(args)
    else:
        print("Impossible de lancer la génération à cause d'erreurs d'import.")