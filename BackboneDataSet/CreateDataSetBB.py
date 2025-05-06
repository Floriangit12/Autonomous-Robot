# commande : python dataset_generator.py input1.txt input2.txt --output-dir dataset


# format dataset a mettre dans un fichier txt
# NomDeLaCarte
# X Y Z YAW
# X Y Z YAW
# ...

# format de sortie 
# dataset/
# ├── 1/
# │   ├── rgb.png
# │   ├── depth.npy
# │   └── semantic.png
# ├── 2/
# ...


import carla
import os
import argparse
import queue
import numpy as np
import cv2 # Utilisé pour imwrite pour semantic.png

def main():
    parser = argparse.ArgumentParser(description='Generate CARLA dataset')
    parser.add_argument('input_files', nargs='+', help='Input text files with positions')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    # Configuration CAMERA
    IMAGE_WIDTH = 720
    IMAGE_HEIGHT = 480

    # Connexion à CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)
    
    # Création du dossier de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    sample_count = 1
    
    world = None
    original_settings = None # Pour stocker les paramètres originaux du monde

    try:
        for file_path in args.input_files:
            print(f"\nTraitement du fichier d'entrée : {file_path}")
            with open(file_path, 'r') as f:
                content = f.read().splitlines()
            
            if not content:
                print(f"  Avertissement : Le fichier {file_path} est vide. Passage au suivant.")
                continue

            map_name = content[0]
            positions_data = []
            for i, line_str in enumerate(content[1:]):
                parts = line_str.split()
                if len(parts) == 4:
                    try:
                        x, y, z, yaw = map(float, parts)
                        positions_data.append({'x': x, 'y': y, 'z': z, 'yaw': yaw})
                    except ValueError:
                        print(f"  Avertissement : Format de coordonnées invalide dans {file_path} à la ligne {i+2}: '{line_str}'. Position ignorée.")
                elif parts: # Si la ligne n'est pas vide mais n'a pas 4 éléments
                    print(f"  Avertissement : Format de ligne invalide dans {file_path} à la ligne {i+2}: '{line_str}'. Attendu 4 valeurs. Position ignorée.")
            
            if not positions_data:
                print(f"  Avertissement : Aucune position valide trouvée dans {file_path} pour la carte {map_name}. Fichier ignoré.")
                continue
            
            print(f"  Chargement de la carte : {map_name}...")
            try:
                world = client.load_world(map_name) # Charge la carte et retourne l'objet World
                if world is None: # Vérification supplémentaire si load_world peut retourner None
                    print(f"  Erreur : Impossible de charger la carte {map_name}. Fichier ignoré.")
                    continue
                print(f"  Carte {map_name} chargée.")
                original_settings = world.get_settings()
            except RuntimeError as e:
                print(f"  Erreur lors du chargement de la carte {map_name}: {e}. Fichier ignoré.")
                world = None # Assurer que world est None si le chargement échoue
                continue


            # Configuration mode synchrone
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05 # Une bonne valeur pour la stabilité des données capteur
            world.apply_settings(settings)
            print("  Mode synchrone activé.")

            # Blueprints des capteurs
            bp_lib = world.get_blueprint_library()
            rgb_bp = bp_lib.find('sensor.camera.rgb')
            depth_bp = bp_lib.find('sensor.camera.depth')
            semantic_bp = bp_lib.find('sensor.camera.semantic_segmentation')

            # Configuration des attributs communs des caméras
            for bp in [rgb_bp, depth_bp, semantic_bp]:
                bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
                bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
                bp.set_attribute('fov', '90')

            for pos_dict in positions_data: # C'est la boucle qui correspondait à la ligne 43
                x, y, z, yaw = pos_dict['x'], pos_dict['y'], pos_dict['z'], pos_dict['yaw']
                
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
                )
                
                sample_dir = os.path.join(args.output_dir, str(sample_count))
                os.makedirs(sample_dir, exist_ok=True)

                # Initialiser les variables de capteur à None
                rgb_cam, depth_cam, semantic_cam = None, None, None
                sensor_actors = [] # Liste pour garder une trace des acteurs créés

                try:
                    # Création des capteurs
                    rgb_cam = world.spawn_actor(rgb_bp, transform)
                    sensor_actors.append(rgb_cam)
                    depth_cam = world.spawn_actor(depth_bp, transform)
                    sensor_actors.append(depth_cam)
                    semantic_cam = world.spawn_actor(semantic_bp, transform)
                    sensor_actors.append(semantic_cam)
                    # print(f"    Capteurs créés pour la position ({x:.2f},{y:.2f},{z:.2f})")


                    # Création des files d'attente pour les données capteur
                    q_rgb = queue.Queue()
                    q_depth = queue.Queue()
                    q_semantic = queue.Queue()

                    # Attacher les listeners
                    rgb_cam.listen(q_rgb.put)
                    depth_cam.listen(q_depth.put)
                    semantic_cam.listen(q_semantic.put)

                    # Avancer la simulation d'un tick pour capturer les images
                    world.tick()

                    # Récupérer les données (avec un timeout pour éviter un blocage infini)
                    timeout_sec = 2.0
                    rgb_image_data = q_rgb.get(timeout=timeout_sec)
                    depth_image_data = q_depth.get(timeout=timeout_sec)
                    semantic_image_data = q_semantic.get(timeout=timeout_sec)
                    
                    # Arrêter l'écoute des capteurs pour éviter d'accumuler des données inutiles
                    # et pour libérer les ressources plus proprement avant la destruction.
                    for cam_actor in sensor_actors:
                        if cam_actor is not None and cam_actor.is_listening:
                            cam_actor.stop()

                    # Enregistrement RGB
                    rgb_image_data.save_to_disk(os.path.join(sample_dir, 'rgb.png'))

                    # Traitement et enregistrement Depth
                    depth_buffer = bytes(depth_image_data.raw_data) # Convertir carla.Buffer en bytes
                    depth_array = np.frombuffer(depth_buffer, dtype=np.float32)
                    depth_array = np.reshape(depth_array, (IMAGE_HEIGHT, IMAGE_WIDTH))
                    # depth_array contient la distance en mètres. Multiplier par 1000 pour mm si besoin.
                    np.save(os.path.join(sample_dir, 'depth.npy'), depth_array)

                    # Traitement et enregistrement Semantic Segmentation
                    # Les données sémantiques sont encodées dans le canal Rouge (R) d'une image BGRA par défaut.
                    semantic_buffer = bytes(semantic_image_data.raw_data) # Convertir carla.Buffer en bytes
                    semantic_array_bgra = np.frombuffer(semantic_buffer, dtype=np.uint8)
                    semantic_array_bgra = np.reshape(semantic_array_bgra, (IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                    semantic_labels = semantic_array_bgra[:, :, 2] # Extraire le canal Rouge (index 2 en BGRA)
                    cv2.imwrite(os.path.join(sample_dir, 'semantic.png'), semantic_labels)
                    
                    print(f"    Échantillon {sample_count} sauvegardé dans {sample_dir}")
                    sample_count += 1

                except queue.Empty: # queue.get() a expiré
                    print(f"  Avertissement : Timeout lors de la récupération des données capteur pour l'échantillon {sample_count} à la position ({x:.2f},{y:.2f},{z:.2f}). Échantillon ignoré.")
                except Exception as e_inner:
                    print(f"  Erreur lors du traitement de la position ({x:.2f},{y:.2f},{z:.2f}) pour l'échantillon {sample_count}: {e_inner}")
                finally:
                    # Nettoyage des capteurs pour cette position
                    # print(f"    Nettoyage des capteurs pour la position ({x:.2f},{y:.2f},{z:.2f})...")
                    for actor in sensor_actors:
                        if actor is not None:
                            if actor.is_listening: # S'assurer que l'écoute est arrêtée
                                actor.stop()
                            if actor.is_alive: # Vérifier si l'acteur est toujours valide
                                actor.destroy()
                    # print(f"    Capteurs nettoyés.")
            
            # Réinitialiser les paramètres du monde pour cette carte avant de passer à la suivante (si applicable)
            if world is not None and original_settings is not None:
                print(f"  Réinitialisation des paramètres pour la carte {map_name}.")
                world.apply_settings(original_settings)
                original_settings = None # Réinitialiser pour la prochaine carte
            world = None # S'assurer que la référence au monde est nettoyée

    # Correction de la clause except de votre bloc try externe
    except KeyboardInterrupt: # Gérer l'interruption utilisateur proprement
        print("\nProcessus de génération de dataset interrompu par l'utilisateur.")
    except Exception as e_outer:
        print(f"Une erreur majeure est survenue : {e_outer}")
    finally:
        # S'assurer que le dernier monde chargé est remis en mode asynchrone si nécessaire
        print("\nNettoyage final...")
        if world is not None and original_settings is not None: # Si une carte était encore chargée et non réinitialisée
            try:
                current_settings_final = world.get_settings()
                if current_settings_final.synchronous_mode:
                    world.apply_settings(original_settings)
                    print("  Paramètres du dernier monde réinitialisés en mode asynchrone.")
            except RuntimeError as e_final_ru: # Ex: client déconnecté
                print(f"  Erreur lors du nettoyage final des paramètres du monde: {e_final_ru}")
        
        print(f"Génération du dataset terminée. Total d'échantillons créés : {sample_count - 1}")

if __name__ == '__main__':
    main()