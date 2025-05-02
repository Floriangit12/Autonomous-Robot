import carla
from Camera import *
from RobotCamera import *
from displayCamRobot import *


if __name__ == '__main__':
    client = None
    robot_cam = None
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(15.0) # Augmenter un peu le timeout pour le spawn multiple
        world = client.get_world()

        # Choisir un point de départ pour le robot
        start_location = carla.Location(x=0, y=2.0, z=0.5) # z=0.5 pour être sûr d'être au-dessus du sol
        start_rotation = carla.Rotation(yaw=0)
        start_transform = carla.Transform(start_location, start_rotation)

        # Créer l'instance du robot avec ses caméras
        print("Création de l'instance RobotCamera...")
        robot_cam = RobotCamera(
            world=world,
            transform=start_transform,
            name="PetitRobot",
            common_res=[480, 720], # Plus petite résolution pour l'affichage combiné
            common_fov=100.0,
            common_fps=20.0
        )
        print("RobotCamera créé.")

        # Attendre un court instant pour que les premières images arrivent
        time.sleep(1.0)

        # --- Option 1: Afficher le flux en direct ---
        display_robot_cameras(robot_cam)

        # --- Option 2: Déplacer et prendre des snapshots (après fermeture de la fenêtre) ---
        print("\nDéplacement du robot et prise de snapshots...")
        # Déplacer un peu vers l'avant
        new_location = start_location + carla.Location(x=5.0) # Avancer de 5m
        new_transform = carla.Transform(new_location, start_rotation)
        robot_cam.set_transform(new_transform)
        print(f"Robot déplacé vers {new_location}")

        # Attendre que les caméras capturent la nouvelle vue
        time.sleep(1.0)

        # Prendre les snapshots
        snapshots = robot_cam.get_snapshots()
        print(f"Snapshots pris depuis la nouvelle position. Clés: {list(snapshots.keys())}")

        # Sauvegarder les snapshots (exemple pour la caméra avant)
        img_front = snapshots.get('front')
        if img_front is not None:
             # Convertir BGRA en BGR pour imwrite
             img_front_bgr = cv2.cvtColor(img_front, cv2.COLOR_BGRA2BGR)
             save_path = "_output/snapshot_front.png"
             cv2.imwrite(save_path, img_front_bgr)
             print(f"Snapshot 'front' sauvegardé dans {save_path}")
        else:
             print("Snapshot 'front' non disponible.")

        # Attendre un peu avant de nettoyer
        time.sleep(2)


    except (RuntimeError, ValueError) as e:
         print(f"Erreur lors de l'initialisation ou de l'exécution: {e}")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nNettoyage final...")
        if robot_cam:
            robot_cam.destroy()

        # Détruire les fenêtres OpenCV au cas où elles seraient restées ouvertes
        cv2.destroyAllWindows()

        # (Pas besoin de détruire client ici)
        print("Terminé.")