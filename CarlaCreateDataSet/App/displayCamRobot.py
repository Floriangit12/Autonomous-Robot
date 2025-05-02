# Installer opencv-python si nécessaire: pip install opencv-python
import cv2

def display_robot_cameras(robot_camera_instance, window_name="Robot Camera Feeds"):
    """
    Affiche les flux des 4 caméras d'une instance RobotCamera dans une seule fenêtre.
    Appuyez sur 'q' pour quitter.

    :param robot_camera_instance: L'objet RobotCamera à afficher.
    :param window_name: Le nom de la fenêtre OpenCV.
    """
    print(f"Démarrage de l'affichage pour {robot_camera_instance.name}. Appuyez sur 'q' dans la fenêtre pour quitter.")

    # Récupérer la résolution pour dimensionner la fenêtre d'affichage
    # Suppose que toutes les caméras ont la même résolution
    res_w, res_h = 0, 0
    if robot_camera_instance.cameras:
        first_cam = next(iter(robot_camera_instance.cameras.values()))
        res_w = first_cam.resolution_width
        res_h = first_cam.resolution_height
    else:
        print("Erreur: Aucune caméra trouvée dans l'instance RobotCamera.")
        return

    if res_w == 0 or res_h == 0:
        print("Erreur: Résolution de caméra invalide.")
        return

    # Créer une image noire de fond pour les cas où une image manque
    black_image = np.zeros((res_h, res_w, 3), dtype=np.uint8)

    while True:
        # Récupérer les dernières images (données numpy BGRA ou None)
        images_data = robot_camera_instance.get_snapshots()

        # Préparer les images pour l'affichage (convertir BGRA -> BGR)
        display_images = {}
        for cam_id in ['front', 'back', 'left', 'right']:
            img_bgra = images_data.get(cam_id) # Utilise .get pour éviter KeyError si une caméra manque
            if img_bgra is not None and img_bgra.shape == (res_h, res_w, 4):
                # Convertir BGRA en BGR pour OpenCV imshow
                img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
                display_images[cam_id] = img_bgr
            else:
                # Afficher une image noire si les données manquent ou sont invalides
                # print(f"Warning: No valid image for {cam_id}") # Peut être verbeux
                display_images[cam_id] = black_image

        # Assembler les images en une grille 2x2
        top_row = np.hstack((display_images['front'], display_images['right']))
        bottom_row = np.hstack((display_images['back'], display_images['left']))
        combined_image = np.vstack((top_row, bottom_row))

        # Afficher l'image combinée
        cv2.imshow(window_name, combined_image)

        # Attendre une touche, quitter si 'q' est pressé
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fermer la fenêtre à la fin
    cv2.destroyAllWindows()
    print("Fenêtre d'affichage fermée.")