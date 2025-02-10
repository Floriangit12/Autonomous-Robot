# sensors.py
import carla

def process_image(image, side=''):
    """
    Callback pour le traitement ou l'enregistrement des images.
    Ici, on affiche simplement l'ID de frame.
    """
    print("Camera {}: frame {}".format(side, image.frame))
    # Vous pouvez ajouter ici le code pour sauvegarder l'image sur disque ou la traiter.

def process_gnss(gnss):
    """
    Callback pour le capteur GNSS.
    Affiche la latitude, la longitude et l'altitude.
    """
    print("GNSS: lat = {:.6f}, lon = {:.6f}, alt = {:.2f}".format(
        gnss.latitude, gnss.longitude, gnss.altitude))

def spawn_sensors(world, vehicle, blueprint_library):
    """
    Crée et attache les capteurs au véhicule :
      - Deux caméras (gauche et droite)
      - Un capteur GNSS
    Retourne la liste des acteurs capteurs.
    """
    sensor_list = []

    # --- Caméra gauche
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '800')  # Résolution ajustable
    cam_bp.set_attribute('image_size_y', '600')
    cam_bp.set_attribute('fov', '90')
    left_cam_transform = carla.Transform(carla.Location(x=1.5, y=-0.3, z=2.4))
    left_camera = world.spawn_actor(cam_bp, left_cam_transform, attach_to=vehicle)
    left_camera.listen(lambda image: process_image(image, side='gauche'))
    sensor_list.append(left_camera)

    # --- Caméra droite
    right_cam_transform = carla.Transform(carla.Location(x=1.5, y=0.3, z=2.4))
    right_camera = world.spawn_actor(cam_bp, right_cam_transform, attach_to=vehicle)
    right_camera.listen(lambda image: process_image(image, side='droite'))
    sensor_list.append(right_camera)

    # --- Capteur GNSS
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', '1.0')  # Fréquence de mise à jour (1 sec)
    gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.0))
    gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    gnss_sensor.listen(process_gnss)
    sensor_list.append(gnss_sensor)

    return sensor_list
