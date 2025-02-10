# controller.py
import math
import carla

def compute_control(vehicle, target_wp, target_speed, steer_gain=0.02):
    """
    Calcule les commandes de contrôle pour suivre le waypoint cible.
    Retourne un tuple (control, reached) où 'reached' est True si le waypoint est atteint.
    """
    control = carla.VehicleControl()
    v_loc = vehicle.get_location()
    wp_loc = target_wp.transform.location
    distance = v_loc.distance(wp_loc)
    
    # Si le véhicule est proche du waypoint, on considère qu'il l'a atteint
    if distance < 2.0:
        return control, True

    # Calcul de l'angle désiré en fonction de la position du waypoint
    vehicle_yaw = vehicle.get_transform().rotation.yaw
    desired_yaw = math.degrees(math.atan2(wp_loc.y - v_loc.y, wp_loc.x - v_loc.x))
    yaw_error = desired_yaw - vehicle_yaw
    # Normalisation de l'erreur dans [-180, 180]
    while yaw_error > 180:
        yaw_error -= 360
    while yaw_error < -180:
        yaw_error += 360
    control.steer = max(-1.0, min(1.0, steer_gain * yaw_error))
    
    # Contrôle de la vitesse
    vel = vehicle.get_velocity()
    speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
    if speed < target_speed:
        control.throttle = 0.5
        control.brake = 0.0
    else:
        control.throttle = 0.0
        control.brake = 0.1
    return control, False

def check_stop_zones(vehicle_location, zones):
    """
    Vérifie si la position du véhicule se trouve dans une zone d'arrêt.
    Retourne un tuple (stop_active, duration).
    """
    for zone in zones:
        if vehicle_location.distance(zone['location']) < zone['radius']:
            return True, zone['duration']
    return False, 0
