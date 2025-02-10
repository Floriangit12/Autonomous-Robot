#!/usr/bin/env python
# main.py
import carla
import random
import time

from sensors import spawn_sensors
from route_planner import generate_route
from controller import compute_control, check_stop_zones
import config

def main():
    ##############################
    # 1. Connexion à Carla et configuration
    ##############################
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    
    print("Connexion à Carla réussie, carte :", world.get_map().name)
    
    ##############################
    # 2. Spawn du véhicule (robot)
    ##############################
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("Aucun point de spawn disponible.")
        return
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle is None:
        print("Le spawn du véhicule a échoué.")
        return
    print("Véhicule spawné en :", spawn_point.location)
    
    ##############################
    # 3. Attache des capteurs
    ##############################
    sensor_list = spawn_sensors(world, vehicle, blueprint_library)
    
    ##############################
    # 4. Génération de l'itinéraire sur les voies autorisées
    ##############################
    # Récupération du waypoint le plus proche de la position actuelle du véhicule
    current_wp = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=False)
    if current_wp.lane_type not in config.ALLOWED_LANE_TYPES:
        # Si le waypoint courant n'est pas sur une voie autorisée,
        # on cherche dans les environs un waypoint acceptable.
        all_wps = world.get_map().generate_waypoints(2.0)
        allowed_wps = [wp for wp in all_wps 
                       if wp.lane_type in config.ALLOWED_LANE_TYPES and
                          wp.transform.location.distance(vehicle.get_location()) < 50.0]
        if allowed_wps:
            current_wp = allowed_wps[0]
        else:
            print("Aucun waypoint autorisé trouvé proche du véhicule, on continue avec le waypoint actuel.")
    
    route = generate_route(current_wp, config.ALLOWED_LANE_TYPES, config.NUM_ROUTE_STEPS, config.STEP_DISTANCE)
    print("Itinéraire généré avec {} waypoints.".format(len(route)))
    
    ##############################
    # 5. Boucle de contrôle du véhicule
    ##############################
    current_wp_index = 0
    stop_active = False
    stop_timer = 0.0

    print("Démarrage de la boucle de contrôle. Appuyez sur Ctrl+C pour arrêter.")
    try:
        while True:
            world.tick()
            vehicle_location = vehicle.get_location()
            
            # Vérifier si le véhicule se trouve dans une zone d'arrêt ou un passage piétons
            is_stop_zone, duration_stop = check_stop_zones(vehicle_location, config.STOP_ZONES)
            is_ped_crossing, duration_ped = check_stop_zones(vehicle_location, config.PED_CROSSINGS)
            if is_stop_zone or is_ped_crossing:
                stop_active = True
                # Choix de la durée d'arrêt (ici, on prend la durée la plus élevée)
                stop_timer = max(duration_stop, duration_ped)
                print("Zone d'arrêt détectée, durée d'arrêt: {} sec".format(stop_timer))
            
            if stop_active:
                # Si on est dans une zone d'arrêt, le véhicule freine complètement
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                stop_timer -= config.TICK_SLEEP
                if stop_timer <= 0:
                    stop_active = False
                    print("Fin de la période d'arrêt, reprise du mouvement.")
            else:
                # Suivre l'itinéraire
                if current_wp_index < len(route):
                    target_wp = route[current_wp_index]
                    control, reached = compute_control(vehicle, target_wp, config.TARGET_SPEED)
                    if reached:
                        current_wp_index += 1
                        continue
                else:
                    print("Fin de l'itinéraire atteinte.")
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.brake = 1.0
            
            vehicle.apply_control(control)
            time.sleep(config.TICK_SLEEP)
    
    except KeyboardInterrupt:
        print("Simulation interrompue par l'utilisateur.")
    finally:
        # Nettoyage : arrêter et détruire tous les capteurs et le véhicule
        for sensor in sensor_list:
            sensor.stop()
            sensor.destroy()
        vehicle.destroy()
        print("Tous les acteurs ont été détruits. Fin du script.")

if __name__ == '__main__':
    main()
