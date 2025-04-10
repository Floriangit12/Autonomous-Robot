#!/usr/bin/env python

import carla
import random
import time
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_carla_environment(
    client,
    world,
    num_vehicles=50,
    num_pedestrians=100,
    fog_density=0.1,
    fog_distance=10.0, # Distance de début du brouillard exponentiel
    sun_altitude=-10.0 # Soleil bas pour ombres longues (ou positif pour jour)
):
    """
    Configure l'environnement CARLA avec des véhicules, piétons et une météo spécifique.

    Args:
        client: L'objet client CARLA.
        world: L'objet monde CARLA.
        num_vehicles: Nombre de véhicules PNJ à faire spawner.
        num_pedestrians: Nombre de piétons PNJ à faire spawner.
        fog_density: Densité du brouillard (0 à 1).
        fog_distance: Distance de début du brouillard.
        sun_altitude: Angle d'altitude du soleil (degrés).

    Returns:
        tuple: Contenant les listes des IDs des acteurs spawnés
               (vehicle_ids, walker_ids, controller_ids).
    """
    vehicle_ids = []
    walker_ids = []
    controller_ids = []
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        logging.warning("La carte ne contient pas de points de spawn prédéfinis.")
        # Utiliser des emplacements aléatoires comme solution de repli, peut être moins fiable
        # return [], [], [] # Ou lever une exception

    logging.info(f"Configuration de l'environnement: {num_vehicles} véhicules, {num_pedestrians} piétons, brouillard={fog_density}")

    try:
        # === Configuration Météo ===
        weather = world.get_weather()
        weather.fog_density = fog_density
        weather.fog_distance = fog_distance
        weather.sun_altitude_angle = sun_altitude
        # weather.sun_azimuth_angle = 0 # Position du soleil
        # weather.precipitation = 0 # Pluie
        # weather.wind_intensity = 0
        world.apply_weather(weather)
        logging.info(f"Météo appliquée: Brouillard densité={weather.fog_density}, Altitude Soleil={weather.sun_altitude_angle}")
        world.tick() # Appliquer la météo

        # === Spawn des Véhicules PNJ ===
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter('vehicle.*')
        # Exclure vélos/motos si désiré
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

        if len(vehicle_bps) == 0:
             logging.warning("Aucun blueprint de véhicule à 4 roues trouvé.")
        else:
            logging.info(f"Tentative de spawn de {num_vehicles} véhicules PNJ...")
            count_vehicles = 0
            random.shuffle(spawn_points) # Mélanger pour éviter concentration
            for spawn_point in spawn_points:
                if count_vehicles >= num_vehicles:
                    break
                bp = random.choice(vehicle_bps)
                vehicle = world.try_spawn_actor(bp, spawn_point)
                if vehicle is not None:
                    vehicle.set_autopilot(True, client.get_trafficmanager_port())
                    vehicle_ids.append(vehicle.id)
                    count_vehicles += 1
                # else: logging.debug(f"Spawn véhicule échoué à {spawn_point.location}") # Peut être verbeux
            logging.info(f"{len(vehicle_ids)} véhicules PNJ spawnés avec succès.")
            if len(vehicle_ids) < num_vehicles:
                 logging.warning(f"N'a pu spawner que {len(vehicle_ids)} sur {num_vehicles} véhicules demandés.")

        # === Spawn des Piétons PNJ ===
        walker_bps = blueprint_library.filter('walker.pedestrian.*')
        # Contrôleur AI pour les piétons
        walker_controller_bp = blueprint_library.find('controller.ai.walker')

        if len(walker_bps) == 0:
             logging.warning("Aucun blueprint de piéton trouvé.")
        elif not world.get_map().get_spawn_points(): # Besoin de navigation pour piétons
            logging.warning("Carte sans points de spawn ou navigation, impossible de spawner les piétons correctement.")
        else:
            logging.info(f"Tentative de spawn de {num_pedestrians} piétons PNJ...")
            count_walkers = 0
            # Spawner les contrôleurs d'abord par batch peut être plus rapide
            batch_controller = []
            for _ in range(num_pedestrians):
                 batch_controller.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), '0')) # Attach later

            results_controller = client.apply_batch_sync(batch_controller, True) # True pour continuer même si erreur
            valid_controller_ids = []
            for i, res in enumerate(results_controller):
                if not res.error:
                    valid_controller_ids.append(res.actor_id)
                # else: logging.debug(f"Erreur spawn contrôleur {i}: {res.error}")

            # Spawner les piétons et les attacher aux contrôleurs
            batch_walker = []
            walker_controller_map = {} # Pour lier walker_id à controller_id
            spawned_locations = set() # Pour éviter de spawner au même endroit

            for controller_id in valid_controller_ids:
                 # Trouver un point de spawn de navigation aléatoire non utilisé
                spawn_loc = None
                for _ in range(5): # Essayer 5 fois de trouver un lieu unique
                    loc = world.get_random_location_from_navigation()
                    if loc and tuple(int(c*10) for c in [loc.x, loc.y, loc.z]) not in spawned_locations: # Clé approximative
                        spawn_loc = loc
                        spawned_locations.add(tuple(int(c*10) for c in [loc.x, loc.y, loc.z]))
                        break

                if spawn_loc is None:
                    # logging.debug("Impossible de trouver un point de spawn unique pour piéton.")
                    continue # Passer au suivant

                walker_bp = random.choice(walker_bps)
                # Rendre les piétons invincibles (optionnel)
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'true')

                transform = carla.Transform(spawn_loc)
                # Utiliser SpawnActor pour pouvoir récupérer l'ID et lier au contrôleur
                batch_walker.append(carla.command.SpawnActor(walker_bp, transform))


            # Appliquer le batch de spawn des piétons
            results_walker = client.apply_batch_sync(batch_walker, True)

            # Lier piétons et contrôleurs, démarrer les contrôleurs
            paired_controllers = 0
            controller_idx = 0
            for i, res in enumerate(results_walker):
                 if not res.error and controller_idx < len(valid_controller_ids):
                      walker_id = res.actor_id
                      controller_id = valid_controller_ids[controller_idx]
                      walker_ids.append(walker_id)
                      controller_ids.append(controller_id) # Garder trace pour cleanup
                      controller_actor = world.get_actor(controller_id)
                      if controller_actor:
                           # Attacher le contrôleur au piéton (important !)
                           # Il semble que l'attachement n'est pas direct via une commande
                           # Le contrôleur prend l'ID du parent lors de son start() s'il est spawné attaché
                           # Alternative: contrôler manuellement via API
                           controller_actor.start()
                           controller_actor.go_to_location(world.get_random_location_from_navigation())
                           controller_actor.set_max_speed(1 + random.random())  # Vitesse variable
                           paired_controllers += 1
                      controller_idx += 1 # Passer au contrôleur suivant


            logging.info(f"{len(walker_ids)} piétons PNJ spawnés et {paired_controllers} contrôleurs démarrés.")
            if len(walker_ids) < num_pedestrians:
                 logging.warning(f"N'a pu spawner que {len(walker_ids)} sur {num_pedestrians} piétons demandés.")

        # Attendre un peu pour que les acteurs se mettent en place
        for _ in range(5):
             world.tick()

        logging.info("Configuration de l'environnement terminée.")
        return vehicle_ids, walker_ids, controller_ids

    except Exception as e:
        logging.error(f"Erreur lors de la configuration de l'environnement: {e}", exc_info=True)
        # Essayer de nettoyer ce qui a déjà été spawné en cas d'erreur partielle
        cleanup_carla_environment(client, vehicle_ids, walker_ids, controller_ids)
        return [], [], [] # Retourner des listes vides en cas d'échec


def cleanup_carla_environment(client, vehicle_ids, walker_ids, controller_ids):
    """
    Nettoie les acteurs PNJ (véhicules, piétons, contrôleurs) spawnés.

    Args:
        client: L'objet client CARLA.
        vehicle_ids: Liste des IDs des véhicules à détruire.
        walker_ids: Liste des IDs des piétons à détruire.
        controller_ids: Liste des IDs des contrôleurs AI à détruire.
    """
    logging.info("Nettoyage de l'environnement PNJ...")

    # Arrêter les contrôleurs AI
    all_controller_actors = []
    # Tenter de récupérer les acteurs contrôleurs (peuvent déjà être détruits)
    if controller_ids:
        controllers_batch_fetch = [carla.command.GetActorState(actor_id) for actor_id in controller_ids]
        # Il n'y a pas de GetActor directement en batch, utiliser GetActorState ou GetActors(ids)
        # Alternative: Boucle simple world.get_actor()
        for actor_id in controller_ids:
             actor = client.get_world().get_actor(actor_id)
             if actor: all_controller_actors.append(actor)

    # Batch pour arrêter les contrôleurs (s'ils existent encore)
    if all_controller_actors:
         batch_stop_controller = [carla.command.StopWalkerController(actor.id) for actor in all_controller_actors]
         try:
              client.apply_batch_sync(batch_stop_controller, True) # True = ignorer les erreurs (ex: acteur déjà détruit)
              logging.info(f"{len(batch_stop_controller)} contrôleurs AI arrêtés (ou tentative).")
         except Exception as e_stop:
              logging.warning(f"Erreur lors de l'arrêt des contrôleurs AI : {e_stop}")

    # Batch pour détruire tous les acteurs (véhicules, piétons, contrôleurs)
    all_ids_to_destroy = list(vehicle_ids) + list(walker_ids) + list(controller_ids)
    if all_ids_to_destroy:
        logging.info(f"Tentative de destruction de {len(all_ids_to_destroy)} acteurs PNJ...")
        batch_destroy = [carla.command.DestroyActor(actor_id) for actor_id in all_ids_to_destroy]
        try:
            results = client.apply_batch_sync(batch_destroy, True) # True = ignorer les erreurs
            destroyed_count = sum(1 for res in results if not res.error)
            error_count = len(results) - destroyed_count
            logging.info(f"Destruction PNJ: {destroyed_count} succès, {error_count} erreurs (probablement déjà détruits).")
        except Exception as e_destroy:
            logging.error(f"Erreur lors de la destruction des acteurs PNJ en batch: {e_destroy}")

    logging.info("Nettoyage de l'environnement PNJ terminé.")

# Exemple d'utilisation si ce fichier est exécuté directement (pour test)
if __name__ == '__main__':
    client = None
    original_settings = None
    world = None
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world()
        original_settings = world.get_settings()

        # Configurer le mode synchrone pour le test
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        world.apply_settings(settings)

        # Tester la configuration
        vehicle_ids, walker_ids, controller_ids = setup_carla_environment(
            client, world, num_vehicles=20, num_pedestrians=30, fog_density=0.2
        )

        print(f"Environnement configuré pour test. Véhicules: {len(vehicle_ids)}, Piétons: {len(walker_ids)}")
        print("Attente de 10 secondes...")
        for _ in range(100): # 10 sec * 10 ticks/sec
            world.tick()
            time.sleep(0.01) # Petit délai pour ne pas surcharger

    except Exception as e:
        logging.error(f"Erreur dans le test direct: {e}", exc_info=True)
    finally:
        if world and original_settings:
            world.apply_settings(original_settings) # Rétablir les settings
        if client and 'vehicle_ids' in locals(): # Si la configuration a été tentée
            # Nettoyer les acteurs spawnés pendant le test
            cleanup_carla_environment(client, vehicle_ids, walker_ids, controller_ids)
        print("Test direct terminé.")