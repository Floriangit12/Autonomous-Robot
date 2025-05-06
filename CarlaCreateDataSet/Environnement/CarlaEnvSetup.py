import carla
import random
import time
import logging

# Configuration du logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

class CarlaEnvSetup:
    """
    Une classe pour configurer l'environnement de simulation CARLA.
    Permet de définir la météo et de faire apparaître des véhicules et des piétons.
    Assure le nettoyage des acteurs créés.
    """

    def __init__(self, host='localhost', port=2000, timeout=10.0, world_name=None):
        """
        Initialise la connexion au client CARLA et récupère le monde.

        Args:
            host (str): Adresse IP de l'hôte du serveur CARLA.
            port (int): Port TCP du serveur CARLA.
            timeout (float): Temps maximum d'attente pour la connexion (secondes).
            world_name (str, optional): Nom de la carte à charger.
                                        Si None, utilise la carte actuelle.
        """
        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle_list = []
        self.walker_list = []
        self.controller_list = []
        self.original_weather = None

        try:
            logging.info(f"Connexion au serveur CARLA sur {host}:{port}")
            self.client = carla.Client(host, port)
            self.client.set_timeout(timeout)

            if world_name:
                logging.info(f"Chargement de la carte : {world_name}")
                self.world = self.client.load_world(world_name)
            else:
                logging.info("Récupération de la carte actuelle.")
                self.world = self.client.get_world()

            self.blueprint_library = self.world.get_blueprint_library()
            self.original_weather = self.world.get_weather()
            logging.info("Connexion et récupération du monde réussies.")

        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation de CarlaEnvSetup: {e}")
            # Propager l'erreur pour que l'utilisateur sache que l'initialisation a échoué
            raise

    def set_weather(self, preset=None, **kwargs):
        """
        Définit les conditions météorologiques dans la simulation.

        Args:
            preset (carla.WeatherParameters, optional): Un preset météo CARLA
                (ex: carla.WeatherParameters.ClearNoon, carla.WeatherParameters.HardRainSunset).
            **kwargs: Paramètres météo individuels si preset n'est pas utilisé.
                      Ex: cloudiness=10.0, precipitation=0.0, wind_intensity=0.35,
                          sun_azimuth_angle=0.0, sun_altitude_angle=75.0, fog_density=0.0, etc.
        """
        if preset is not None and isinstance(preset, carla.WeatherParameters):
            logging.info(f"Application du preset météo : {preset.__class__.__name__}")
            self.world.set_weather(preset)
        else:
            # Utilise les paramètres actuels comme base et modifie ceux fournis
            weather = self.world.get_weather()
            logging.info(f"Modification des paramètres météo actuels : {kwargs}")
            for key, value in kwargs.items():
                if hasattr(weather, key):
                    setattr(weather, key, value)
                else:
                    logging.warning(f"Paramètre météo inconnu : {key}")
            self.world.set_weather(weather)
        logging.info("Conditions météorologiques mises à jour.")

    def spawn_vehicles(self, num_vehicles, safe=True, autopilot=True, vehicle_filter='vehicle.*'):
        """
        Fait apparaître un nombre donné de véhicules à des emplacements aléatoires.

        Args:
            num_vehicles (int): Nombre de véhicules à faire apparaître.
            safe (bool): Si True, utilise try_spawn_actor pour éviter les collisions au spawn.
            autopilot (bool): Si True, active le pilote automatique pour les véhicules spawnés.
            vehicle_filter (str): Filtre pour les blueprints de véhicules.
        """
        logging.info(f"Tentative de spawn de {num_vehicles} véhicules...")
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("Impossible de récupérer les points de spawn de la carte.")
            return

        if num_vehicles > len(spawn_points):
            logging.warning(f"Nombre de véhicules demandé ({num_vehicles}) "
                            f"supérieur au nombre de points de spawn ({len(spawn_points)}). "
                            f"Limitation à {len(spawn_points)}.")
            num_vehicles = len(spawn_points)

        blueprints = self.blueprint_library.filter(vehicle_filter)
        # Filtrer les blueprints qui pourraient causer problème (ex: vélos sans physique correcte parfois)
        blueprints = [bp for bp in blueprints if int(bp.get_attribute('number_of_wheels')) >= 4]
        if not blueprints:
             logging.error(f"Aucun blueprint trouvé pour le filtre '{vehicle_filter}' avec >= 4 roues.")
             return

        # Mélanger les points de spawn pour éviter de toujours utiliser les mêmes
        random.shuffle(spawn_points)
        spawn_count = 0

        for i in range(num_vehicles):
            blueprint = random.choice(blueprints)
            transform = spawn_points[i] # Utilise les points mélangés

            if safe:
                actor = self.world.try_spawn_actor(blueprint, transform)
            else:
                # Attention : peut générer des erreurs si l'emplacement est occupé
                actor = self.world.spawn_actor(blueprint, transform)

            if actor is not None:
                self.vehicle_list.append(actor)
                if autopilot:
                    actor.set_autopilot(True)
                spawn_count += 1
                # logging.debug(f"Véhicule {actor.id} ({actor.type_id}) spawné à {transform.location}")
            else:
                logging.warning(f"Échec du spawn du véhicule {i+1}/{num_vehicles} à {transform.location}. Emplacement peut-être occupé.")

        logging.info(f"{spawn_count}/{num_vehicles} véhicules spawnés avec succès.")


    def spawn_pedestrians(self, num_pedestrians, safe=True):
        """
        Fait apparaître un nombre donné de piétons et leur contrôleur IA.

        Args:
            num_pedestrians (int): Nombre de piétons à faire apparaître.
            safe (bool): Utilise try_spawn_actor pour les piétons.
        """
        logging.info(f"Tentative de spawn de {num_pedestrians} piétons...")

        # Blueprints pour les piétons
        blueprints_walkers = self.blueprint_library.filter('walker.pedestrian.*')
        if not blueprints_walkers:
             logging.error("Aucun blueprint de piéton trouvé.")
             return

        # Blueprint pour le contrôleur IA des piétons
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        if not walker_controller_bp:
            logging.error("Blueprint 'controller.ai.walker' non trouvé.")
            return

        spawn_count = 0
        walker_spawned_ids = []

        for _ in range(num_pedestrians):
            # Choisir un blueprint de piéton aléatoire
            walker_bp = random.choice(blueprints_walkers)
            # Les piétons ne peuvent pas avoir d'attribut 'role_name' défini
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')

            # Trouver un point de spawn aléatoire sur le trottoir/zone navigable
            spawn_location = self.world.get_random_location_from_navigation()
            if spawn_location is None:
                logging.warning("Impossible de trouver un point de spawn de navigation aléatoire pour le piéton.")
                continue

            spawn_transform = carla.Transform(spawn_location)

            # Spawner le piéton
            if safe:
                walker_actor = self.world.try_spawn_actor(walker_bp, spawn_transform)
            else:
                walker_actor = self.world.spawn_actor(walker_bp, spawn_transform)

            if walker_actor is not None:
                # Spawner le contrôleur pour ce piéton
                controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker_actor)
                if controller is not None:
                    # Démarrer le contrôleur et lui donner une destination
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    # Optionnel: définir une vitesse de marche
                    # controller.set_max_speed(1.4 + random.random() * 0.5) # Vitesse entre 1.4 et 1.9 m/s

                    self.walker_list.append(walker_actor)
                    self.controller_list.append(controller)
                    walker_spawned_ids.append(walker_actor.id) # Pour debug
                    spawn_count += 1
                else:
                    logging.warning(f"Échec du spawn du contrôleur pour le piéton {walker_actor.id}.")
                    # Détruire le piéton si son contrôleur n'a pas pu spawner
                    walker_actor.destroy()
            else:
                # Le spawn du piéton lui-même a peut-être échoué
                # logging.debug(f"Échec du spawn du piéton à {spawn_transform.location}")
                pass # Pas besoin de warning si try_spawn échoue silencieusement

        logging.info(f"{spawn_count}/{num_pedestrians} piétons (et leurs contrôleurs) spawnés avec succès.")
        # logging.debug(f"IDs des piétons spawnés : {walker_spawned_ids}")


    def cleanup(self, restore_weather=False):
        """
        Nettoie tous les acteurs (véhicules, piétons, contrôleurs) créés par cette instance.
        """
        logging.info("Nettoyage des acteurs spawnés...")

        # Arrêter les contrôleurs IA
        if self.controller_list:
            logging.info(f"Arrêt de {len(self.controller_list)} contrôleurs IA...")
            for controller in self.controller_list:
                try:
                    if controller and controller.is_alive:
                        controller.stop()
                except RuntimeError as e:
                    logging.warning(f"Erreur à l'arrêt du contrôleur {getattr(controller, 'id', 'inconnu')}: {e}")
            # Attente courte pour laisser le temps aux commandes de passer
            time.sleep(0.5)


        # Détruire les acteurs par batch pour l'efficacité
        batches = []
        if self.controller_list:
            batches.append(carla.command.DestroyActorList(self.controller_list))
            logging.info(f"Préparation de la destruction de {len(self.controller_list)} contrôleurs.")
        if self.walker_list:
            batches.append(carla.command.DestroyActorList(self.walker_list))
            logging.info(f"Préparation de la destruction de {len(self.walker_list)} piétons.")
        if self.vehicle_list:
            batches.append(carla.command.DestroyActorList(self.vehicle_list))
            logging.info(f"Préparation de la destruction de {len(self.vehicle_list)} véhicules.")

        if batches and self.client:
            try:
                logging.info("Exécution de la destruction par batch...")
                results = self.client.apply_batch_sync(batches, True) # True pour la synchronisation atomique
                # Vérifier les erreurs (facultatif)
                error_count = 0
                for i, result in enumerate(results):
                    if result.has_error():
                         error_count += 1
                         logging.warning(f"Erreur lors de la destruction (Batch {i}): {result.error}")
                if error_count == 0:
                    logging.info("Destruction par batch réussie.")
                else:
                     logging.warning(f"{error_count} erreurs lors de la destruction par batch.")

            except Exception as e:
                logging.error(f"Erreur lors de l'exécution de apply_batch_sync: {e}")
                # Tentative de destruction individuelle en dernier recours
                logging.warning("Tentative de destruction individuelle...")
                all_actors = self.controller_list + self.walker_list + self.vehicle_list
                for actor in all_actors:
                    try:
                         if actor and actor.is_alive:
                             actor.destroy()
                    except RuntimeError as e_destroy:
                         logging.warning(f"Erreur à la destruction de l'acteur {getattr(actor, 'id', 'inconnu')}: {e_destroy}")

        # Vider les listes locales
        self.vehicle_list = []
        self.walker_list = []
        self.controller_list = []

        # Restaurer la météo originale si demandé
        if restore_weather and self.original_weather:
            logging.info("Restauration de la météo originale.")
            self.world.set_weather(self.original_weather)

        logging.info("Nettoyage terminé.")

    # --- Méthodes pour utilisation comme context manager ---
    def __enter__(self):
        """Permet d'utiliser 'with CarlaEnvSetup(...) as setup:'"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Appelé automatiquement à la sortie du bloc 'with', assure le nettoyage."""
        self.cleanup()
        # Si une exception s'est produite dans le bloc 'with', elle est passée ici.
        # On ne la supprime pas (on retourne None ou False implicitement), elle sera propagée.
        if exc_type:
             logging.error(f"Une exception s'est produite dans le bloc 'with': {exc_val}")


# --- Exemple d'utilisation ---
if __name__ == '__main__':

    # Utilisation simple
    # setup_instance = None
    # try:
    #     setup_instance = CarlaEnvSetup() # Utilise la carte actuelle
    #     # setup_instance = CarlaEnvSetup(world_name='Town05') # Charge une carte spécifique

    #     # Définir une météo (exemple: après-midi pluvieux)
    #     setup_instance.set_weather(
    #         cloudiness=80,
    #         precipitation=60,
    #         precipitation_deposits=50,
    #         sun_altitude_angle=20,
    #         wind_intensity=0.5,
    #         fog_density = 5.0
    #     )

    #     # Faire apparaître des véhicules et des piétons
    #     setup_instance.spawn_vehicles(num_vehicles=50, autopilot=True)
    #     setup_instance.spawn_pedestrians(num_pedestrians=100)

    #     # Laisser la simulation tourner pendant un moment
    #     logging.info("Simulation configurée. Laisser tourner pendant 60 secondes...")
    #     time.sleep(60)

    # except Exception as e:
    #     logging.error(f"Une erreur s'est produite dans l'exécution principale: {e}")

    # finally:
    #     if setup_instance:
    #         setup_instance.cleanup(restore_weather=True) # Nettoie et restaure la météo

    # Utilisation avec le context manager (préférable)
    try:
        # Le 'with' garantit que __exit__ (et donc cleanup) sera appelé
        # même si une erreur se produit à l'intérieur du bloc.
        # with CarlaEnvSetup(world_name='Town03') as setup: # Charge Town03
        with CarlaEnvSetup() as setup: # Utilise la carte actuelle
             # Définir une météo claire
            setup.set_weather(preset=carla.WeatherParameters.ClearNoon)

            # Spawner les acteurs
            setup.spawn_vehicles(num_vehicles=30, autopilot=True)
            setup.spawn_pedestrians(num_pedestrians=50)

            logging.info("Simulation configurée avec context manager. Laisser tourner 30 secondes...")
            time.sleep(30)
            logging.info("Fin de la pause.")
            # Le nettoyage sera appelé automatiquement ici via __exit__

    except Exception as e:
         logging.error(f"Une erreur s'est produite dans l'exécution principale avec context manager: {e}")

    logging.info("Script principal terminé.")
