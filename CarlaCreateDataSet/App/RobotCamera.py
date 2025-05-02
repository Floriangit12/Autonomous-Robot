import carla
from Camera import *
class RobotCamera:
    def __init__(self, world, transform, name="robot_cam", common_res=[640, 480], common_fov=90.0, common_fps=15.0):
        """
        Crée un robot simulé avec 4 caméras attachées sur le dessus.

        :param world: Objet carla.World.
        :param transform: carla.Transform initial du centre du robot.
        :param name: Nom de base pour le robot et ses composants.
        :param common_res: Résolution [width, height] commune aux 4 caméras.
        :param common_fov: FOV commun aux 4 caméras.
        :param common_fps: FPS commun aux 4 caméras.
        """
        self.world = world
        self.name = name
        self.robot_actor = None
        self.cameras = {} # Dictionnaire pour stocker les CameraSensor par nom

        # Dimensions approximatives du robot (pour le placement des caméras)
        robot_x_half = 0.10 # Moitié de 20cm
        robot_y_half = 0.10
        robot_z_top = 0.15 # Hauteur où placer les caméras (un peu au-dessus de 20cm)

        # 1. Créer le corps du robot (un cube statique comme représentation)
        try:
            bp_lib = self.world.get_blueprint_library()
            # Utiliser une petite boîte comme corps. Vous pourriez chercher d'autres props.
            robot_bp = bp_lib.find('static.prop.box01')
            # Malheureusement, le scaling direct via set_attribute n'est pas standard
            # Il faudrait choisir un prop de la bonne taille ou accepter sa taille par défaut.
            self.robot_actor = self.world.try_spawn_actor(robot_bp, transform)

            if self.robot_actor is None:
                raise ValueError(f"Impossible de spawner le corps du robot '{self.name}' à {transform}")
            print(f"Corps du robot '{self.name}' créé (ID: {self.robot_actor.id})")

        except Exception as e:
            print(f"Erreur lors de la création du corps du robot: {e}")
            self.destroy() # Nettoyer si échec partiel
            raise # Propager l'erreur

        # 2. Définir les positions relatives et rotations pour les 4 caméras sur le dessus
        #    Les transformations sont relatives au robot_actor !
        #    X: avant, Y: gauche, Z: haut (dans le repère de l'acteur parent)
        cam_locations = {
            'front': carla.Location(x=robot_x_half, y=0,              z=robot_z_top),
            'back':  carla.Location(x=-robot_x_half,y=0,              z=robot_z_top),
            'left':  carla.Location(x=0,              y=-robot_y_half,z=robot_z_top),
            'right': carla.Location(x=0,              y=robot_y_half, z=robot_z_top),
        }
        cam_rotations = {
            'front': carla.Rotation(pitch=-10, yaw=0),    # Regarde vers l'avant, légèrement vers le bas
            'back':  carla.Rotation(pitch=-10, yaw=180),  # Regarde vers l'arrière
            'left':  carla.Rotation(pitch=-10, yaw=-90),  # Regarde vers la gauche
            'right': carla.Rotation(pitch=-10, yaw=90),   # Regarde vers la droite
        }
        cam_names = ['front', 'back', 'left', 'right']

        # 3. Créer et attacher les 4 instances de CameraSensor
        spawn_failed = False
        for cam_id in cam_names:
            cam_name = f"{self.name}_cam_{cam_id}"
            try:
                camera_sensor = CameraSensor(
                    world=self.world,
                    cameraName=cam_name,
                    cameraResolution=common_res,
                    cameraPosition=cam_locations[cam_id],
                    cameraRotation=cam_rotations[cam_id],
                    cameraFOV=common_fov,
                    frame_rate=common_fps,
                    attach_to=self.robot_actor, # <-- Point clé : attacher au robot
                    attachment_type=carla.AttachmentType.Rigid
                )
                if camera_sensor.sensor_actor is None:
                    print(f"Échec critique: Impossible de créer la caméra attachée '{cam_name}'.")
                    spawn_failed = True
                    break # Arrêter si une caméra échoue
                self.cameras[cam_id] = camera_sensor
            except Exception as e:
                 print(f"Erreur lors de la création de la caméra '{cam_name}': {e}")
                 spawn_failed = True
                 break

        if spawn_failed:
             print("Nettoyage suite à un échec de création de caméra.")
             self.destroy() # Nettoyer toutes les caméras créées et le robot
             raise RuntimeError("Échec de l'initialisation de RobotCamera.")


    def set_transform(self, new_transform):
        """Déplace le robot (et donc les caméras attachées) à une nouvelle transformation."""
        if self.robot_actor:
            self.robot_actor.set_transform(new_transform)
            # print(f"Robot '{self.name}' déplacé vers {new_transform.location}")
        else:
            print(f"Avertissement: Robot '{self.name}' non trouvé pour le déplacement.")

    def get_snapshots(self):
        """Prend une 'photo' de chaque caméra et retourne un dictionnaire d'images numpy."""
        snapshots = {}
        for cam_id, camera_sensor in self.cameras.items():
            snapshots[cam_id] = camera_sensor.get_latest_image_data() # Peut être None
        return snapshots

    def destroy(self):
        """Détruit toutes les caméras et le corps du robot."""
        print(f"Destruction de RobotCamera '{self.name}'...")
        # Détruire les caméras d'abord
        for cam_id, camera_sensor in self.cameras.items():
             if camera_sensor: # Vérifier si l'objet existe
                 camera_sensor.destroy()
        self.cameras = {} # Vider le dictionnaire

        # Détruire le corps du robot ensuite
        if self.robot_actor:
            destroyed = self.robot_actor.destroy()
            if destroyed:
                 # print(f"Corps du robot '{self.name}' détruit.")
                 self.robot_actor = None
            # else: # Peut arriver si déjà détruit
                 # print(f"Échec destruction corps du robot '{self.name}'.")