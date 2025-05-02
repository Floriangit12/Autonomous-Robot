# Installer numpy si nécessaire: pip install numpy
import numpy as np

# (Le reste des imports : carla, time)
import carla
import time

# --- Fonction Callback Externe (inchangée) ---
def camera_callback(image, camera_name):
    print(f'Image reçue de {camera_name} au timestamp {image.timestamp} (Frame: {image.frame})')
    # Pas de traitement ici, car il sera géré dans la classe ou pour l'affichage

class CameraSensor:
    def __init__(self, world, cameraName, cameraResolution, cameraPosition, cameraRotation, cameraFOV, frame_rate=20.0, attach_to=None, attachment_type=carla.AttachmentType.Rigid):
        """ Version Modifiée:
            - Ajout de 'attach_to' et 'attachment_type' pour l'attachement.
            - Stocke la dernière image reçue.
            - Configure un callback interne par défaut pour stocker l'image.
        """
        self.world = world
        self.name = cameraName
        self.resolution_width = cameraResolution[0]
        self.resolution_height = cameraResolution[1]
        self.fov = float(cameraFOV)
        self.frame_rate = float(frame_rate)
        self.sensor_actor = None
        self._latest_image = None # Pour stocker la dernière image CARLA reçue
        self._image_lock = threading.Lock() # Sécurité si accès multi-threadé (optionnel ici)

        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.resolution_width))
        camera_bp.set_attribute('image_size_y', str(self.resolution_height))
        camera_bp.set_attribute('fov', str(self.fov))
        sensor_tick = str(1.0 / self.frame_rate)
        camera_bp.set_attribute('sensor_tick', sensor_tick)

        # La transformation est relative si 'attach_to' est fourni
        transform = carla.Transform(cameraPosition, cameraRotation)

        self.sensor_actor = self.world.try_spawn_actor(camera_bp, transform, attach_to=attach_to, attachment_type=attachment_type)

        if self.sensor_actor is None:
            parent_id = attach_to.id if attach_to else "world"
            print(f"Erreur : N'a pas pu spawner la caméra '{self.name}' attachée à {parent_id} @ {transform}.")
            return # Important de sortir si ça échoue

        print(f"Caméra '{self.name}' créée (ID: {self.sensor_actor.id}), attachée à {parent_id if attach_to else 'world'}")

        # Configuration du callback interne pour stocker l'image
        self.sensor_actor.listen(self._save_latest_image)


    def _save_latest_image(self, image):
        """Callback interne pour stocker l'image brute."""
        # print(f"Debug: Image reçue par {self.name}") # Décommenter pour débugger le flux
        with self._image_lock:
            self._latest_image = image

    def get_latest_image_data(self):
        """
        Retourne les données de la dernière image sous forme de tableau numpy (BGRA).
        Retourne None si aucune image n'a encore été reçue.
        """
        with self._image_lock:
            if self._latest_image is None:
                return None
            # Les données brutes sont souvent BGRA
            image_data = np.frombuffer(self._latest_image.raw_data, dtype=np.uint8)
            # Remodeler en H x W x 4 (BGRA)
            image_reshaped = np.reshape(image_data, (self._latest_image.height, self._latest_image.width, 4))
            return image_reshaped

    # On garde la possibilité de définir un callback externe si nécessaire,
    # mais il écrasera le callback interne _save_latest_image.
    def set_external_callback(self, external_callback_function, *args, **kwargs):
        if self.sensor_actor:
            callback_wrapper = lambda image: external_callback_function(image, self.name, *args, **kwargs)
            self.sensor_actor.listen(callback_wrapper)
            print(f"Callback externe '{external_callback_function.__name__}' défini pour '{self.name}' (remplace le stockage interne).")
        else:
            print(f"Impossible de définir le callback pour '{self.name}', l'acteur n'existe pas.")


    def destroy(self):
        if self.sensor_actor is not None:
            print(f"Destruction de la caméra '{self.name}' (ID: {self.sensor_actor.id})")
            if self.sensor_actor.is_listening:
                 self.sensor_actor.stop()
            destroyed = self.sensor_actor.destroy()
            # (...) reste du code de destroy
            if destroyed:
                # print(f"Caméra '{self.name}' détruite.") # Un peu verbeux
                self.sensor_actor = None

# --- Fin de CameraSensor modifié ---

import threading # Pour le Lock