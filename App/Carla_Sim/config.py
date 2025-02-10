# config.py
import carla

# Types de voies autorisées : ici, les trottoirs et les pistes cyclables.
ALLOWED_LANE_TYPES = {carla.LaneType.Sidewalk, carla.LaneType.Biking}

# Paramètres du véhicule et de l'itinéraire
TARGET_SPEED = 10.0          # Vitesse cible en m/s
NUM_ROUTE_STEPS = 50         # Nombre de waypoints à générer pour l'itinéraire
STEP_DISTANCE = 2.0          # Distance (en m) entre deux waypoints
TICK_SLEEP = 0.05            # Intervalle de temps entre deux ticks dans la boucle de contrôle

# Définition des zones de signalisation (coordonnées fictives, à adapter selon la carte)
STOP_ZONES = [
    {'location': carla.Location(x=230, y=195, z=0), 'radius': 10.0, 'duration': 3},  # zone de panneau stop
]

PED_CROSSINGS = [
    {'location': carla.Location(x=250, y=210, z=0), 'radius': 10.0, 'duration': 3},  # zone de passage piétons
]
