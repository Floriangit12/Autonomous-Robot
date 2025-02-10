import torch
from torchinfo import summary
from model import AutonomousDrivingModel  # Assurez-vous que le chemin d'importation est correct

# Création d'une instance de votre modèle
model = AutonomousDrivingModel()

# Préparez les tenseurs factices correspondant aux différentes entrées :
# - Deux caméras : dimensions (B, 3, 224, 224)
# - GPS : dimensions (B, 3) (par exemple : latitude, longitude, altitude ou vitesse)
# - Route : dimensions (B, 10) (par exemple, un vecteur de caractéristiques de la route)
# - Dynamics : dimensions (B, 5) (par exemple, vitesse, accélération, etc.)
input_data = [
    torch.randn(1, 3, 224, 224),  # cam1
    torch.randn(1, 3, 224, 224),  # cam2
    torch.randn(1, 3),            # gps
    torch.randn(1, 10),           # route_seq
    torch.randn(1, 5)             # dynamics
]

# Afficher le résumé du modèle
summary(model, input_data=input_data, col_names=["input_size", "output_size", "num_params", "kernel_size"], depth=4)
