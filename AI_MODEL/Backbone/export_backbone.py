# export_backbone.py
import torch
import torch.nn as nn # Nécessaire pour que les classes du modèle se chargent
import os
import argparse
import warnings

# --- Importer les définitions du modèle ---
try:
    # Assumer que backbone_module.py est accessible
    from backbone_module import (
        MultiTaskNet,
        regnet_1600M_config,
        # Importer d'autres composants si MultiTaskNet en dépend directement
        # (normalement non si bien encapsulé)
    )
    print("Import des modules du backbone réussi.")
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer depuis backbone_module: {e}")
    print("Assurez-vous que le fichier backbone_module.py est accessible.")
    exit()
# Supprimer les warnings spécifiques à l'instanciation si nécessaire
warnings.filterwarnings("ignore", message=".*contiguous.*") # Ignore les warnings BiFPN sur les niveaux P

def export_model(trained_ckpt_path, save_path, args):
    """Charge un checkpoint entraîné et resauvegarde le state_dict."""

    if not os.path.exists(trained_ckpt_path):
        print(f"Erreur: Checkpoint entraîné non trouvé à {trained_ckpt_path}")
        return

    print(f"Chargement du state_dict depuis: {trained_ckpt_path}")
    # Charger sur CPU pour éviter les problèmes de device
    state_dict = torch.load(trained_ckpt_path, map_location='cpu')

    # --- Recréer la configuration exacte du modèle entraîné ---
    # !!! IMPORTANT: Ces paramètres DOIVENT correspondre à ceux utilisés pour l'entraînement !!!
    HEAD_OUTPUT_RES = (args.out_height, args.out_width) # Ex: (480, 720)
    NUM_CLASSES_SEGMENTATION = args.num_seg_classes

    backbone_cfg = regnet_1600M_config # Assumer que c'est la bonne config
    # Config Neck doit correspondre à l'entraînement
    neck_cfg = dict(fpn_channels=128, num_repeats=4, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU)
    # Déduire les niveaux FPN produits pour configurer les têtes
    stage_map={'s1':2,'s2':3,'s3':4,'s4':5};
    produced_fpn_levels=sorted([f"P{stage_map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))

    segmentation_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_RES, levels_to_use=produced_fpn_levels)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_RES, levels_to_use=produced_fpn_levels)

    # --- Instancier le modèle ---
    print("Instanciation de l'architecture MultiTaskNet...")
    # Pas besoin de poids pré-entraînés ImageNet ici, on charge le checkpoint complet
    model = MultiTaskNet(
        backbone_config=backbone_cfg,
        backbone_pretrained=False,
        backbone_ckpt_path=None,
        neck_config=neck_cfg,
        segmentation_config=segmentation_cfg,
        depth_config=depth_cfg
    )

    # --- Charger le state_dict dans le modèle ---
    try:
        # Charger les poids
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        print("State_dict chargé avec succès.")
        if missing_keys:
            print(f"  Clés manquantes (Warning): {missing_keys}")
        if unexpected_keys:
            print(f"  Clés inattendues (Warning): {unexpected_keys}") # Peut arriver si ckpt contient optimizer etc.
        model.eval() # Mettre en mode évaluation
    except Exception as e:
        print(f"Erreur lors du chargement du state_dict: {e}")
        print("Vérifiez que l'architecture du modèle correspond au checkpoint.")
        return

    # --- Ré-sauvegarder le state_dict (format standard) ---
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Modèle (state_dict) exporté/resauvegardé vers : {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du state_dict: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and Re-Save Trained MultiTaskNet Model')
    parser.add_argument('--trained_ckpt_path', type=str, required=True, help='Path to the TRAINED MultiTaskNet checkpoint (.pth)')
    parser.add_argument('--save_path', type=str, default='./multitask_exported.pth', help='Path to save the exported state_dict')

    # Arguments pour recréer l'architecture EXACTE du modèle entraîné
    parser.add_argument('--out_height', type=int, default=480, help='Head output height used during training')
    parser.add_argument('--out_width', type=int, default=720, help='Head output width used during training')
    parser.add_argument('--num_seg_classes', type=int, default=8, help='Number of segmentation classes used during training')
    # Ajouter d'autres arguments si votre __init__ de MultiTaskNet ou ses composants en dépendent (ex: fpn_channels, num_repeats...)

    args = parser.parse_args()
    export_model(args.trained_ckpt_path, args.save_path, args)