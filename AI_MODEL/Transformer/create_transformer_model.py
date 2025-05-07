# load_full_pipeline.py
import torch
import torch.nn as nn # Nécessaire
import os
import argparse
import warnings

# --- Importer les définitions des modèles ---
try:
    from backbone_module import MultiTaskNet, regnet_1600M_config
    from transformer_module import ViTOccupancyPredictor, print_vit_summary
    print("Import des modules backbone et transformer réussi.")
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer les modules: {e}")
    print("Assurez-vous que backbone_module.py et transformer_module.py sont accessibles.")
    exit()
warnings.filterwarnings("ignore", message=".*contiguous.*")

def setup_pipeline(args):
    """Charge le MultiTaskNet pré-entraîné et instancie le ViT."""

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # --- 1. Charger le Générateur de Features (MultiTaskNet pré-entraîné) ---
    if not os.path.exists(args.multitask_ckpt_path):
        print(f"ERREUR: Checkpoint MultiTaskNet pré-entraîné non trouvé à {args.multitask_ckpt_path}")
        return None, None

    print("Instantiating PRE-TRAINED MultiTaskNet model...")
    # Recréer la config exacte du modèle pré-entraîné
    HEAD_OUTPUT_RES = (args.feature_height, args.feature_width) # Sortie MultiTaskNet -> Entrée ViT
    NUM_CLASSES_SEGMENTATION = args.num_seg_classes
    backbone_cfg = regnet_1600M_config
    neck_cfg = dict(fpn_channels=128, num_repeats=4, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU)
    stage_map={'s1':2,'s2':3,'s3':4,'s4':5}; prod_fpn_lvls=sorted([f"P{stage_map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))
    segmentation_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_RES, levels_to_use=prod_fpn_lvls)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_RES, levels_to_use=prod_fpn_lvls)

    feature_generator_model = MultiTaskNet(backbone_cfg, False, None, neck_cfg, segmentation_cfg, depth_cfg)

    print(f"Loading MultiTaskNet weights from: {args.multitask_ckpt_path}")
    try:
        state_dict = torch.load(args.multitask_ckpt_path, map_location='cpu')
        feature_generator_model.load_state_dict(state_dict)
        feature_generator_model.eval() # Mode évaluation
        feature_generator_model.to(device)
         # Geler les poids si on ne veut pas le fine-tuner
        if args.freeze_backbone:
            for param in feature_generator_model.parameters():
                param.requires_grad = False
            print("MultiTaskNet weights frozen.")
        print("MultiTaskNet loaded, set to eval mode, and moved to device.")
    except Exception as e:
        print(f"Erreur lors du chargement du checkpoint MultiTaskNet: {e}")
        return None, None

    # --- 2. Instancier le ViT ---
    # Les canaux d'entrée dépendent des sorties du MultiTaskNet
    vit_input_channels = {'segmentation': NUM_CLASSES_SEGMENTATION, 'depth': 1}
    OUTPUT_GRID_SIZE = tuple(args.output_grid_size)

    print("\nInstantiating ViTOccupancyPredictor model...")
    vit_model = ViTOccupancyPredictor(
        input_feature_channels=vit_input_channels,
        img_size=HEAD_OUTPUT_RES, # Entrée ViT = Sortie Têtes MultiTaskNet
        patch_size=args.vit_patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_heads,
        output_grid_size=OUTPUT_GRID_SIZE
    )
    vit_model.eval() # Mettre en eval par défaut, passer en train si entraînement
    vit_model.to(device)
    print("ViTOccupancyPredictor instantiated and moved to device.")

    return feature_generator_model, vit_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load ViT with Pre-trained Feature Generator')
    # Paths
    parser.add_argument('--multitask_ckpt_path', type=str, required=True, help='Path to the PRE-TRAINED MultiTaskNet checkpoint (.pth)')

    # Config MultiTaskNet (doit correspondre au ckpt chargé)
    parser.add_argument('--feature_height', type=int, default=480, help='Feature map height (MultiTaskNet output -> ViT input)')
    parser.add_argument('--feature_width', type=int, default=720, help='Feature map width (MultiTaskNet output -> ViT input)')
    parser.add_argument('--num_seg_classes', type=int, default=8, help='Number of segmentation classes in MultiTaskNet')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze weights of the loaded MultiTaskNet')

    # Config ViT
    parser.add_argument('--vit_patch_size', type=int, default=24, help='Patch size for ViT (must divide feature H/W)')
    parser.add_argument('--vit_embed_dim', type=int, default=768, help='Embedding dimension for ViT')
    parser.add_argument('--vit_depth', type=int, default=12, help='Number of Transformer layers for ViT')
    parser.add_argument('--vit_heads', type=int, default=12, help='Number of Attention heads for ViT')
    parser.add_argument('--output_grid_size', type=int, nargs=3, default=[100, 100, 20], help='Target occupancy grid size (H W Z)')

    # System
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    # --- Charger et Afficher les modèles ---
    feature_generator, vit_predictor = setup_pipeline(args)

    if feature_generator and vit_predictor:
        print("\n--- Pipeline Setup Successful ---")

        # Afficher les résumés des paramètres
        if BACKBONE_AVAILABLE and 'print_model_summary' in globals():
             print("\n--- MultiTaskNet Summary ---")
             print_model_summary(feature_generator) # Assurez-vous que la fonction existe

        if 'print_vit_summary' in globals():
             print("\n--- ViT Summary ---")
             print_vit_summary(vit_predictor)

        # Ici, vous pourriez lancer une inférence de test, ou passer ces modèles
        # à une boucle d'entraînement pour le ViT.
        # Exemple d'inférence rapide (nécessite une image d'entrée)
        print("\n--- Testing inference (optional) ---")
        try:
             dummy_image = torch.randn(1, 3, args.feature_height, args.feature_width).to(next(feature_generator.parameters()).device) # Utiliser la taille d'entrée ViT comme si elle venait du dataloader
             with torch.no_grad():
                  features = feature_generator(dummy_image)
                  # S'assurer que les features existent avant de les passer au ViT
                  if 'segmentation' in features and 'depth' in features:
                      vit_input = {'segmentation': features['segmentation'], 'depth': features['depth']}
                      output = vit_predictor(vit_input)
                      print("Inference test successful. Occupancy shape:", output['occupancy'].shape)
                  else:
                      print("Inference test skipped: Missing features from feature_generator.")
        except Exception as e:
             print(f"Inference test failed: {e}")

    else:
        print("\n--- Pipeline Setup Failed ---")