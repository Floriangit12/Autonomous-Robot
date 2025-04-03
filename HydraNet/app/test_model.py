# /app/test_model.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.models as models # Gardé pour ResNet
from PIL import Image
import numpy as np
import os
import ssl
import random
import matplotlib.pyplot as plt
import traceback

# Importer depuis model.py
try:
    # On a besoin de DepthSegNet et RandomDepthDataset
    from model import DepthSegNet, RandomDepthDataset
    # BDD100KSegmentation n'est plus nécessaire dans CE fichier de test
except ImportError as e:
    print(f"Erreur d'import depuis model.py: {e}")
    print("Vérifiez que model.py existe et contient DepthSegNet et RandomDepthDataset.")
    exit()

# --- Configuration SSL ---
try: _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context; print("Warn: Contexte SSL non vérifié.")

# --- Constantes et Utilitaires pour Visualisation ---
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
bdd100k_colors = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0) ]
ignore_index_global = 255; num_classes_display = 19

def apply_color_map(mask, color_map, num_classes=num_classes_display, ignore_idx=ignore_index_global):
    """Applique une palette de couleurs à un masque de segmentation 2D."""
    if isinstance(mask, torch.Tensor): mask = mask.squeeze().cpu().numpy().astype(int)
    elif isinstance(mask, Image.Image): mask_np = np.array(mask); mask = mask_np.astype(int) if mask_np.ndim == 2 else None
    if mask is None: return np.zeros((224, 224, 3), dtype=np.uint8) # Placeholder
    h, w = mask.shape; colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    processed_map = np.array(color_map[:num_classes], dtype=np.uint8)
    ignore_color = color_map[num_classes] if ignore_idx is not None and len(color_map) > num_classes else [0, 0, 0]
    valid_idx_mask = (mask >= 0) & (mask < num_classes)
    ignore_idx_mask = (mask == ignore_idx) if ignore_idx is not None else np.zeros_like(mask, dtype=bool)
    valid_indices = mask[valid_idx_mask]; colored_mask[valid_idx_mask] = processed_map[valid_indices]
    if ignore_idx is not None: colored_mask[ignore_idx_mask] = ignore_color
    return colored_mask

def unnormalize_image(tensor, mean=MEAN, std=STD):
    """Dés-normalise un tensor image (C, H, W) pour l'affichage."""
    tensor = tensor.clone().cpu();
    for t, m, s in zip(tensor, mean, std): t.mul_(s).add_(m)
    tensor = torch.clamp(tensor, 0, 1)
    img_np = tensor.permute(1, 2, 0).numpy(); img_np = (img_np * 255).astype(np.uint8)
    return img_np

# --- Fonction 1: Affichage Comparaison Segmentation ---
def display_segmentation_comparison(model, device, image_path, gt_seg_mask_path, input_transforms):
    """
    Affiche 3 volets: Image Originale, Segmentation GT, Segmentation Prédite.
    Charge l'image et le masque GT depuis les chemins fournis.
    """
    print(f"\nAffichage Comparaison Segmentation pour: {os.path.basename(image_path)}")
    # 1. Chargement
    try: original_pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError: print(f"Erreur: Image non trouvée: {image_path}"); return
    except Exception as e: print(f"Erreur chargement image {image_path}: {e}"); return
    target_pil_mask = None
    if gt_seg_mask_path and os.path.exists(gt_seg_mask_path):
        try: target_pil_mask = Image.open(gt_seg_mask_path).convert('L')
        except Exception as e: print(f"Warn: Erreur chargement masque GT {gt_seg_mask_path}: {e}")
    else: print(f"Info: Chemin masque GT ('{gt_seg_mask_path}') non fourni ou invalide.")

    # 2. Préparation Image pour Inférence
    try: image_tensor = input_transforms(original_pil_image)
    except Exception as e: print(f"Erreur application transforms: {e}"); return
    image_batch = image_tensor.unsqueeze(0).to(device)
    target_h, target_w = image_tensor.shape[1:] # Taille après transformation

    # 3. Inférence (on a besoin que de la sortie segmentation)
    model.eval();
    with torch.no_grad():
        try: _, pred_seg_logits = model(image_batch) # Ignorer la sortie profondeur
        except Exception as e_infer: print(f"Erreur inférence: {e_infer}"); traceback.print_exc(); return

    # 4. Post-traitement Prédiction Segmentation
    try: pred_mask_tensor = torch.argmax(pred_seg_logits, dim=1).squeeze().cpu(); pred_mask_color = apply_color_map(pred_mask_tensor, bdd100k_colors)
    except Exception as e_seg: print(f"Err post-proc seg pred: {e_seg}"); pred_mask_color = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # 5. Préparation Images pour Affichage
    original_image_display = np.array(original_pil_image)
    target_mask_color = None
    if target_pil_mask:
        try: target_pil_mask_resized = target_pil_mask.resize((target_w, target_h), Image.NEAREST); target_mask_color = apply_color_map(target_pil_mask_resized, bdd100k_colors)
        except Exception as e_gt_mask: print(f"Err coloration/resize masque GT: {e_gt_mask}")

    # 6. Affichage Figure 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Segmentation: {os.path.basename(image_path)}', fontsize=14)
    axes[0].imshow(original_image_display); axes[0].set_title('Originale'); axes[0].axis('off')
    if target_mask_color is not None: axes[1].imshow(target_mask_color); axes[1].set_title('Segmentation GT (Théorie)'); axes[1].axis('off')
    else: axes[1].text(0.5, 0.5, 'Non Fourni', ha='center', va='center'); axes[1].set_title('Segmentation GT'); axes[1].axis('off')
    try: axes[2].imshow(pred_mask_color); axes[2].set_title('Segmentation Prédite'); axes[2].axis('off')
    except Exception as e_plot_pred_s: print(f"Err plot pred seg: {e_plot_pred_s}"); axes[2].set_title('Err Pred Seg'); axes[2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


# --- Fonction 2: Affichage Comparaison Profondeur (avec RandomDepthDataset) ---
def display_depth_comparison(model, device, depth_dataset, idx=None):
    """
    Affiche 3 volets: Image Aléatoire, Profondeur GT Aléatoire, Profondeur Prédite.
    Utilise RandomDepthDataset pour obtenir l'image et la GT profondeur.
    """
    if not depth_dataset or len(depth_dataset) == 0: print("Erreur: RandomDepthDataset vide/None."); return
    if idx is None: idx = random.randint(0, len(depth_dataset) - 1)
    elif idx >= len(depth_dataset): print(f"Erreur: Index {idx} hors limites."); return

    print(f"\nAffichage Comparaison Profondeur pour Index Aléatoire: {idx}")
    try: image_tensor, gt_depth_tensor = depth_dataset[idx] # Récupère image et profondeur GT (aléatoires)
    except Exception as e: print(f"Erreur récupération data[{idx}] depuis RandomDepthDataset: {e}"); return

    # Inférence
    model.eval()
    image_batch = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        try: pred_depth, _ = model(image_batch) # Ignorer la sortie segmentation
        except Exception as e_infer: print(f"Erreur inférence: {e_infer}"); traceback.print_exc(); return

    # Post-traitement Prédiction Profondeur
    try: pred_depth_np = pred_depth.squeeze().cpu().numpy()
    except Exception as e_depth: print(f"Err post-proc prof pred: {e_depth}"); pred_depth_np = np.zeros(image_tensor.shape[1:])

    # Préparation GT Profondeur et Image Originale pour Affichage
    try: gt_depth_np = gt_depth_tensor.squeeze().cpu().numpy() # La GT est déjà un tensor
    except Exception as e_gt_depth: print(f"Err post-proc prof GT: {e_gt_depth}"); gt_depth_np = np.zeros_like(pred_depth_np)
    try: original_image_display = unnormalize_image(image_tensor) # Dés-normalise l'image aléatoire
    except Exception as e_unnorm: print(f"Err unnormalize: {e_unnorm}"); original_image_display = np.zeros((*image_tensor.shape[1:], 3), dtype=np.uint8)


    # Affichage Figure 1x3
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f'Profondeur (Image Aléatoire Idx {idx})', fontsize=14)
    axes[0].imshow(original_image_display); axes[0].set_title('Image Aléatoire (Entrée)'); axes[0].axis('off')
    try: im_gt = axes[1].imshow(gt_depth_np, cmap='magma'); axes[1].set_title('Profondeur GT (Aléatoire)'); axes[1].axis('off'); fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)
    except Exception as e_plot_gt_d: print(f"Err plot GT prof: {e_plot_gt_d}"); axes[1].set_title('Err GT Prof'); axes[1].axis('off')
    try: im_pred = axes[2].imshow(pred_depth_np, cmap='viridis'); axes[2].set_title('Profondeur Prédite'); axes[2].axis('off'); fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)
    except Exception as e_plot_pred_d: print(f"Err plot pred prof: {e_plot_pred_d}"); axes[2].set_title('Err Pred Prof'); axes[2].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()


# --- Script Principal de Test ---
if __name__ == '__main__':
    # --- Paramètres ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_height = 224; input_width = 224
    num_semantic_classes_bdd = 19
    depth_output_resolution_internal = (input_height, input_width) # Doit correspondre au modèle
    segmentation_output_resolution_internal = (input_height, input_width) # Doit correspondre au modèle
    epoch_to_load = 5 # Époque du modèle à charger
    model_load_path = f'../model/DepthSegNet/depth_seg_epoch_{epoch_to_load}.pth'
    root_data_dir = '../data' # Chemin vers le dossier data parent

    print(f"--- Test du Modèle DepthSegNet (Époque {epoch_to_load}) ---")
    print(f"Device: {device}, Input Size: ({input_height}, {input_width})")
    print(f"Chargement modèle depuis: {model_load_path}")

    # --- Transformations (Identiques à l'entraînement) ---
    inference_transforms = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    # --- Chargement du Modèle ---
    if not os.path.exists(model_load_path): print(f"FATAL: Fichier modèle non trouvé: {model_load_path}"); exit()
    model_test = DepthSegNet(
        num_semantic_classes=num_semantic_classes_bdd,
        depth_output_resolution=depth_output_resolution_internal,
        segmentation_output_resolution=segmentation_output_resolution_internal,
        resnet_model_path=None, pretrained=False
    ).to(device)
    try: model_test.load_state_dict(torch.load(model_load_path, map_location=device)); print("Modèle chargé avec succès.")
    except Exception as e_load: print(f"FATAL: Erreur chargement state_dict: {e_load}"); traceback.print_exc(); exit()


    # **************************************************
    # ************ TEST VISUEL SEGMENTATION ************
    # **************************************************
    print("\n--- Test Affichage Segmentation ---")
    # !!! MODIFIEZ CES CHEMINS vers une image et son masque GT BDD réels !!!
    split_seg_test = 'val' # ou 'train'
    image_name_seg = 'b1c9c847-3bda4659.jpg' # Exemple
    test_image_path_seg = os.path.join(root_data_dir, 'bdd100k_seg', 'bdd100k', 'seg', 'images', split_seg_test, image_name_seg)
    test_gt_seg_path = os.path.join(root_data_dir, 'bdd100k_seg', 'bdd100k', 'seg', 'labels', split_seg_test, os.path.splitext(image_name_seg)[0] + '.png')

    if os.path.exists(test_image_path_seg) and os.path.exists(test_gt_seg_path):
         display_segmentation_comparison(
             model=model_test,
             device=device,
             image_path=test_image_path_seg,
             gt_seg_mask_path=test_gt_seg_path,
             input_transforms=inference_transforms
         )
    else:
         print(f"Fichiers non trouvés pour le test de segmentation:")
         if not os.path.exists(test_image_path_seg): print(f"  - Image: {test_image_path_seg}")
         if not os.path.exists(test_gt_seg_path): print(f"  - Masque GT: {test_gt_seg_path}")


    # **************************************************
    # ************ TEST VISUEL PROFONDEUR **************
    # **************************************************
    print("\n--- Test Affichage Profondeur (avec RandomDepthDataset) ---")
    # Créer une instance du dataset aléatoire pour obtenir des paires (image_alea, gt_depth_alea)
    try:
        # Utiliser les mêmes transforms et la résolution de sortie de la tête de profondeur
        rand_depth_test_dataset = RandomDepthDataset(
            num_samples=10, # Créer quelques échantillons pour le test
            transform=inference_transforms,
            output_resolution=depth_output_resolution_internal
        )

        # Afficher une comparaison pour un échantillon aléatoire de ce dataset
        display_depth_comparison(
            model=model_test,
            device=device,
            depth_dataset=rand_depth_test_dataset
            # ou spécifier un index si vous voulez toujours le même
        )
    except Exception as e_depth_test:
        print(f"Erreur lors du test de profondeur avec RandomDepthDataset: {e_depth_test}")
        traceback.print_exc()


    print("\n--- Test Terminé ---")