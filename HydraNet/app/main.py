# /app/main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset # Importer Subset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode # Pour Resize moderne
# Retiré : import torchvision.datasets as datasets # Non utilisé directement
# Retiré : from torchvision.models.segmentation import deeplabv3_resnet101 # Non utilisé
# Retiré : import torchvision.models as models # Non utilisé directement
from PIL import Image
import numpy as np
import os
import ssl
import random
import matplotlib.pyplot as plt
import traceback # Pour afficher les traces d'erreur

# S'assurer que model.py est dans le même répertoire ou accessible via PYTHONPATH
try:
    from model import DepthSegNet, RandomDepthDataset, BDD100KSegmentation
except ImportError as e:
    print(f"Erreur d'import depuis model.py: {e}")
    print("Vérifiez que model.py est dans le bon répertoire et ne contient pas d'erreurs.")
    exit()


# Pour résoudre les problèmes de certificat SSL (temporaire, à résoudre !)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    print("Avertissement: Contexte SSL non vérifié activé.")

# --- Constantes et Fonctions Utilitaires pour la Visualisation ---

# Moyenne et Écart-type utilisés pour la normalisation (doivent correspondre à l'entraînement)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Palette de couleurs BDD100K (19 classes) - Liste de tuples RGB (0-255)
bdd100k_colors = [
    (128, 64, 128),  # 0 road
    (244, 35, 232),  # 1 sidewalk
    (70, 70, 70),    # 2 building
    (102, 102, 156), # 3 wall
    (190, 153, 153), # 4 fence
    (153, 153, 153), # 5 pole
    (250, 170, 30),  # 6 traffic light
    (220, 220, 0),   # 7 traffic sign
    (107, 142, 35),  # 8 vegetation
    (152, 251, 152), # 9 terrain
    (70, 130, 180),  # 10 sky
    (220, 20, 60),   # 11 person
    (255, 0, 0),     # 12 rider
    (0, 0, 142),     # 13 car
    (0, 0, 70),      # 14 truck
    (0, 60, 100),    # 15 bus
    (0, 80, 100),    # 16 train
    (0, 0, 230),     # 17 motorcycle
    (119, 11, 32),   # 18 bicycle
    (0, 0, 0)        # 19: Couleur pour index ignoré ou inconnu (optionnel, si ignore_index=19)
]
# Index à ignorer (utilisé pour la coloration et la loss)
ignore_index_global = 255
num_classes_display = 19 # Nombre de classes valides pour la coloration

def apply_color_map(mask, color_map, num_classes=num_classes_display, ignore_idx=ignore_index_global):
    """Applique une palette de couleurs à un masque de segmentation 2D."""
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze().cpu().numpy().astype(int) # Assurer type entier

    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    # Prendre les couleurs des classes valides (0 à num_classes-1)
    processed_map = np.array(color_map[:num_classes], dtype=np.uint8)

    # Définir une couleur pour l'index ignoré (utiliser la dernière couleur si disponible)
    ignore_color = [0, 0, 0] # Noir par défaut
    if ignore_idx is not None and len(color_map) > num_classes:
         ignore_color = color_map[num_classes]
    elif ignore_idx is not None: # Si la couleur n'est pas définie mais ignore_idx l'est
         ignore_color = [255, 255, 255] # Blanc par exemple

    # Appliquer les couleurs pixel par pixel
    for r in range(h):
        for c in range(w):
            idx = mask[r, c]
            if ignore_idx is not None and idx == ignore_idx:
                 colored_mask[r, c, :] = ignore_color
            elif 0 <= idx < num_classes: # Index de classe valide
                 colored_mask[r, c, :] = processed_map[idx]
            # else: Laisser noir pour les index invalides/inattendus

    return colored_mask

# --- Fonction d'Affichage pour une Image Spécifique ---
def display_prediction_from_path(model, image_path, device, input_transforms):
    """
    Charge une image depuis un chemin, effectue une prédiction avec le modèle,
    et affiche l'image originale, la profondeur et la segmentation prédites.
    """
    print(f"\nAffichage de la prédiction pour : {os.path.basename(image_path)}")
    try:
        original_pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError: print(f"Erreur: Image non trouvée: {image_path}"); return
    except Exception as e: print(f"Erreur chargement image {image_path}: {e}"); return

    try: image_tensor = input_transforms(original_pil_image)
    except Exception as e: print(f"Erreur application transforms: {e}"); return

    image_batch = image_tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        try: pred_depth, pred_seg_logits = model(image_batch)
        except Exception as e_infer: print(f"Erreur durant inférence: {e_infer}"); traceback.print_exc(); return

    try: pred_depth_np = pred_depth.squeeze().cpu().numpy()
    except Exception as e_depth: print(f"Erreur post-traitement profondeur: {e_depth}"); pred_depth_np = np.zeros(image_tensor.shape[1:])

    try:
        pred_mask_tensor = torch.argmax(pred_seg_logits, dim=1).squeeze().cpu()
        pred_mask_color = apply_color_map(pred_mask_tensor, bdd100k_colors, ignore_idx=ignore_index_global)
    except Exception as e_seg: print(f"Erreur post-traitement segmentation: {e_seg}"); pred_mask_color = np.zeros((*image_tensor.shape[1:], 3), dtype=np.uint8)

    original_image_display = np.array(original_pil_image)
    num_plots = 3; fig_width = num_plots * 6; fig_height = 5
    fig, axes = plt.subplots(1, num_plots, figsize=(fig_width, fig_height))
    if num_plots == 1: axes = [axes]

    axes[0].imshow(original_image_display); axes[0].set_title(f'Image: {os.path.basename(image_path)}'); axes[0].axis('off')
    try:
        im_depth = axes[1].imshow(pred_depth_np, cmap='viridis'); axes[1].set_title('Prédiction Profondeur'); axes[1].axis('off')
        fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)
    except Exception as e_plot_depth: print(f"Err affichage prof: {e_plot_depth}"); axes[1].set_title('Err Profondeur'); axes[1].axis('off')
    try: axes[2].imshow(pred_mask_color); axes[2].set_title('Prédiction Segmentation'); axes[2].axis('off')
    except Exception as e_plot_seg: print(f"Err affichage seg: {e_plot_seg}"); axes[2].set_title('Err Segmentation'); axes[2].axis('off')

    plt.tight_layout(); plt.show()


# --- Section Principale ---
if __name__ == '__main__':
    # --- Paramètres ---
    # Adaptez ces paramètres selon vos besoins et capacités matérielles
    batch_size = 8 # Réduit pour potentiellement moins de charge mémoire
    learning_rate = 0.001
    num_epochs = 5 # Nombre d'époques d'entraînement
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 0 # 0 pour simple, >0 peut accélérer si CPU/IO le permet
    limit_images_bdd = 100 # Limite pour l'entraînement BDD
    input_height = 224
    input_width = 224
    num_semantic_classes_bdd = 19
    # Résolutions INTERNES des têtes du modèle (doivent correspondre à l'architecture de model.py)
    # La résolution de sortie de la tête de profondeur
    depth_output_resolution_internal = (224, 224) # Exemple: Adapter si besoin
     # La résolution de sortie de la tête de segmentation (où elle fait l'upsample final)
    segmentation_output_resolution_internal = (input_height, input_width) # Généralement la même que l'entrée/cible

    print(f"Utilisation du device: {device}")
    print(f"Taille d'entrée/cible: ({input_height}, {input_width})")
    print(f"Entraînement sur {limit_images_bdd} images BDD.")

    # --- Transformations (Utilisées pour entraînement ET inférence/affichage) ---
    common_transforms = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    target_transforms = transforms.Compose([
        transforms.Resize((input_height, input_width), interpolation=InterpolationMode.NEAREST),
        # La conversion en Tensor Long est maintenant dans le Dataset BDD100KSegmentation
    ])

    # --- Initialisation du Modèle ---
    resnet_path = '../model/resnet/resnet50-0676ba61.pth'
    if not os.path.exists(resnet_path):
        print(f"Avertissement: Poids ResNet non trouvés à {resnet_path}. Utilisation poids ImageNet si pretrained=True.")
        resnet_path_arg = None; pretrained_arg = True
    else:
        resnet_path_arg = resnet_path; pretrained_arg = False

    model = DepthSegNet(
        num_semantic_classes=num_semantic_classes_bdd,
        depth_output_resolution=depth_output_resolution_internal,
        segmentation_output_resolution=segmentation_output_resolution_internal,
        resnet_model_path=resnet_path_arg,
        pretrained=pretrained_arg
    ).to(device)

    # --- Optimiseurs et Fonctions de Perte ---
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion_depth = nn.MSELoss()
    criterion_segmentation = nn.CrossEntropyLoss(ignore_index=ignore_index_global)

    # --- Préparation des Datasets et DataLoaders ---
    # Dataset Profondeur Aléatoire
    depth_dataset = RandomDepthDataset(num_samples=1000, transform=common_transforms, output_resolution=depth_output_resolution_internal)
    depth_loader = DataLoader(depth_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)

    # Dataset Segmentation BDD100K (avec Subset)
    root_bdd = '../data/bdd100k'
    segmentation_loader = None
    bdd_train_subset = None
    bdd_train_full_len = 0
    dataset_for_display = None # Variable pour stocker le dataset à utiliser pour l'affichage

    try:
        bdd_train_full = BDD100KSegmentation(root_bdd, split='train', transform=common_transforms,
                                             target_transform=target_transforms, ignore_index=ignore_index_global)
        bdd_train_full_len = len(bdd_train_full)
        if bdd_train_full_len > 0:
            indices = list(range(min(limit_images_bdd, bdd_train_full_len)))
            bdd_train_subset = Subset(bdd_train_full, indices)
            print(f"Dataset BDD100K (train) chargé. Subset de {len(bdd_train_subset)} images utilisé.")
            segmentation_loader = DataLoader(bdd_train_subset, batch_size=batch_size, shuffle=True,
                                             num_workers=num_workers, pin_memory=True if device.type == 'cuda' else False)
            dataset_for_display = bdd_train_subset # Utiliser le subset pour l'affichage aussi dans cet exemple
        else: print(f"Avertissement: Dataset BDD100K vide depuis {root_bdd}")
    except FileNotFoundError as e: print(f"Erreur Fichier Non Trouvé: Dataset BDD100K ({root_bdd}). Détail: {e}")
    except Exception as e: print(f"Erreur chargement dataset BDD100K: {e}"); traceback.print_exc()


    # **************************************************
    # ************ SECTION D'ENTRAÎNEMENT ************
    # **************************************************
    print("\n--- Début de l'Entraînement ---")
    if not segmentation_loader:
        print("ATTENTION: DataLoader segmentation non dispo. Entraînement profondeur (aléatoire) seulement.")

    for epoch in range(num_epochs):
        model.train()
        total_loss_depth = 0; total_loss_segmentation = 0
        processed_seg_samples = 0
        batch_count = 0

        # Itérer sur le loader principal (segmentation si dispo, sinon profondeur)
        main_loader = segmentation_loader if segmentation_loader else depth_loader
        if not main_loader: print("Aucun DataLoader disponible, arrêt entraînement."); break
        num_batches_epoch = len(main_loader)

        # Utiliser des itérateurs séparés si les deux tâches sont actives
        depth_iter = iter(depth_loader)
        seg_iter = iter(segmentation_loader) if segmentation_loader else None

        for i in range(num_batches_epoch):
            optimizer.zero_grad()
            combined_loss = 0
            current_batch_size = 0 # Pour normaliser la loss

            # --- Tâche Segmentation (si active) ---
            if seg_iter:
                try:
                    images_seg, segmentation_targets = next(seg_iter)
                    images_seg = images_seg.to(device)
                    segmentation_targets = segmentation_targets.to(device, dtype=torch.long)
                    current_batch_size = images_seg.size(0)

                    # Inférer pour les deux têtes avec cette image
                    depth_preds, segmentation_preds = model(images_seg)

                    # Loss Segmentation
                    loss_segmentation = criterion_segmentation(segmentation_preds, segmentation_targets)
                    total_loss_segmentation += loss_segmentation.item() * current_batch_size
                    processed_seg_samples += current_batch_size
                    combined_loss += loss_segmentation

                    # Loss Profondeur (avec cible aléatoire pour l'exemple)
                    try:
                        _, depth_targets_rand = next(depth_iter) # Ignorer l'image, prendre la cible
                        depth_targets_rand = depth_targets_rand.to(device)
                        # S'assurer que la cible correspond à la prédiction
                        if depth_preds.shape == depth_targets_rand.shape:
                            loss_depth = criterion_depth(depth_preds, depth_targets_rand)
                            total_loss_depth += loss_depth.item() * current_batch_size
                            combined_loss += loss_depth # Ajouter au total
                        else: pass # Ignorer si tailles incompatibles
                    except StopIteration: pass # Ignorer si depth_loader est fini

                except StopIteration: break # Fin du loader de segmentation
                except Exception as e_batch: print(f"Erreur batch seg {i}: {e_batch}"); traceback.print_exc(); continue

            # --- Tâche Profondeur (si seule active) ---
            elif not seg_iter: # Seulement si segmentation_loader n'existe pas
                 try:
                    images_depth, depth_targets = next(depth_iter)
                    images_depth = images_depth.to(device); depth_targets = depth_targets.to(device)
                    current_batch_size = images_depth.size(0)
                    depth_preds, _ = model(images_depth)
                    loss_depth = criterion_depth(depth_preds, depth_targets)
                    total_loss_depth += loss_depth.item() * current_batch_size
                    combined_loss += loss_depth
                 except StopIteration: break # Fin du loader de profondeur
                 except Exception as e_batch: print(f"Erreur batch prof {i}: {e_batch}"); traceback.print_exc(); continue

            # --- Rétropropagation & Optimisation ---
            if isinstance(combined_loss, torch.Tensor) and combined_loss != 0:
                 combined_loss.backward()
                 optimizer.step()
            elif combined_loss == 0 and i < num_batches_epoch -1 : # Ne pas afficher si c'est juste la fin
                 print(f"Warn: Batch {i} avec loss=0 ou non-Tensor. Step ignoré.")

            batch_count += 1
            if batch_count % 10 == 0: print(f'  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_count}/{num_batches_epoch}] traité.')

        # Fin de l'époque
        avg_loss_depth = total_loss_depth / len(depth_dataset) if len(depth_dataset) > 0 else 0
        avg_loss_segmentation = total_loss_segmentation / processed_seg_samples if processed_seg_samples > 0 else 0
        print(f"Fin Epoch [{epoch+1}/{num_epochs}], Avg Loss Depth: {avg_loss_depth:.4f}, Avg Loss Seg (Subset): {avg_loss_segmentation:.4f}")

        # Sauvegarde du modèle
        model_save_dir = '../model/DepthSegNet'
        os.makedirs(model_save_dir, exist_ok=True)
        save_path = os.path.join(model_save_dir, f'depth_seg_epoch_{epoch+1}.pth')
        try: torch.save(model.state_dict(), save_path); print(f"Modèle sauvegardé: {save_path}")
        except Exception as e_save: print(f"Erreur sauvegarde modèle: {e_save}")

    print("\n--- Entraînement Terminé ---")


    # ******************************************************
    # ************ SECTION D'AFFICHAGE POST-TRAIN ************
    # ******************************************************
    print("\n--- Début de la Section d'Affichage ---")
    # Charger le dernier modèle sauvegardé
    epoch_to_load = num_epochs
    model_load_path = f'../model/DepthSegNet/depth_seg_epoch_{epoch_to_load}.pth'

    if os.path.exists(model_load_path):
        print(f"\nChargement du modèle depuis: {model_load_path} pour affichage.")
        # Ré-instancier avec la MÊME architecture que pendant l'entraînement
        model_eval = DepthSegNet(
            num_semantic_classes=num_semantic_classes_bdd,
            depth_output_resolution=depth_output_resolution_internal,
            segmentation_output_resolution=segmentation_output_resolution_internal,
            resnet_model_path=None, pretrained=False # Charger uniquement le state_dict
        ).to(device)
        try:
            model_eval.load_state_dict(torch.load(model_load_path, map_location=device))
            print("Modèle chargé avec succès pour l'affichage.")

            # --- Chemins vers les images à tester ---
            # !!! MODIFIEZ CES CHEMINS vers des images réelles !!!
            test_image_paths = [
                 "../data/bdd100k/images/10k/val/b1c9c847-3bda4659.jpg", # Exemple BDD Val
                 "../data/bdd100k/images/10k/val/b1c9c847-7a8b3c6a.jpg", # Exemple BDD Val
                 # Ajoutez ici les chemins absolus ou relatifs (depuis /app) vers VOS images
                 # "/chemin/complet/vers/mon_image.jpg",
            ]

            valid_test_paths = [p for p in test_image_paths if os.path.exists(p)]
            if not valid_test_paths:
                 print("\nAucun chemin d'image valide trouvé. Modifiez 'test_image_paths'.")
            else:
                 print(f"\nAffichage des prédictions pour {len(valid_test_paths)} image(s)...")
                 for img_path in valid_test_paths:
                     # Utiliser les mêmes transformations que l'entraînement pour l'inférence
                     display_prediction_from_path(
                         model=model_eval,
                         image_path=img_path,
                         device=device,
                         input_transforms=common_transforms
                     )
        except FileNotFoundError: print(f"Erreur: Fichier modèle non trouvé: {model_load_path}")
        except Exception as e_load: print(f"Erreur chargement state_dict: {e_load}"); traceback.print_exc()
    else:
        print(f"\nModèle de l'époque {epoch_to_load} non trouvé ({model_load_path}). Impossible d'afficher.")

    print("\n--- Script Principal Terminé ---")