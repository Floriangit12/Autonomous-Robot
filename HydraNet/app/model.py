# /app/model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import ssl
from torchinfo import summary
import torch.quantization # Importer pour la fusion et la quantification
import functools # Nécessaire pour afficher le qconfig complet
import traceback # Pour afficher les traces d'erreur

# Pour résoudre les problèmes de certificat SSL (solution temporaire !)
# Il est préférable de configurer correctement les certificats système.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Python < 2.7.9 ou OpenSSL < 1.0.0 ne gère pas les contextes SSL de cette manière
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    print("Avertissement: Contexte SSL non vérifié activé.")


# --- Définitions des Datasets ---
class RandomDepthDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, transform=None, output_resolution=(256, 512)):
        self.num_samples = num_samples
        self.transform = transform
        self.output_resolution = output_resolution
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        image_array = np.random.randint(0, 256, size=(3, 256, 512), dtype=np.uint8).transpose(1, 2, 0) # HWC
        image = Image.fromarray(image_array)
        depth_map = torch.rand(1, *self.output_resolution) # 1xHxW
        if self.transform:
            image = self.transform(image)
        return image, depth_map

# --- Classe Dataset BDD100K Corrigée ---
class BDD100KSegmentation(torch.utils.data.Dataset):
     def __init__(self, root_dir, split='train', transform=None, target_transform=None, ignore_index=255):
         """
         Initialise le dataset BDD100K pour la segmentation.

         Args:
             root_dir (str): Chemin vers le dossier PARENT contenant bdd100k_seg
                             (ex: '../data' si exécuté depuis /app)
             split (str): 'train', 'val', ou 'test'.
             transform (callable, optional): Transformations à appliquer à l'image.
             target_transform (callable, optional): Transformations à appliquer au masque.
             ignore_index (int): Valeur d'index à ignorer dans les masques.
         """
         self.root_dir = root_dir
         self.split = split
         self.transform = transform
         self.target_transform = target_transform
         self.ignore_index = ignore_index

         # --- CHEMINS CORRIGÉS ---
         # Chemin basé sur la structure visible dans l'image: root_dir -> bdd100k_seg -> bdd100k -> seg -> images/labels -> split
         self.image_dir = os.path.join(self.root_dir, 'bdd100k_seg', 'bdd100k', 'seg', 'images', self.split)
         self.mask_dir = os.path.join(self.root_dir, 'bdd100k_seg', 'bdd100k', 'seg', 'labels', self.split)
         # --- FIN DES CHEMINS CORRIGÉS ---

         # Vérifier si les dossiers (corrigés) existent
         if not os.path.isdir(self.image_dir):
             raise FileNotFoundError(f"Dossier images (corrigé) non trouvé: {self.image_dir}")
         if not os.path.isdir(self.mask_dir):
             # L'erreur originale venait d'ici, avec l'ancien chemin incorrect
             raise FileNotFoundError(f"Dossier masques (corrigé) non trouvé: {self.mask_dir}")

         # Liste des noms de fichiers image (souvent .jpg)
         self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
         # Liste des noms de fichiers masque (souvent .png)
         self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])

         # Vérification de la correspondance basée sur le nom de fichier sans extension
         img_basenames = {os.path.splitext(f)[0] for f in self.image_filenames}
         mask_basenames = {os.path.splitext(f)[0] for f in self.mask_filenames}

         if img_basenames != mask_basenames:
            print(f"Attention: Discordance fichiers image/masque dans split '{self.split}'.")
            common_basenames = sorted(list(img_basenames.intersection(mask_basenames)))
            self.image_filenames = [f"{name}.jpg" for name in common_basenames]
            self.mask_filenames = [f"{name}.png" for name in common_basenames]
            print(f"  Utilisation de {len(self.image_filenames)} paires correspondantes.")

         if len(self.image_filenames) == 0:
             print(f"Avertissement : Aucun fichier image/masque trouvé/correspondant dans les dossiers corrigés pour le split '{self.split}'.")

     def __len__(self):
         return len(self.image_filenames)

     def __getitem__(self, idx):
         img_name = self.image_filenames[idx]
         # Construire le nom du masque à partir du nom de l'image
         mask_name = os.path.splitext(img_name)[0] + '.png'

         img_path = os.path.join(self.image_dir, img_name)
         mask_path = os.path.join(self.mask_dir, mask_name)

         # Vérification finale de l'existence du masque (sécurité)
         if not os.path.exists(mask_path):
             # Tenter de trouver un masque avec un nom similaire ou lever une erreur plus spécifique
             # Alternative: chercher mask_name dans self.mask_filenames (plus lent)
             raise FileNotFoundError(f"Masque spécifique {mask_path} non trouvé pour l'image {img_path}.")

         try:
             image = Image.open(img_path).convert('RGB')
             mask = Image.open(mask_path).convert('L') # Charger en niveaux de gris (indices de classe)
         except Exception as e:
             print(f"Erreur ouverture image/masque: Img={img_path}, Mask={mask_path}")
             raise e

         # Appliquer les transformations
         if self.transform: image = self.transform(image)
         if self.target_transform: mask = self.target_transform(mask)

         # Assurer que le masque est un Tensor Long si pas déjà fait
         if not isinstance(mask, torch.Tensor):
             # Convertir l'image PIL en Tensor
             mask_tensor = transforms.functional.to_tensor(mask) # Donne [1, H, W] type FloatTensor [0,1]
             # Convertir en LongTensor avec les indices de classe [0-N] et squeeze dimension canal
             mask = (mask_tensor * 255).long().squeeze(0) # Donne [H, W] type LongTensor [0-255]

         # Gérer l'ignore_index si nécessaire DANS le tensor
         if self.ignore_index is not None:
             # Appliquer l'ignore_index après conversion en tensor Long
             # Attention si ignore_index=255 et que les masques contiennent déjà 255
             mask[mask == 255] = self.ignore_index # Mappe la valeur 255 du fichier vers ignore_index

         return image, mask
     
# --- Définitions des Modules ---
class SharedBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, model_path=None):
        super(SharedBackbone, self).__init__()
        self.backbone = None; weights = None
        if model_path and os.path.exists(model_path):
            print(f"Tentative chargement local: {model_path}")
            try:
                self.backbone = getattr(models, backbone_name)(weights=None)
                state_dict = torch.load(model_path, map_location='cpu')
                miss, unexp = self.backbone.load_state_dict(state_dict, strict=False)
                if unexp: print(f"  Warn: Clés inattendues: {unexp[:5]}...")
                if miss: print(f"  Warn: Clés manquantes: {miss[:5]}...")
                print(f"Poids locaux chargés (strict=False)")
                pretrained = False
            except Exception as e: print(f"Échec chargement local: {e}. Tentative torchvision."); self.backbone = None
        if self.backbone is None and pretrained:
             print(f"Utilisation poids torchvision pour {backbone_name}.")
             try:
                 if backbone_name == 'resnet50': weights = models.ResNet50_Weights.IMAGENET1K_V1
                 elif backbone_name == 'resnet101': weights = models.ResNet101_Weights.IMAGENET1K_V1
                 elif backbone_name == 'resnet18': weights = models.ResNet18_Weights.IMAGENET1K_V1
                 else: print(f"Warn: Poids non gérés: {backbone_name}.")
                 self.backbone = getattr(models, backbone_name)(weights=weights)
                 print("Poids torchvision chargés.")
             except Exception as e: print(f"Erreur chargement torchvision: {e}. Init sans poids."); self.backbone = getattr(models, backbone_name)(weights=None)
        elif self.backbone is None:
             print(f"Init {backbone_name} sans poids pré-entraînés."); self.backbone = getattr(models, backbone_name)(weights=None)

        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        if backbone_name in ['resnet50', 'resnet101', 'resnet152']: self.out_channels = 2048
        elif backbone_name in ['resnet18', 'resnet34']: self.out_channels = 512
        else:
            try:
                with torch.no_grad(): self.out_channels = self.features(torch.randn(1, 3, 224, 224)).shape[1]
            except Exception as e:
                print(f"Warn: Déterm. dynamique out_channels échouée: {e}")
                if backbone_name.startswith('resnet'): self.out_channels = 2048
                else: raise ValueError(f"Cannot determine out_channels for {backbone_name}")
        print(f"Backbone {backbone_name} prêt. Out channels: {self.out_channels}")
    def forward(self, x): return self.features(x)

class DepthHead(nn.Module):
    def __init__(self, in_channels, output_resolution=(256, 512)):
        super(DepthHead, self).__init__()
        self.output_resolution = output_resolution
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), # 0, 1
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True), # 2, 3
            nn.Conv2d(128, 1, kernel_size=1, bias=True), # 4
            nn.Upsample(size=self.output_resolution, mode='bilinear', align_corners=False) # 5
        )
    def forward(self, x): return self.head(x)

class SemanticSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, output_resolution=(1080, 1920)):
        super(SemanticSegmentationHead, self).__init__()
        self.output_resolution = output_resolution
        intermediate_channels = 256
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, padding=1, bias=False), # 0
            nn.BatchNorm2d(intermediate_channels), # 1
            nn.ReLU(inplace=True), # 2
            nn.Conv2d(intermediate_channels, num_classes, kernel_size=1, bias=True), # 3
            nn.Upsample(size=self.output_resolution, mode='bilinear', align_corners=False) # 4
        )
    def forward(self, x): return self.head(x)

class DepthSegNet(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, depth_output_resolution=(256, 512),
                 segmentation_output_resolution=(1080, 1920), num_semantic_classes=20, resnet_model_path=None):
        super(DepthSegNet, self).__init__()
        print(f"Init DepthSegNet: backbone={backbone_name}, pretrained={pretrained}, classes={num_semantic_classes}")
        self.shared_backbone = SharedBackbone(backbone_name, pretrained, resnet_model_path)
        backbone_out_channels = self.shared_backbone.out_channels
        self.depth_head = DepthHead(backbone_out_channels, depth_output_resolution)
        self.segmentation_head = SemanticSegmentationHead(backbone_out_channels, num_semantic_classes, segmentation_output_resolution)
    def forward(self, x):
        shared_features = self.shared_backbone(x)
        depth_output = self.depth_head(shared_features)
        segmentation_output = self.segmentation_head(shared_features)
        return depth_output, segmentation_output

# --- Fonction de Fusion (Mise à jour pour tentative MAXIMALE) ---
def fuse_depthsegnet_modules(model_to_fuse):
    """
    Tente de fusionner le MAXIMUM de modules possibles (ConvBn, ConvReLU, ConvBnReLU)
    dans un modèle DepthSegNet pour la quantification PTQ. Modifie le modèle inplace.
    """
    print("\n--- Début de la fusion MAXIMALE des modules (pour PTQ) ---")
    if not isinstance(model_to_fuse, DepthSegNet):
        print("Avertissement: Le modèle fourni n'est pas une instance de DepthSegNet.")
        return model_to_fuse
    model_to_fuse.eval() # IMPORTANT pour la fusion BN
    total_fusions_attempted = 0
    total_fusions_succeeded = 0

    # --- 1. Fusion dans SharedBackbone (ResNet interne) ---
    if hasattr(model_to_fuse.shared_backbone, 'backbone') and isinstance(model_to_fuse.shared_backbone.backbone, models.ResNet):
        print("Fusion dans SharedBackbone (ResNet)...")
        resnet_model = model_to_fuse.shared_backbone.backbone
        try: # Fusion initiale
            total_fusions_attempted += 1
            if hasattr(resnet_model, 'conv1') and hasattr(resnet_model, 'bn1') and hasattr(resnet_model, 'relu'):
                 torch.quantization.fuse_modules(resnet_model, [['conv1', 'bn1', 'relu']], inplace=True)
                 print("  - Succès: Backbone initial [conv1, bn1, relu]")
                 total_fusions_succeeded += 1
            elif hasattr(resnet_model, 'conv1') and hasattr(resnet_model, 'bn1'):
                 torch.quantization.fuse_modules(resnet_model, [['conv1', 'bn1']], inplace=True)
                 print("  - Succès: Backbone initial [conv1, bn1]")
                 total_fusions_succeeded += 1
        except Exception as e_init: print(f"  - Info: Échec fusion initiale: {e_init}")

        # Itérer et tenter toutes fusions standards par bloc
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(resnet_model, layer_name):
                layer = getattr(resnet_model, layer_name)
                print(f"  - Traitement de {layer_name}...")
                for block_idx, module in enumerate(layer):
                    block_type = type(module).__name__; print(f"    - Traitement Bloc {block_idx} ({block_type})")
                    # Tentative Downsample
                    try:
                        if module.downsample is not None and len(module.downsample) == 2 and \
                           isinstance(module.downsample[0], nn.Conv2d) and isinstance(module.downsample[1], nn.BatchNorm2d):
                            total_fusions_attempted += 1
                            torch.quantization.fuse_modules(module.downsample, ['0', '1'], inplace=True)
                            print(f"      - Succès: Downsample [Conv, BN]")
                            total_fusions_succeeded += 1
                    except Exception as e_ds: print(f"      - Info: Échec/NA fusion downsample: {e_ds}")
                    # Tentatives Bottleneck
                    if isinstance(module, models.resnet.Bottleneck):
                        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(module, ['conv1', 'bn1', 'relu'], inplace=True); print(f"      - Succès: Bottleneck [conv1, bn1, relu]"); total_fusions_succeeded += 1
                        except Exception as e_b1: print(f"      - Info: Échec/NA fusion [conv1, bn1, relu]: {e_b1}")
                        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(module, ['conv2', 'bn2', 'relu'], inplace=True); print(f"      - Succès: Bottleneck [conv2, bn2, relu]"); total_fusions_succeeded += 1
                        except Exception as e_b2: print(f"      - Info: Échec/NA fusion [conv2, bn2, relu]: {e_b2}")
                        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(module, ['conv3', 'bn3'], inplace=True); print(f"      - Succès: Bottleneck [conv3, bn3]"); total_fusions_succeeded += 1
                        except Exception as e_b3: print(f"      - Info: Échec/NA fusion [conv3, bn3]: {e_b3}")
                    # Tentatives BasicBlock
                    elif isinstance(module, models.resnet.BasicBlock):
                        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(module, ['conv1', 'bn1', 'relu'], inplace=True); print(f"      - Succès: BasicBlock [conv1, bn1, relu]"); total_fusions_succeeded += 1
                        except Exception as e_bb1: print(f"      - Info: Échec/NA fusion [conv1, bn1, relu]: {e_bb1}")
                        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(module, ['conv2', 'bn2'], inplace=True); print(f"      - Succès: BasicBlock [conv2, bn2]"); total_fusions_succeeded += 1
                        except Exception as e_bb2: print(f"      - Info: Échec/NA fusion [conv2, bn2]: {e_bb2}")
    else: print("Avertissement: Backbone ResNet non détecté.")

    # --- 2. Fusion dans DepthHead ---
    print("Fusion dans DepthHead...")
    if hasattr(model_to_fuse, 'depth_head') and hasattr(model_to_fuse.depth_head, 'head') and isinstance(model_to_fuse.depth_head.head, nn.Sequential):
        dh_seq = model_to_fuse.depth_head.head
        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(dh_seq, ['0', '1'], inplace=True); print("  - Succès: DepthHead [Conv(0), ReLU(1)]"); total_fusions_succeeded += 1
        except Exception as e_dh1: print(f"  - Info: Échec/NA fusion DepthHead [0, 1]: {e_dh1}")
        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(dh_seq, ['2', '3'], inplace=True); print("  - Succès: DepthHead [Conv(2), ReLU(3)]"); total_fusions_succeeded += 1
        except Exception as e_dh2: print(f"  - Info: Échec/NA fusion DepthHead [2, 3]: {e_dh2}")
    else: print("  - Info: Structure DepthHead non compatible.")

    # --- 3. Fusion dans SemanticSegmentationHead ---
    print("Fusion dans SemanticSegmentationHead...")
    if hasattr(model_to_fuse, 'segmentation_head') and hasattr(model_to_fuse.segmentation_head, 'head') and isinstance(model_to_fuse.segmentation_head.head, nn.Sequential):
        sh_seq = model_to_fuse.segmentation_head.head
        try: total_fusions_attempted += 1; torch.quantization.fuse_modules(sh_seq, ['0', '1', '2'], inplace=True); print("  - Succès: SegHead [Conv(0), BN(1), ReLU(2)]"); total_fusions_succeeded += 1
        except Exception as e_sh1: print(f"  - Info: Échec/NA fusion SegHead [0, 1, 2]: {e_sh1}")
    else: print("  - Info: Structure SegHead non compatible.")

    print(f"--- Fusion MAXIMALE terminée. {total_fusions_succeeded}/{total_fusions_attempted} tentatives standards réussies. ---")
    return model_to_fuse

# --- Fonction get_model_info ---
def get_model_info(model, input_size=(1, 3, 720, 1280), device='cpu', state_name="Original"):
    try:
        model_name = model.__class__.__name__
        print(f"\n--- Résumé du modèle {model_name} ({state_name}) ---")
        model.to(device)
        # Créer une entrée dummy sur le bon device
        dummy_input = torch.randn(input_size, device=device, dtype=torch.float32)
        # Pour les modèles quantifiés, l'entrée doit parfois être quantifiée aussi,
        # mais torchinfo gère souvent les entrées float32 pour l'analyse de structure.
        # Si erreur spécifique, envisager: dummy_input = torch.quantize_per_tensor(dummy_input, 1.0, 0, torch.quint8)
        print(summary(model, input_data=dummy_input,
                      col_names=["input_size", "output_size", "num_params", "mult_adds"],
                      device=device, depth=3 ))
        print("--- Fin du résumé ---")
    except Exception as e:
        print(f"Erreur durant torchinfo.summary pour état '{state_name}': {e}")
        traceback.print_exc()



# #########  test du model avec affichage #######
# # --- Paramètres ---
# batch_size = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Utilisation du device: {device}")

# # --- Initialisation du Modèle ---
# num_semantic_classes_bdd = 19
# depth_res = (256, 512)
# seg_res = (720, 1280) # HD pour limiter usage mémoire
# resnet_weights_path = '../model/resnet/resnet50-0676ba61.pth'
# if not os.path.exists(resnet_weights_path):
#     print(f"Avertissement: Poids ResNet non trouvés: {resnet_weights_path}.")
#     resnet_weights_path = None

# model_fp32 = DepthSegNet(num_semantic_classes=num_semantic_classes_bdd, depth_output_resolution=depth_res,
#                          segmentation_output_resolution=seg_res, resnet_model_path=resnet_weights_path,
#                          pretrained=(resnet_weights_path is None), backbone_name='resnet50').to(device)

# summary_input_size = (batch_size, 3, seg_res[0], seg_res[1])
# get_model_info(model_fp32, input_size=summary_input_size, device=device, state_name="Original FP32")

# # --- ÉTAPE DE FUSION ---
# model_fp32_fused = fuse_depthsegnet_modules(model_fp32)

# # --- Afficher le résumé du modèle FUSIONNÉ ---
# get_model_info(model_fp32_fused, input_size=summary_input_size, device=device, state_name="FP32 Fusionné")

# # --- Suite : Quantification Post-Entraînement (PTQ) ---
# print("\n--- Préparation pour la Quantification Post-Entraînement (PTQ) ---")
# model_fp32_fused.eval()
# model_fp32_fused_cpu = model_fp32_fused.to('cpu')
# print(f"Modèle déplacé sur CPU pour quantification.")

# # --- Forcer le backend FBGEMM ---
# print("Configuration backend quantification -> FBGEMM (pour CPU x86)")
# torch.backends.quantized.engine = 'fbgemm'
# q_backend = torch.backends.quantized.engine
# print(f"Backend utilisé: {q_backend}")

# try:
#     model_fp32_fused_cpu.qconfig = torch.quantization.get_default_qconfig(q_backend)
#     print(f"Utilisation qconfig PTQ: {model_fp32_fused_cpu.qconfig}")

#     model_fp32_prepared = torch.quantization.prepare(model_fp32_fused_cpu, inplace=False)
#     print("Modèle préparé pour PTQ (observateurs insérés - CALIBRATION REQUISE).")

#     # --- CALIBRATION (ÉTAPE MANQUANTE MAIS CRUCIALE) ---
#     print("\n--- Début Étape Calibration (MANQUANTE - À FAIRE) ---")
#     print("!!! ATTENTION: CALIBRATION avec données réelles est ESSENTIELLE pour précision INT8 !!!")
#     # ICI : Ajouter la boucle de calibration avec un DataLoader (ex: calibration_loader)
#     # model_fp32_prepared.eval()
#     # with torch.no_grad():
#     #     for calib_images, _ in calibration_loader:
#     #         model_fp32_prepared(calib_images.to('cpu')) # Passe avant sur CPU
#     # print("--- Calibration (théoriquement) terminée ---")
#     # -----------------------------------------------------

#     # Conversion en INT8
#     # model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
#     # print("Modèle converti en INT8 (basé sur observateurs).")
#     # model_int8.eval()

#     # Afficher résumé INT8
    
#     get_model_info(model_fp32, input_size=summary_input_size, device='cpu', state_name="sans fused model ")
#     get_model_info(model_fp32_prepared, input_size=summary_input_size, device='cpu', state_name="INT8 Quantifié")

# except Exception as e_quant:
#     print(f"\n--- ERREUR durant processus quantification ---")
#     traceback.print_exc()

# print("\nProcessus terminé.")