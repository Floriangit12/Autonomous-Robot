# transformer_module.py (Corrigé et Reformaté pour entrée 480x720)
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import numpy as np
import warnings
import sys
import os

# --- Imports pour FLOPs et Mémoire ---
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    fvcore_available = True
except ImportError:
    print("Avertissement: fvcore non trouvé. Calcul FLOPs ignoré. pip install fvcore")
    fvcore_available = False
import torch.profiler
# --- Fin Imports ---

# --- Importer MultiTaskNet (optionnel, pour test dans main) ---
try:
    # Assumer que backbone_module est dans le même dossier ou PYTHONPATH
    from backbone_module import MultiTaskNet, regnet_1600M_config, print_model_summary
    BACKBONE_AVAILABLE = True
except ImportError:
    print("Warning: backbone_module not found. ViT analysis utilisera features simulées.")
    BACKBONE_AVAILABLE = False
    regnet_1600M_config = {} # Placeholder

# =============================================================================
# ====== SECTION : Vision Transformer pour l'Occupation 3D ==================
# =============================================================================

class ViTPatchEmbedding(nn.Module):
    """Image to Patch Embedding - Adapté pour 480x720 et patch 24x24"""
    def __init__(self, img_size=(480, 720), patch_size=24, in_chans=9, embed_dim=768): # *** MODIFIÉ ***
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Vérifier la divisibilité
        if not (img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0):
            raise ValueError(f'Image dimensions {img_size} must be divisible by patch size {patch_size}.')
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size) # (20, 30)
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 600
        print(f"ViTPatchEmbedding: Input={img_size}, Patch={patch_size}, Grid={self.grid_size}, Patches={self.num_patches}")
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Redimensionner si nécessaire (avec warning)
        if H != self.img_size[0] or W != self.img_size[1]:
            warnings.warn(f"ViTPatchEmbedding input size {H}x{W} != expected {self.img_size}. Resizing...", RuntimeWarning)
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)

        x = self.proj(x)       # (B, embed_dim, grid_h, grid_w)
        x = x.flatten(2)       # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class OccupancyHead(nn.Module):
    """Tête pour prédire la grille d'occupation 3D."""
    def __init__(self, embed_dim, grid_size, target_size=(100, 100, 20), final_channels=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_h, self.grid_w = grid_size # Sera (20, 30) pour entrée 480x720 patch 24
        self.target_h, self.target_w, self.target_z = target_size
        self.final_channels = final_channels
        # Couches convolutives pour traiter les patchs réorganisés
        self.conv_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Prédit Z*C canaux pour chaque pixel (H, W) de la grille cible
            nn.Conv2d(256, self.target_z * self.final_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x): # Input x: (B, num_patches, embed_dim)
        B = x.shape[0]
        # 1. Reshape en grille 2D: (B, embed_dim, grid_h, grid_w)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, self.grid_h, self.grid_w)
        # 2. Upsample spatialement à la taille HxW cible (100x100)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        # 3. Prédire les canaux Z via convolutions
        x = self.conv_head(x) # Shape: (B, Z*C, H_target, W_target)
        # 4. Reshape final pour obtenir la grille 3D (B, C, H, W, Z)
        x = x.view(B, self.final_channels, self.target_z, self.target_h, self.target_w)
        x = x.permute(0, 1, 3, 4, 2) # (B, C, H_target, W_target, Z_target)
        return x

class ViTOccupancyPredictor(nn.Module):
    """Vision Transformer pour prédire l'occupation 3D."""
    def __init__(self, input_feature_channels: dict, img_size=(480, 720), patch_size=24, # *** MODIFIÉ ***
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 output_grid_size=(100, 100, 20), pos_embed_dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.input_feature_keys = list(input_feature_channels.keys())
        total_input_channels = sum(input_feature_channels.values())
        print(f"ViT Init: Inputs={self.input_feature_keys}, Total chans={total_input_channels}, Input Size={img_size}, Patch={patch_size}")
        self.output_grid_size = output_grid_size
        self.embed_dim = embed_dim

        # Patch Embedding
        self.patch_embed = ViTPatchEmbedding(img_size, patch_size, total_input_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches # 600
        self.grid_size = self.patch_embed.grid_size     # (20, 30)

        # Positional Embedding (apprenable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_embed_dropout)

        # Transformer Encoder
        # Utiliser norm_first=True et GELU est courant et souvent plus stable
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=attn_dropout,
            activation='gelu', # GELU est fréquent dans les ViT
            batch_first=True,
            norm_first=True # Appliquer LayerNorm avant Attention/MLP
        )
        encoder_norm = nn.LayerNorm(embed_dim) # Norme finale après les couches
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, depth, encoder_norm)

        # Tête de prédiction 3D
        self.occupancy_head = OccupancyHead(embed_dim, self.grid_size, output_grid_size, final_channels=1)

        # Initialisation des poids
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialise les poids du modèle."""
        # Positional embedding avec Truncated Normal
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        # Appliquer l'initialisation spécifique aux sous-modules
        self.apply(self._init_weights_vit)

    def _init_weights_vit(self, m):
        """Initialisation spécifique pour les couches Linear, LayerNorm, Conv2d."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            # Kaiming init pour les convolutions
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, input_features: dict):
        """
        Args:
            input_features (dict[str, torch.Tensor]): Dictionnaire contenant les features
                                                     d'entrée (e.g., 'segmentation', 'depth').
                                                     Doivent être à la résolution img_size.
        Returns:
            dict[str, torch.Tensor]: Dictionnaire contenant la grille d'occupation prédite.
        """
        list_of_features = []
        # Concaténer les features d'entrée fournies
        for key in self.input_feature_keys:
            tensor = input_features.get(key)
            if tensor is None:
                raise ValueError(f"Input feature '{key}' missing in ViT input dictionary.")
            list_of_features.append(tensor)

        if not list_of_features:
            raise ValueError("No input features provided to ViT.")
        combined_features = torch.cat(list_of_features, dim=1)

        # 1. Patch Embedding
        x = self.patch_embed(combined_features) # (B, num_patches, embed_dim)

        # 2. Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 3. Transformer Encoder
        x = self.transformer_encoder(x) # (B, num_patches, embed_dim)

        # 4. Occupancy Head Prediction
        # Les patch tokens (sans CLS token ici) sont passés à la tête
        patch_tokens = x
        occupancy_grid = self.occupancy_head(patch_tokens) # (B, 1, H_out, W_out, Z_out)

        return {'occupancy': occupancy_grid}

# --- Fonction de Résumé ViT ---
def print_vit_summary(model):
    """Affiche un résumé des paramètres du modèle ViT."""
    print("-" * 80); print(f"ViT Summary ({model.__class__.__name__}):"); print("-" * 80); total_params=0
    def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) if hasattr(m,'parameters') else (m.numel() if hasattr(m,'numel') else 0)
    # Afficher les composants principaux
    parts = {
        'PatchEmbed': getattr(model, 'patch_embed', None),
        'PosEmbed': getattr(model, 'pos_embed', None), # Traité comme paramètre
        'PosDrop': getattr(model, 'pos_drop', None),
        'Transformer': getattr(model, 'transformer_encoder', None),
        'OccHead': getattr(model, 'occupancy_head', None)
    }
    for name, part_module in parts.items():
        params = count(part_module)
        if params > 0 or isinstance(part_module, (nn.TransformerEncoder, OccupancyHead, nn.Dropout)): # Inclure Dropout même si 0 params
            class_name = part_module.__class__.__name__ if part_module is not None else "Parameter" if name=='PosEmbed' else "N/A"
            print(f"  {name}: {class_name} ({params:,} params)")
            total_params += params
            # Détails pour Transformer et Head
            if name == 'Transformer' and part_module and hasattr(part_module, 'layers'):
                print(f"    - Layers: {len(part_module.layers)}")
            elif name == 'OccHead' and part_module:
                for sub_name, sub_child in part_module.named_children():
                    sub_params = count(sub_child)
                    if sub_params > 0:
                        print(f"    - {sub_name}: {sub_child.__class__.__name__} ({sub_params:,} params)")
    print("=" * 80); print(f"Total ViT Trainable Params: {total_params:,}"); print("=" * 80)

# --- Wrapper pour FLOPs ---
class ViTWrapperForFlops(nn.Module):
     """Wrapper pour passer les entrées dict comme tuple à FlopCountAnalysis."""
     def __init__(self, vit_model):
         super().__init__()
         self.vit_model = vit_model
         # Récupérer l'ordre des clés attendu par le modèle ViT
         self.input_keys = vit_model.input_feature_keys

     def forward(self, *features_tuple):
         # Recréer le dictionnaire à partir du tuple d'entrée
         # Assurer que le nombre d'arguments correspond aux clés attendues
         if len(features_tuple) != len(self.input_keys):
              raise ValueError(f"Wrapper expected {len(self.input_keys)} inputs ({self.input_keys}), got {len(features_tuple)}")
         inputs = {key: tensor for key, tensor in zip(self.input_keys, features_tuple)}
         return self.vit_model(inputs)

# =============================================================================
# ============ SECTION : MAIN BLOCK (Analyse avec résolution 480x720) =======
# =============================================================================

if __name__ == "__main__":
    # --- Configuration ---
    VIT_INPUT_RES = (480, 720)      # Entrée ViT = Sortie Têtes (H x W)
    VIT_PATCH_SIZE = 24             # Taille de patch qui divise 480 et 720
    NUM_CLASSES_SEGMENTATION = 8
    OUTPUT_GRID_SIZE = (100, 100, 20) # Occupation 3D cible

    # Config ViT
    VIT_EMBED_DIM = 768; VIT_DEPTH = 12; VIT_HEADS = 12

    try:
        # --- Simuler les features d'entrée pour ViT ---
        print(f"Simulating input features at {VIT_INPUT_RES} for ViT...")
        sim_seg_out = torch.randn(1, NUM_CLASSES_SEGMENTATION, VIT_INPUT_RES[0], VIT_INPUT_RES[1])
        sim_depth_out = torch.randn(1, 1, VIT_INPUT_RES[0], VIT_INPUT_RES[1])
        input_features_for_vit = {'segmentation': sim_seg_out, 'depth': sim_depth_out}

        # --- Instancier le ViT ---
        vit_input_channels = {'segmentation': sim_seg_out.shape[1], 'depth': sim_depth_out.shape[1]}
        print("\nInstantiating ViTOccupancyPredictor model...")
        vit_model = ViTOccupancyPredictor(
            input_feature_channels=vit_input_channels,
            img_size=VIT_INPUT_RES, # Entrée ViT
            patch_size=VIT_PATCH_SIZE,
            embed_dim=VIT_EMBED_DIM,
            depth=VIT_DEPTH,
            num_heads=VIT_HEADS,
            output_grid_size=OUTPUT_GRID_SIZE
        )
        vit_model.eval()
        print("ViTOccupancyPredictor instantiated.")
        print_vit_summary(vit_model)

        # --- Calcul FLOPs (ViT seulement) ---
        if fvcore_available:
            print("\nCalculating FLOPs for ViT...")
            try:
                # Préparer l'entrée tuple pour le wrapper Flops
                vit_inputs_tuple = tuple(input_features_for_vit[key] for key in vit_model.input_feature_keys)
                vit_wrapper = ViTWrapperForFlops(vit_model)
                if vit_inputs_tuple[0].shape[0] == 0: raise ValueError("Batch size 0 for FLOPs")

                flops_vit = FlopCountAnalysis(vit_wrapper, vit_inputs_tuple)
                print(f"ViT FLOPs: {flops_vit.total() / 1e9:.2f} GFLOPs")
                # print(flop_count_table(flops_vit)) # Optionnel: tableau détaillé
            except Exception as e:
                print(f"Could not calculate FLOPs for ViT: {e}")
        else:
            print("\nFLOPs calculation skipped (fvcore not installed).")

        # --- Profiling Mémoire (ViT seulement) ---
        print("\nProfiling Memory Usage for ViT...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
        vit_model.to(device)
        input_features_for_vit_dev = {k: v.to(device) for k, v in input_features_for_vit.items()}
        activities = [torch.profiler.ProfilerActivity.CPU];
        if torch.cuda.is_available(): activities.append(torch.profiler.ProfilerActivity.CUDA)

        print("Running profiled forward pass (ViT only)...")
        with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=False) as prof:
            with torch.no_grad():
                _ = vit_model(input_features_for_vit_dev)
        print("Profiled forward pass complete.")
        print("\nProfiler Results (Memory Focused, ViT only):")
        sort_key = "self_cuda_memory_usage" if torch.cuda.is_available() else "self_cpu_memory_usage"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
        if torch.cuda.is_available():
            print(f"\nPeak CUDA Mem Allocated (ViT): {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB")
            torch.cuda.reset_peak_memory_stats(device) # Reset stats

        # --- Test Forward Final ---
        print("\nTesting ViT forward pass with simulated features...")
        with torch.no_grad():
            vit_outputs = vit_model(input_features_for_vit_dev)
        print("ViT Forward pass successful!")
        occupancy_output = vit_outputs.get('occupancy')
        if occupancy_output is not None:
             print(f"  Occupancy grid output shape: {occupancy_output.shape}")
             expected_shape = (1, 1, OUTPUT_GRID_SIZE[0], OUTPUT_GRID_SIZE[1], OUTPUT_GRID_SIZE[2])
             assert occupancy_output.shape == expected_shape, f"Shape mismatch! Got {occupancy_output.shape}, expected {expected_shape}"
        else:
             print("  Occupancy output is None.")
        print("\nAnalysis complete.")

    except Exception as e:
        print(f"\n--- ERROR ---"); import traceback; traceback.print_exc(); print(f"Error: {e}")