# train_backbone.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import warnings

# --- Importer les définitions du modèle ---
try:
    from backbone_module import (
        MultiTaskNet,
        regnet_1600M_config,
        create_regnet_backbone # Pour charger poids ImageNet
    )
    print("Import des modules du backbone réussi.")
except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer depuis backbone_module: {e}")
    exit()
warnings.filterwarnings("ignore", message=".*contiguous.*")

# --- Dataset (Exemple simplifié - ADAPTER À VOS DONNÉES) ---
# Ce dataset charge des images et prépare des GT pour Seg/Depth
# à la résolution de sortie des têtes (480x720 dans ce cas)
from nuscenes.nuscenes import NuScenes # Nécessite nuscenes-devkit

class RoadDataset(Dataset): # Renommé pour clarté
    # !!! Adapter cette classe pour charger VOS vraies données GT !!!
    # !!! Datasets recommandés : Cityscapes, BDD100K (Seg); DDAD, KITTI (Depth) !!!
    def __init__(self, dataset_root, dataset_type='cityscapes', # ou 'kitti', 'bdd', 'custom'
                 split='train',
                 network_input_size=(480, 720), # Taille entrée réseau HxW
                 head_output_size=(480, 720)): # Taille sortie têtes HxW (IDENTIQUE INPUT)
        self.dataset_root = dataset_root
        self.dataset_type = dataset_type.lower()
        self.split = split
        self.network_input_size = network_input_size
        self.head_output_size = head_output_size

        # TODO: Logique pour lister les fichiers images/gt selon le dataset et split
        # Exemple simplifié : on suppose des dossiers img/, seg_gt/, depth_gt/
        self.img_dir = os.path.join(dataset_root, split, 'image')
        self.seg_gt_dir = os.path.join(dataset_root, split, 'seg_gt')   # Ex: Fichiers PNG avec classes ID
        self.depth_gt_dir = os.path.join(dataset_root, split, 'depth_gt') # Ex: Fichiers NPY ou PNG 16bit (mètres * X)

        if not os.path.isdir(self.img_dir): raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        # Ne pas crasher si GT manque, mais avertir
        self.has_seg = os.path.isdir(self.seg_gt_dir)
        self.has_depth = os.path.isdir(self.depth_gt_dir)
        if not self.has_seg: print(f"Warning: Seg GT dir not found: {self.seg_gt_dir}")
        if not self.has_depth: print(f"Warning: Depth GT dir not found: {self.depth_gt_dir}")

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        print(f"Found {len(self.image_files)} images in {self.img_dir} for split '{split}'")

        # Transformations
        self.img_transform = transforms.Compose([
            transforms.Resize(self.network_input_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.seg_gt_transform = transforms.Compose([
            transforms.Resize(self.head_output_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToImage(),
        ])
        self.depth_gt_transform = transforms.Compose([
             transforms.Resize(self.head_output_size, interpolation=transforms.InterpolationMode.BILINEAR), # Bilinear est ok pour depth
             transforms.ToImage(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # --- Charger Image ---
        try: img = Image.open(img_path).convert('RGB'); img_tensor = self.img_transform(img)
        except Exception as e: print(f"Erreur image {img_path}: {e}"); img_tensor = torch.zeros((3, self.network_input_size[0], self.network_input_size[1]))

        # --- Charger Segmentation GT ---
        seg_gt_tensor = torch.full(self.head_output_size, 255, dtype=torch.long) # Utiliser ignore_index par défaut
        if self.has_seg:
            # Trouver le fichier GT correspondant (peut nécessiter adaptation du nom)
            seg_gt_name = img_name.replace('.jpg', '_gtFine_labelIds.png') # Exemple Cityscapes
            # seg_gt_name = img_name.replace('.jpg', '_drivable_id.png') # Exemple BDD100K Drivable Area
            seg_gt_path = os.path.join(self.seg_gt_dir, seg_gt_name)
            if os.path.exists(seg_gt_path):
                 try:
                     seg_map_pil = Image.open(seg_gt_path) # Assumer que c'est une image avec des IDs
                     seg_gt_tensor = self.seg_gt_transform(seg_map_pil).squeeze(0).to(torch.long)
                 except Exception as e: print(f"Erreur seg GT {seg_gt_path}: {e}")
            # else: print(f"Seg GT non trouvé: {seg_gt_path}") # Trop verbeux

        # --- Charger Profondeur GT ---
        depth_gt_tensor = torch.zeros((1, self.head_output_size[0], self.head_output_size[1])) # Zéro par défaut
        if self.has_depth:
            # Trouver le fichier GT correspondant
            depth_gt_name = img_name.replace('.png', '_depth.npy') # Exemple (si .npy)
            # depth_gt_name = img_name.replace('.png', '_depth.png') # Exemple (si 16bit png)
            depth_gt_path = os.path.join(self.depth_gt_dir, depth_gt_name)
            if os.path.exists(depth_gt_path):
                try:
                    if depth_gt_name.endswith('.npy'):
                        depth_array = np.load(depth_gt_path).astype(np.float32) # Assumer mètres
                        depth_map_pil = Image.fromarray(depth_array)
                    elif depth_gt_name.endswith('.png'):
                         # Charger PNG 16bit et convertir (ex: divisier par 256 si KITTI format)
                         depth_map_pil = Image.open(depth_gt_path)
                         depth_array = np.array(depth_map_pil, dtype=np.float32) / 256.0 # Exemple KITTI
                         depth_map_pil = Image.fromarray(depth_array)
                    else: raise NotImplementedError("Format Depth GT non supporté")
                    depth_gt_tensor = self.depth_gt_transform(depth_map_pil)
                except Exception as e: print(f"Erreur depth GT {depth_gt_path}: {e}")
            # else: print(f"Depth GT non trouvé: {depth_gt_path}") # Trop verbeux

        return img_tensor, seg_gt_tensor, depth_gt_tensor

# --- Fonction d'Entraînement Epoch ---
def train_epoch(model, dataloader, optimizer, criterion_seg, criterion_depth, lambda_seg, lambda_depth, device, scaler):
    model.train(); total_loss, total_seg_loss, total_depth_loss = 0.0, 0.0, 0.0
    pbar = tqdm(dataloader, desc="Training Epoch")
    for images, seg_gt, depth_gt in pbar:
        images, seg_gt, depth_gt = images.to(device), seg_gt.to(device), depth_gt.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            seg_pred, depth_pred = outputs.get('segmentation'), outputs.get('depth')
            loss, seg_loss, depth_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            if seg_pred is not None and seg_gt.numel() > 0: # Check if GT non vide
                assert seg_pred.shape[-2:]==seg_gt.shape[-2:], "Seg shape mismatch"
                seg_loss = criterion_seg(seg_pred, seg_gt)
                if not torch.isnan(seg_loss): loss += lambda_seg * seg_loss
            if depth_pred is not None and depth_gt.numel() > 0: # Check if GT non vide
                assert depth_pred.shape[-2:]==depth_gt.shape[-2:], "Depth shape mismatch"
                mask = (depth_gt > 1e-3) & (depth_gt < 100.0) # Masque profondeur valide
                if mask.sum() > 0: depth_loss = criterion_depth(depth_pred[mask], depth_gt[mask])
                if not torch.isnan(depth_loss): loss += lambda_depth * depth_loss
        if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        elif loss > 0: loss.backward(); optimizer.step()
        total_loss+=loss.item() if loss>0 else 0; total_seg_loss+=seg_loss.item(); total_depth_loss+=depth_loss.item()
        pbar.set_postfix(loss=f"{loss.item():.3f}", seg=f"{seg_loss.item():.3f}", depth=f"{depth_loss.item():.3f}")
    n=len(dataloader); avg_loss=total_loss/n if n>0 else 0; avg_seg=total_seg_loss/n if n>0 else 0; avg_depth=total_depth_loss/n if n>0 else 0;
    print(f"Epoch Train Summary: Avg Loss={avg_loss:.4f}, Seg={avg_seg:.4f}, Depth={avg_depth:.4f}"); return avg_loss

# --- Main Script ---
def main(args):
    NETWORK_INPUT_SIZE = (args.img_height, args.img_width)  # Ex: (480, 720)
    HEAD_OUTPUT_SIZE = (args.out_height, args.out_width) # Ex: (480, 720) - IDENTIQUE
    print(f"Network Input Size (HxW): {NETWORK_INPUT_SIZE}")
    print(f"Head Output Size (HxW): {HEAD_OUTPUT_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"); print(f"Using device: {device}")
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None; print(f"AMP Enabled: {scaler is not None}")

    # Config Modèle
    backbone_cfg = regnet_1600M_config
    neck_cfg = dict(fpn_channels=128, num_repeats=4, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU)
    NUM_CLASSES_SEGMENTATION = args.num_seg_classes

    stage_map={'s1':2,'s2':3,'s3':4,'s4':5}; prod_fpn_lvls=sorted([f"P{stage_map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))
    seg_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_SIZE, levels_to_use=prod_fpn_lvls)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_SIZE, levels_to_use=prod_fpn_lvls)

    # Instanciation Modèle + Poids Initiaux
    print("Instantiating MultiTaskNet model...")
    model = MultiTaskNet(backbone_cfg, True, args.load_backbone_pth, neck_cfg, seg_cfg, depth_cfg)
    model.to(device); print("Model instantiated and moved to device.")

    # Dataset et DataLoader
    print("Loading dataset...")
    try:
        train_dataset = RoadDataset(dataset_root=args.dataset_root, dataset_type=args.dataset_type, split='train', network_input_size=NETWORK_INPUT_SIZE, head_output_size=HEAD_OUTPUT_SIZE)
        if len(train_dataset) == 0: raise ValueError("Dataset est vide!")
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}"); return
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Pertes, Optimiseur, Scheduler
    criterion_seg = nn.CrossEntropyLoss(ignore_index=args.seg_ignore_index)
    criterion_depth = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)

    # Entraînement
    print("Starting training...")
    os.makedirs(args.save_dir, exist_ok=True); start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        avg_loss = train_epoch(model, train_loader, optimizer, criterion_seg, criterion_depth, args.lambda_seg, args.lambda_depth, device, scaler)
        scheduler.step(); current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Current LR: {current_lr:.6f}")
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_path = os.path.join(args.save_dir, f"multitask_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path); print(f"Model state_dict saved to {save_path}")
    final_save_path = os.path.join(args.save_dir, "multitask_final.pth"); torch.save(model.state_dict(), final_save_path); print(f"Final model state_dict saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiTaskNet Backbone Training')
    # Paths
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset root (e.g., /path/to/cityscapes)')
    parser.add_argument('--dataset_type', type=str, default='cityscapes', choices=['cityscapes', 'bdd100k', 'kitti', 'custom'], help='Type of dataset being used (influences GT loading)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_backbone', help='Directory to save checkpoints')
    parser.add_argument('--load_backbone_pth', type=str, default=None, help='Optional: Path to pre-trained RegNet backbone weights (ImageNet)')
    # Training Params
    parser.add_argument('--epochs', type=int, default=40, help='Num epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_step_size', type=int, default=15, help='LR scheduler step size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--lambda_seg', type=float, default=1.0, help='Segmentation loss weight')
    parser.add_argument('--lambda_depth', type=float, default=0.5, help='Depth loss weight')
    parser.add_argument('--num_seg_classes', type=int, default=8, help='Number of segmentation classes (MUST match dataset GT)')
    parser.add_argument('--seg_ignore_index', type=int, default=255, help='Ignore index for segmentation loss (e.g., 255 for Cityscapes void)')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint frequency (epochs)')
    # Data Params (Résolution 480x720 In/Out)
    parser.add_argument('--img_height', type=int, default=480, help='Target network input image height')
    parser.add_argument('--img_width', type=int, default=720, help='Target network input image width')
    parser.add_argument('--out_height', type=int, default=480, help='Target head output height') # IDENTIQUE INPUT
    parser.add_argument('--out_width', type=int, default=720, help='Target head output width')   # IDENTIQUE INPUT
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    # System
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    main(args)