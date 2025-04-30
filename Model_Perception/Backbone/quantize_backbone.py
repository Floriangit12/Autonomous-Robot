# quantize_backbone.py
import torch
import torch.nn as nn
import torch.quantization
import os
import argparse
from tqdm import tqdm
import copy # Pour copier le modèle avant la fusion/quantification

# --- Importer la définition du modèle et les configs ---
try:
    from backbone_module import (
        MultiTaskNet,
        regnet_1600M_config,
        # Importer la classe Dataset si nécessaire pour la calibration
        # ou définir une version simplifiée ici
    )
    print("Import des modules du backbone réussi.")

    # Définir une classe Dataset simplifiée pour la calibration (charge seulement les images)
    from torchvision.transforms import v2 as transforms
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    from nuscenes.nuscenes import NuScenes # Nécessite nuscenes-devkit

    class CalibrationDataset(Dataset):
        def __init__(self, nusc_root, nusc_version='v1.0-mini', split='train',
                     network_input_size=(480, 720), limit=200): # Limiter le nb d'images pour calibration
            self.nusc_root = nusc_root
            self.nusc = NuScenes(version=nusc_version, dataroot=nusc_root, verbose=False)
            self.split = split
            self.network_input_size = network_input_size

            # Simplification pour les splits
            scenes = self.nusc.scene[:8] if 'mini' in nusc_version and split=='train' else \
                     self.nusc.scene[8:] if 'mini' in nusc_version and split=='val' else \
                     self.nusc.scene[:700] if 'trainval' in nusc_version and split=='train' else \
                     self.nusc.scene[700:750] if 'trainval' in nusc_version and split=='val' else \
                     self.nusc.scene[:min(len(self.nusc.scene), 100)] # Fallback limité

            self.samples = [s['data'].get('CAM_FRONT') for scene in scenes for s in self.nusc.sample if s['scene_token'] == scene['token'] and s['data'].get('CAM_FRONT')]
            self.samples = [s for s in self.samples if s][:limit] # Limiter le nombre d'échantillons
            print(f"Calibration dataset using {len(self.samples)} samples.")

            self.img_transform = transforms.Compose([
                transforms.Resize(self.network_input_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            cam_data_token = self.samples[idx]
            cam_data_rec = self.nusc.get('sample_data', cam_data_token)
            img_path = os.path.join(self.nusc_root, cam_data_rec['filename'])
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.img_transform(img)
            except Exception as e:
                print(f"Erreur chargement calibration image {img_path}: {e}")
                img_tensor = torch.zeros((3, self.network_input_size[0], self.network_input_size[1]))
            return img_tensor # Retourner seulement l'image

except ImportError as e:
    print(f"ERREUR CRITIQUE: Impossible d'importer depuis backbone_module: {e}")
    print("Assurez-vous que backbone_module.py est dans le PYTHONPATH ou le même dossier.")
    exit()
except ImportError:
    print("ERREUR CRITIQUE: nuscenes-devkit non trouvé. Installez-le.")
    exit()

# --- Fonction pour afficher la taille du modèle ---
def print_model_size(model, label=""):
    torch.save(model.state_dict(), "temp_model_size.pth")
    size_mb = os.path.getsize("temp_model_size.pth") / 1e6
    print(f"{label} Model size: {size_mb:.2f} MB")
    os.remove("temp_model_size.pth")

# --- Fonction principale de quantification ---
def quantize_model(float_model_path, save_path, calib_loader, device, args):
    """Applique la fusion et la quantification statique PTQ."""

    if not os.path.exists(float_model_path):
        print(f"Erreur: Modèle flottant non trouvé à {float_model_path}")
        return

    # --- 1. Charger le modèle flottant entraîné ---
    print("Chargement du modèle flottant entraîné...")
    # Recréer la config exacte du modèle (IMPORTANT)
    NETWORK_INPUT_SIZE = (args.img_height, args.img_width)
    HEAD_OUTPUT_SIZE = (args.out_height, args.out_width)
    NUM_CLASSES_SEGMENTATION = args.num_seg_classes

    backbone_cfg = regnet_1600M_config # Assumer que c'est la bonne
    neck_cfg = dict(fpn_channels=128, num_repeats=4, levels_to_use=['s2', 's3', 's4'], activation=nn.ReLU) # Doit correspondre à l'entraînement
    stage_map = {'s1':2,'s2':3,'s3':4,'s4':5}; prod_fpn_lvls=sorted([f"P{stage_map.get(lvl,int(lvl[1:])+1)}" for lvl in neck_cfg['levels_to_use']],key=lambda x:int(x[1:]))
    seg_cfg = dict(num_classes=NUM_CLASSES_SEGMENTATION, output_target_size=HEAD_OUTPUT_SIZE, levels_to_use=prod_fpn_lvls)
    depth_cfg = dict(output_target_size=HEAD_OUTPUT_SIZE, levels_to_use=prod_fpn_lvls)

    # Instancier l'architecture
    model_float = MultiTaskNet(backbone_cfg, False, None, neck_cfg, seg_cfg, depth_cfg)
    # Charger les poids entraînés
    model_float.load_state_dict(torch.load(float_model_path, map_location='cpu'))
    model_float.eval() # Très important: passer en mode évaluation
    print("Modèle flottant chargé.")
    print_model_size(model_float, "Float")

    # Copier le modèle pour la quantification (pour garder l'original intact)
    model_to_quantize = copy.deepcopy(model_float)
    model_to_quantize.eval()

    # --- 2. Configurer la Quantification ---
    # Choisir le backend ('fbgemm' pour x86, 'qnnpack' pour ARM)
    backend = args.backend
    qconfig = torch.quantization.get_default_qconfig(backend)
    # Pour une meilleure précision, on peut utiliser qconfig spécifiques comme QAT qconfig,
    # même pour PTQ, car ils utilisent souvent des observateurs plus robustes.
    # qconfig = torch.quantization.get_default_qat_qconfig(backend)
    model_to_quantize.qconfig = qconfig
    print(f"Configuration de quantification définie pour backend '{backend}'.")

    # --- 3. Fusionner les Modules ---
    # Fusionner Conv-BN ou Conv-BN-ReLU améliore précision et vitesse.
    # C'est souvent la partie la plus délicate car il faut connaître la structure.
    print("Fusion des modules (Conv-BN-ReLU)...")
    # Il faut spécifier les séquences de noms de couches à fusionner.
    # Exemple pour le stem simple: ['conv', 'bn', 'relu']
    # Exemple pour un ResBottleneckBlock: peut être complexe, dépend des noms internes.
    # On peut essayer une fusion plus automatique avec fuse_fx ou spécifier manuellement.
    # Pour la simplicité ici, on ne spécifie pas de fusion manuelle complexe,
    # MAIS C'EST FORTEMENT RECOMMANDÉ pour de bons résultats.
    # La fonction `prepare` tentera une fusion basique si on ne le fait pas avant.
    # torch.quantization.fuse_modules(model_to_quantize.backbone.stem, ['conv', 'bn', 'relu'], inplace=True)
    # Fusionner récursivement est mieux :
    model_fused = torch.quantization.fuse_modules_qat(model_to_quantize, qconfig=qconfig) # Utilise fuse_modules_qat qui peut gérer certains cas auto
    # Alternativement, une fonction de fusion manuelle plus ciblée serait préférable ici.
    # model_fused = fuse_model_manually(model_to_quantize) # Si vous créez cette fonction
    print("Fusion terminée (ou tentée par prepare).")


    # --- 4. Préparer le Modèle ---
    # Insère les modules observateurs (QuantStub, DeQuantStub) et prépare les couches pour la quantification.
    print("Préparation du modèle pour la calibration (insertion des observateurs)...")
    # Utiliser prepare_qat car il gère mieux la fusion et les observateurs robustes
    model_prepared = torch.quantization.prepare_qat(model_fused.train(), inplace=True) # Mettre en train pour que les BN calculent les stats lors de la calibration
    model_prepared.eval() # Remettre en eval après prepare si on ne fait que calibrer
    print("Modèle préparé.")

    # --- 5. Calibrer ---
    # Faire passer des données représentatives pour que les observateurs collectent les stats.
    print(f"Calibration avec {len(calib_loader.dataset)} images...")
    model_prepared.to(device)
    with torch.no_grad():
        pbar = tqdm(calib_loader, desc="Calibration")
        for images in pbar:
            images = images.to(device)
            model_prepared(images) # Pas besoin de la sortie, juste le passage forward
    print("Calibration terminée.")
    model_prepared.to('cpu') # Ramener sur CPU avant la conversion

    # --- 6. Convertir en Modèle Quantifié ---
    print("Conversion du modèle en int8...")
    model_quantized = torch.quantization.convert(model_prepared.eval(), inplace=True) # Assurer eval mode
    print("Modèle converti en int8.")
    print_model_size(model_quantized, "Quantized (int8)")

    # --- 7. Sauvegarder le Modèle Quantifié ---
    # Sauvegarder le state_dict du modèle quantifié
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model_quantized.state_dict(), save_path)
    print(f"Modèle quantifié (state_dict) sauvegardé dans : {save_path}")

    # Optionnel : Sauvegarder le modèle entier scripté (pour déploiement C++)
    # try:
    #     scripted_quantized_model = torch.jit.script(model_quantized)
    #     save_path_script = save_path.replace('.pth', '_scripted.pt')
    #     scripted_quantized_model.save(save_path_script)
    #     print(f"Modèle quantifié scripté sauvegardé dans : {save_path_script}")
    # except Exception as e:
    #     print(f"Erreur lors du scripting du modèle quantifié: {e}")


# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post Training Static Quantization for MultiTaskNet')
    # Paths
    parser.add_argument('--float_model_path', type=str, required=True, help='Path to the trained float MultiTaskNet checkpoint (.pth)')
    parser.add_argument('--save_path', type=str, default='./quantized/multitask_quant_static.pth', help='Path to save the quantized model state_dict')
    parser.add_argument('--nusc_root', type=str, required=True, help='Path to NuScenes dataset root (for calibration data)')
    parser.add_argument('--nusc_version', type=str, default='v1.0-mini', help='NuScenes version for calibration')

    # Model/Data Config (DOIT CORRESPONDRE AU MODÈLE ENTRAÎNÉ)
    parser.add_argument('--img_height', type=int, default=480, help='Network input image height used during training')
    parser.add_argument('--img_width', type=int, default=720, help='Network input image width used during training')
    parser.add_argument('--out_height', type=int, default=480, help='Head output height used during training') # *** MISE A JOUR ***
    parser.add_argument('--out_width', type=int, default=720, help='Head output width used during training')   # *** MISE A JOUR ***
    parser.add_argument('--num_seg_classes', type=int, default=8, help='Number of segmentation classes')

    # Quantization Params
    parser.add_argument('--backend', type=str, default='qnnpack', choices=['fbgemm', 'qnnpack', 'onednn'], help='Quantization backend (qnnpack for ARM, fbgemm for x86)')
    parser.add_argument('--calib_batch_size', type=int, default=8, help='Batch size for calibration data')
    parser.add_argument('--calib_limit', type=int, default=200, help='Number of images to use for calibration')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers for calibration')

    # System
    parser.add_argument('--use_gpu_calib', action='store_true', help='Use GPU for calibration forward pass (model converted on CPU)')

    args = parser.parse_args()

    # Préparer le DataLoader de calibration
    calib_device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu_calib else "cpu")
    print(f"Device for calibration pass: {calib_device}")
    calib_dataset = CalibrationDataset(
        nusc_root=args.nusc_root,
        nusc_version=args.nusc_version,
        split='train', # Utiliser une partie du train set pour calibrer
        network_input_size=(args.img_height, args.img_width),
        limit=args.calib_limit
    )
    calib_loader = DataLoader(calib_dataset, batch_size=args.calib_batch_size, shuffle=False, num_workers=args.num_workers)

    # Lancer la quantification
    quantize_model(args.float_model_path, args.save_path, calib_loader, calib_device, args)

    print("\nQuantification script finished.") 