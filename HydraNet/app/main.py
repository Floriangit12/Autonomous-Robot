# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# from torchvision.models.segmentation import deeplabv3_resnet101
# import torchvision.models as models
# from PIL import Image
# import numpy as np
# import os
# import ssl

# # Pour résoudre les problèmes de certificat SSL (temporaire, à résoudre !)
# ssl._create_default_https_context = ssl._create_unverified_context

# # --- Définition du Dataset Personnalisé pour la Profondeur (Reste inchangé) ---
# class RandomDepthDataset(torch.utils.data.Dataset):
#     def __init__(self, num_samples=1000, transform=None, output_resolution=(256, 512)):
#         self.num_samples = num_samples
#         self.transform = transform
#         self.output_resolution = output_resolution

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         image = Image.fromarray(np.random.randint(0, 256, size=(3, 256, 512), dtype=np.uint8).transpose(1, 2, 0))
#         depth_map = torch.rand(1, *self.output_resolution)
#         if self.transform:
#             image = self.transform(image)
#         return image, depth_map

# # --- Définition du Dataset Personnalisé pour BDD100K Segmentation ---
# class BDD100KSegmentation(torch.utils.data.Dataset):
#     def __init__(self, root_dir, split='train', transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.split = split
#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_dir = os.path.join(root_dir, 'images', split)
#         self.mask_dir = os.path.join(root_dir, 'labels', 'sem_seg', 'masks', split)
#         self.image_filenames = sorted(os.listdir(self.image_dir))
#         self.mask_filenames = sorted(os.listdir(self.mask_dir))

#         assert len(self.image_filenames) == len(self.mask_filenames), "Nombre d'images et de masques non concordant."

#     def __len__(self):
#         return len(self.image_filenames)

#     def __getitem__(self, idx):
#         img_name = self.image_filenames[idx]
#         mask_name = self.mask_filenames[idx]
#         img_path = os.path.join(self.image_dir, img_name)
#         mask_path = os.path.join(self.mask_dir, mask_name)

#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')

#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             mask = self.target_transform(mask)

#         if not isinstance(mask, torch.Tensor):
#             mask = torch.from_numpy(np.array(mask)).long()

#         return image, mask

# # --- Définition du Backbone Partagé ---
# class SharedBackbone(nn.Module):
#     def __init__(self, backbone_name='resnet50', pretrained=True):
#         super(SharedBackbone, self).__init__()
#         self.backbone = getattr(models, backbone_name)(pretrained=pretrained)
#         self.features = nn.Sequential(*list(self.backbone.children())[:-2])
#         self.out_channels = self._get_resnet_out_channels()

#     def _get_resnet_out_channels(self):
#         if isinstance(self.backbone.layer4[-1].conv2, nn.Conv2d):
#             return self.backbone.layer4[-1].conv2.out_channels
#         raise ValueError(f"Could not determine output channels for ResNet50")

#     def forward(self, x):
#         return self.features(x)

# # --- Définition des Heads ---
# class DepthHead(nn.Module):
#     def __init__(self, in_channels, output_resolution=(256, 512)):
#         super(DepthHead, self).__init__()
#         self.output_resolution = output_resolution
#         self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(128, 1, kernel_size=1)
#         self.upsample = nn.Upsample(scale_factor=output_resolution[0] / 32, mode='bilinear', align_corners=False)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.conv3(x)
#         x = self.upsample(x)
#         return x

# class SemanticSegmentationHead(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(SemanticSegmentationHead, self).__init__()
#         self.deeplab = deeplabv3_resnet101(pretrained=False, num_classes=num_classes, aux_loss=None)
#         if in_channels != 64:
#             self.deeplab.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

#     def forward(self, x):
#         output = self.deeplab(x)['out']
#         return output

# # --- Modèle Multi-Tâches (Depth et Segmentation) ---
# class DepthSegNet(nn.Module):
#     def __init__(self, backbone_name='resnet50', depth_output_resolution=(256, 512), num_semantic_classes=20):
#         super(DepthSegNet, self).__init__()
#         self.shared_backbone = SharedBackbone(backbone_name=backbone_name, pretrained=True)
#         self.depth_head = DepthHead(self.shared_backbone.out_channels, output_resolution=depth_output_resolution)
#         self.segmentation_head = SemanticSegmentationHead(self.shared_backbone.out_channels, num_semantic_classes)

#     def forward(self, x):
#         shared_features = self.shared_backbone(x)
#         depth = self.depth_head(shared_features)
#         segmentation = self.segmentation_head(shared_features)
#         return depth, segmentation

# if __name__ == '__main__':
#     # --- Paramètres d'entraînement ---
#     batch_size = 16
#     learning_rate = 0.001
#     num_epochs = 5
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # --- Initialisation du Modèle ---
#     # Définir le nombre de classes pour BDD100K (exemple, peut varier selon les annotations que vous utilisez)
#     num_semantic_classes_bdd = 19 # Exemple basé sur les classes courantes
#     depth_output_resolution = (256, 512)
#     model = DepthSegNet(num_semantic_classes=num_semantic_classes_bdd, depth_output_resolution=depth_output_resolution).to(device)

#     # --- Optimiseurs et Fonctions de Perte ---
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion_depth = nn.MSELoss()
#     criterion_segmentation = nn.CrossEntropyLoss(ignore_index=255)

#     # --- Transformations Communes ---
#     common_transforms = transforms.Compose([
#         transforms.Resize((256, 512)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     target_transforms = transforms.Compose([
#         transforms.Resize((256, 512), Image.NEAREST)
#     ])

#     # --- Préparation des Datasets et DataLoaders ---
#     depth_dataset = RandomDepthDataset(num_samples=1000, transform=common_transforms)
#     depth_loader = DataLoader(depth_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#     # **Dataset pour la Segmentation Sémantique (BDD100K)**
#     root_bdd = './data/bdd100k' # Assurez-vous que ce chemin est correct et que les données sont là
#     try:
#         bdd_train = BDD100KSegmentation(root_bdd, split='train', transform=common_transforms, target_transform=target_transforms)
#         segmentation_loader = DataLoader(bdd_train, batch_size=batch_size, shuffle=True, num_workers=4)
#         print("Dataset BDD100K (train) chargé avec succès.")
#     except Exception as e:
#         print(f"Erreur lors du chargement du dataset BDD100K: {e}")
#         segmentation_loader = None

#     # --- Boucle d'Entraînement Conjointe (Simplifiée) ---
#     print("--- Entraînement Conjoint des Têtes de Profondeur et de Segmentation (BDD100K) ---")
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss_depth = 0
#         total_loss_segmentation = 0

#         depth_iter = iter(depth_loader)
#         if segmentation_loader:
#             segmentation_iter = iter(segmentation_loader)
#             num_batches = min(len(depth_loader), len(segmentation_loader))
#         else:
#             num_batches = len(depth_loader)

#         for i in range(num_batches):
#             optimizer.zero_grad()

#             # Entraînement sur la profondeur
#             try:
#                 images_depth, depth_targets = next(depth_iter)
#                 images_depth = images_depth.to(device)
#                 depth_targets = depth_targets.to(device)
#                 depth_preds, _ = model(images_depth)
#                 loss_depth = criterion_depth(depth_preds, depth_targets)
#                 loss_depth.backward(retain_graph=True)
#                 total_loss_depth += loss_depth.item() * images_depth.size(0)
#             except StopIteration:
#                 pass

#             # Entraînement sur la segmentation (BDD100K)
#             if segmentation_loader:
#                 try:
#                     images_seg, segmentation_targets = next(segmentation_iter)
#                     images_seg = images_seg.to(device)
#                     segmentation_targets = segmentation_targets.to(device) # Les labels BDD sont déjà des tensors longs
#                     _, segmentation_preds = model(images_seg)
#                     loss_segmentation = criterion_segmentation(segmentation_preds, segmentation_targets)
#                     loss_segmentation.backward()
#                     total_loss_segmentation += loss_segmentation.item() * images_seg.size(0)
#                 except StopIteration:
#                     pass

#             optimizer.step()

#         avg_loss_depth = total_loss_depth / len(depth_dataset) if len(depth_dataset) > 0 else 0
#         avg_loss_segmentation = total_loss_segmentation / len(bdd_train) if segmentation_loader and len(bdd_train) > 0 else 0

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss Depth: {avg_loss_depth:.4f}, Loss Segmentation (BDD): {avg_loss_segmentation:.4f}")

#     print("\nEntraînement conjoint (simplifié) des têtes de profondeur et de segmentation (BDD100K) terminé.")import kagglehub

# Download latest version
import kagglehub
path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")


print("Path to dataset files:", path)