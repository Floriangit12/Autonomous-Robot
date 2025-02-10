import torch
import torch.nn as nn
import torchvision.models as models

# --- Module HydraNet ---
class HydraNet(nn.Module):
    def __init__(self, backbone_name='resnet18', pretrained=True):
        super(HydraNet, self).__init__()
        
        # Deux backbones séparés pour chaque caméra
        backbone1 = getattr(models, backbone_name)(pretrained=pretrained)
        backbone2 = getattr(models, backbone_name)(pretrained=pretrained)

        # On retire les dernières couches FC
        self.shared_encoder1 = nn.Sequential(*list(backbone1.children())[:-2])
        self.shared_encoder2 = nn.Sequential(*list(backbone2.children())[:-2])

        feature_dim = 512  # Pour ResNet-18
        
        # Fusion après extraction des caractéristiques des deux caméras
        # La concaténation sur le canal donne 1024 canaux
        self.fusion_layer = nn.Linear(feature_dim * 2, feature_dim)  # 1024 -> 512

        # Tête pour la carte de profondeur
        self.depth_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        
        # Tête pour la classification
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )
        
        # Tête pour la détection de feux tricolores
        self.traffic_light_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Trois classes : rouge, jaune, vert
        )
    
    def forward(self, cam1, cam2):
        # Extraction des features des deux caméras
        features1 = self.shared_encoder1(cam1)   # [B, 512, H, W]
        features2 = self.shared_encoder2(cam2)     # [B, 512, H, W]

        # Concaténation sur la dimension des canaux -> [B, 1024, H, W]
        fused_features = torch.cat([features1, features2], dim=1)

        # Pooling adaptatif pour réduire la dimension spatiale à 1x1
        pooled_features = nn.AdaptiveAvgPool2d((1,1))(fused_features)  # [B, 1024, 1, 1]
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # [B, 1024]

        # Réduction de dimension par la couche linéaire
        fused_features = self.fusion_layer(pooled_features)  # [B, 512]

        # Re-former un tenseur 4D pour les têtes basées sur des convolutions
        fused_features = fused_features.view(fused_features.size(0), 512, 1, 1)  # [B, 512, 1, 1]

        # Sortie depth map
        depth_map = self.depth_head(fused_features)
        
        # Pour la classification et traffic light
        classifier_feat = self.classifier_head(fused_features)
        traffic_light_logits = self.traffic_light_head(fused_features)

        # Transformation en tokens pour le Transformer
        tokens = fused_features.view(fused_features.size(0), 512, -1).permute(0, 2, 1)  # [B, 1, 512]

        return depth_map, classifier_feat, traffic_light_logits, tokens

# --- Module FusionTransformer ---
class FusionTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(FusionTransformer, self).__init__()
        # Projection linéaire pour mettre les tokens à la dimension souhaitée pour le Transformer
        self.token_embedding = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, tokens):
        # tokens : [B, N_tokens, input_dim]
        x = self.token_embedding(tokens)  # [B, N_tokens, embed_dim]
        # Transformer attend [N_tokens, B, embed_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # Revenir à [B, N_tokens, embed_dim]
        x = x.permute(1, 0, 2)
        return x

# --- Module AutonomousDrivingModel ---
class AutonomousDrivingModel(nn.Module):
    def __init__(self, transformer_input_dim=512, transformer_embed_dim=256, num_heads=8, num_layers=4, decision_dim=128, output_dim=3):
        super(AutonomousDrivingModel, self).__init__()

        # HydraNet pour extraire les informations des deux caméras
        self.hydranet = HydraNet(backbone_name='resnet18', pretrained=True)

        # Embeddings pour les informations GPS, route et dynamique
        self.gps_embedding = nn.Linear(3, 64)        # Exemple : latitude, longitude, vitesse
        self.route_embedding = nn.Linear(10, 64)       # Exemple : vecteur de taille 10 pour route_seq
        self.dynamics_embedding = nn.Linear(5, 64)     # Exemple : vecteur de taille 5 pour dynamics

        # Transformer pour fusionner les tokens issus du backbone et les données supplémentaires
        self.fusion_transformer = FusionTransformer(input_dim=transformer_input_dim,
                                                    embed_dim=transformer_embed_dim,
                                                    num_heads=num_heads,
                                                    num_layers=num_layers)

        # Tête de décision : on supprime AdaptiveAvgPool1d ici car on a déjà effectué un pooling
        self.decision_head = nn.Sequential(
            nn.Linear(transformer_embed_dim, decision_dim),  # transformer_embed_dim = 256 -> decision_dim = 128
            nn.ReLU(),
            nn.Linear(decision_dim, output_dim)              # decision_dim -> output_dim (par exemple 3)
        )

    def forward(self, cam1, cam2, route_seq, dynamics, gps):
        # Extraction des features des caméras via HydraNet
        depth_map, classifier_feat, traffic_light_logits, tokens = self.hydranet(cam1, cam2)

        # Embeddings pour les autres données
        gps_token = self.gps_embedding(gps).unsqueeze(1)         # [B, 1, 64]
        route_token = self.route_embedding(route_seq).unsqueeze(1) # [B, 1, 64]
        dynamics_token = self.dynamics_embedding(dynamics).unsqueeze(1) # [B, 1, 64]

        # Concaténation des tokens : vision + GPS + route + dynamique
        fused_tokens = torch.cat([tokens, gps_token, route_token, dynamics_token], dim=1)  # [B, N_tokens+3, 512]

        # Passage dans le Transformer
        fused_tokens = self.fusion_transformer(fused_tokens)

        # Pooling global sur les tokens
        pooled = fused_tokens.mean(dim=1)  # [B, transformer_embed_dim] i.e. [B, 256]
        decision = self.decision_head(pooled)  # [B, output_dim]

        return decision, depth_map, classifier_feat, traffic_light_logits

    def summary(self, cam1_shape=(3, 224, 224), cam2_shape=(3, 224, 224), gps_shape=(3,), route_shape=(10,), dynamics_shape=(5,)):
        """
        Affiche un résumé détaillé du modèle avec les dimensions des tenseurs.
        """
        cam1 = torch.randn(1, *cam1_shape)
        cam2 = torch.randn(1, *cam2_shape)
        gps = torch.randn(1, *gps_shape)
        route_seq = torch.randn(1, *route_shape)
        dynamics = torch.randn(1, *dynamics_shape)

        print("===================================")
        print("     Résumé du Modèle")
        print("===================================")
        print(f"Entrées: cam1 {cam1_shape}, cam2 {cam2_shape}, gps {gps_shape}, route {route_shape}, dynamics {dynamics_shape}")

        depth_map, classifier_feat, traffic_light_logits, tokens = self.hydranet(cam1, cam2)
        print(f"HydraNet -> Depth Map : {depth_map.shape}")
        print(f"HydraNet -> Classifier Features : {classifier_feat.shape}")
        print(f"HydraNet -> Traffic Light Logits : {traffic_light_logits.shape}")
        print(f"HydraNet -> Tokens : {tokens.shape}")

        fused_tokens = self.fusion_transformer(tokens)
        print(f"Fusion Transformer -> Tokens Fusionnés : {fused_tokens.shape}")

        pooled = fused_tokens.mean(dim=1)
        print(f"Pooling Global -> Dimension : {pooled.shape}")

        decision = self.decision_head(pooled)
        print(f"Sortie Finale (Commandes de Pilotage) : {decision.shape}")

        print("===================================")

# --- Création du modèle et affichage du résumé ---
model = AutonomousDrivingModel()
model.summary()
