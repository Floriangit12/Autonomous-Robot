import torch
import torch.nn as nn

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
        # Transformer de PyTorch attend un shape [N_tokens, B, embed_dim]
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        # Revenir à [B, N_tokens, embed_dim]
        x = x.permute(1, 0, 2)
        return x
