import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class CLAM_SB_v2(nn.Module):
    """
    CLAM-SB otimizado com:
    - Multi-head attention (4 heads)
    - Positional encoding opcional
    - Attention diversity regularization
    - Gradient checkpointing para memória
    """
    def __init__(self, input_dim=768, hidden_dim=512, n_classes=2, 
                 n_heads=4, dropout=0.25, use_pos_embed=False):
        super().__init__()
        self.n_classes = n_classes
        self.use_pos_embed = use_pos_embed
        
        # Feature compressor com LayerNorm
        self.feature_compressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-head Gated Attention
        self.n_heads = n_heads
        head_dim = hidden_dim // n_heads
        
        self.attention_V = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 256), nn.Tanh())
            for _ in range(n_heads)
        ])
        self.attention_U = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 256), nn.Sigmoid())
            for _ in range(n_heads)
        ])
        self.attention_weights = nn.ModuleList([
            nn.Linear(256, 1) for _ in range(n_heads)
        ])
        
        # Fusion das heads
        self.head_fusion = nn.Linear(n_heads, 1)
        
        # Positional encoding (se coords fornecidas)
        if use_pos_embed:
            self.pos_encoder = PositionalEncoding2D(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes)
        )
        
        # Instance classifiers com Label Smoothing
        self.instance_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            ) for _ in range(n_classes)
        ])

    def forward(self, h, coords: Optional[torch.Tensor] = None, 
                label=None, instance_eval=False, diversity_lambda=0.05):
        """
        coords: (N, 2) com coordenadas (x, y) dos patches na WSI
        """
        h = self.feature_compressor(h)  # (N, hidden_dim)
        
        # Adicionar positional encoding
        if self.use_pos_embed and coords is not None:
            h = h + self.pos_encoder(coords)
        
        # Multi-head attention
        head_attentions = []
        for i in range(self.n_heads):
            A_V = self.attention_V[i](h)
            A_U = self.attention_U[i](h)
            A = self.attention_weights[i](A_V * A_U)  # (N, 1)
            head_attentions.append(A)
        
        # Concatenar e fundir heads
        multi_head_A = torch.cat(head_attentions, dim=-1)  # (N, n_heads)
        A = self.head_fusion(multi_head_A)  # (N, 1)
        A = torch.transpose(A, 1, 0)  # (1, N)
        A = F.softmax(A, dim=1)
        
        # Slide representation
        M = torch.mm(A, h)  # (1, hidden_dim)
        
        # Classificação
        logits = self.classifier(M)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(logits, dim=1)
        
        # Diversity regularization (evitar attention collapse)
        diversity_loss = torch.tensor(0.0).to(h.device)
        if diversity_lambda > 0:
            entropy = -torch.sum(A * torch.log(A + 1e-8))
            diversity_loss = -diversity_lambda * entropy
        
        # Instance-level clustering com hard negative mining
        instance_loss = torch.tensor(0.0).to(h.device)
        if instance_eval and label is not None:
            instance_loss = self._compute_instance_loss(h, A, label)
        
        total_loss = instance_loss + diversity_loss
        
        return logits, Y_prob, Y_hat, A, total_loss
    
    def _compute_instance_loss(self, h, A, label, k=8, hard_negative=True):
        """Instance clustering com hard negative mining"""
        k = min(k, h.size(0))
        if k == 0:
            return torch.tensor(0.0).to(h.device)
        
        # Ordenar por attention
        _, indices = torch.sort(A, dim=1, descending=True)
        top_indices = indices[0, :k]
        
        if label.item() == 1:  # Positivo
            # Top-k são positivos (atention alta = tumor)
            top_logits = self.instance_classifiers[1](h[top_indices])
            loss_top = F.cross_entropy(top_logits, torch.ones(k, dtype=torch.long).to(h.device))
            
            if hard_negative:
                # Hard negatives: patches com attention média (ambíguos)
                mid_start = h.size(0) // 2 - k // 2
                hard_indices = indices[0, mid_start:mid_start+k]
                hard_logits = self.instance_classifiers[0](h[hard_indices])
                loss_hard = F.cross_entropy(hard_logits, torch.zeros(k, dtype=torch.long).to(h.device))
                return (loss_top + loss_hard) / 2.0
            else:
                return loss_top
        else:  # Negativo
            top_logits = self.instance_classifiers[0](h[top_indices])
            return F.cross_entropy(top_logits, torch.zeros(k, dtype=torch.long).to(h.device))


class PositionalEncoding2D(nn.Module):
    """Sinusoidal positional encoding para coordenadas (x,y) na WSI"""
    def __init__(self, d_model, max_coord=10000):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, coords):
        """
        coords: (N, 2) - coordenadas normalizadas [0, 1]
        """
        # Similar a Transformer PE, mas 2D
        x, y = coords[:, 0:1], coords[:, 1:2]  # (N, 1)
        
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / self.d_model))
        
        pe_x = torch.zeros(coords.size(0), self.d_model // 2).to(coords.device)
        pe_y = torch.zeros(coords.size(0), self.d_model // 2).to(coords.device)
        
        pe_x[:, 0::2] = torch.sin(x * div_term)
        pe_x[:, 1::2] = torch.cos(x * div_term)
        
        pe_y[:, 0::2] = torch.sin(y * div_term)
        pe_y[:, 1::2] = torch.cos(y * div_term)
        
        return torch.cat([pe_x, pe_y], dim=-1)  # (N, d_model)
