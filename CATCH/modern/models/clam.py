import torch
import torch.nn as nn
import torch.nn.functional as F

class CLAM_SB(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, dropout=False, n_classes=2, gate=True):
        """
        CLAM-SB (Single Branch) Model.
        Args:
            input_dim (int): Dimension of input features (e.g. 768 for ViT/CTransPath).
            hidden_dim (int): Hidden dimension size.
            dropout (bool): Whether to use dropout.
            n_classes (int): Number of output classes.
            gate (bool): Whether to use Gated Attention.
        """
        super(CLAM_SB, self).__init__()
        self.n_classes = n_classes

        # Feature Compressor
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        self.feature_compressor = nn.Sequential(*layers)

        # Attention Network
        self.gate = gate
        if gate:
            self.attention_V = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(256, 1)
        else:
            self.attention_net = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # Instance Classifiers (for optional instance-level clustering loss)
        # Note: In full CLAM, we train these. For simplicity here, we define them.
        self.instance_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in range(n_classes)
        ])

    def forward(self, h, label=None, instance_eval=False):
        """
        Args:
            h (torch.Tensor): Input features of shape (N, input_dim), where N is number of patches.
            label (torch.Tensor, optional): Ground truth label.
            instance_eval (bool): If True, returns instance-level predictions.
        """
        h = self.feature_compressor(h) # (N, hidden_dim)

        # Attention
        if self.gate:
            A_V = self.attention_V(h)
            A_U = self.attention_U(h)
            A = self.attention_weights(A_V * A_U) # (N, 1)
        else:
            A = self.attention_net(h) # (N, 1)

        A = torch.transpose(A, 1, 0) # (1, N)
        A = F.softmax(A, dim=1) # (1, N)

        # Slide-level representation
        M = torch.mm(A, h) # (1, hidden_dim)

        # Logits
        logits = self.classifier(M) # (1, n_classes)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        # Instance-level Loss (simplified)
        instance_loss = torch.tensor(0.0).to(h.device)
        # To fully implement CLAM loss, we'd need to sample top-k/bottom-k patches
        # and train the instance_classifiers.
        # For this prototype, we return the bag prediction.

        return logits, Y_prob, Y_hat, A, instance_loss
