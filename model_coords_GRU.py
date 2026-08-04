import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

"""
Processes fixations sequentially
Modelled using GRU to represent working memory
"""

class ModelCoordsGRU(nn.Module):
    def __init__(
        self,
        num_classes=128,
        pretrained=False,
        size=180,
        T=16.0,
        hidden_dim=256,
    ):
        super().__init__()

        self.temperature = T
        self.stochastic = False

        # ---------------- ResNet backbone ----------------

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        self.backbone = nn.Sequential(*(list(base.children())[:-2]))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        backbone_dim = 512

        # ---------------- Coordinate network ----------------

        self.coord_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ---------------- Sequential memory ----------------

        self.gru = nn.GRU(
            input_size=backbone_dim + 32,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ---------------- Classification head ----------------

        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, images, coords=None, return_rep=False):
        """
        images : (B,N,C,H,W)
        coords : (B,N,4)
        """

        B, N, C, H, W = images.shape

        # --------------------------------------------------
        # ResNet on every fixation
        # --------------------------------------------------

        x = images.reshape(B * N, C, H, W)

        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map)

        visual_feat = pooled.view(B * N, -1)

        # --------------------------------------------------
        # Coordinates
        # --------------------------------------------------

        if coords is None:
            coords = torch.zeros(
                B,
                N,
                4,
                device=images.device,
                dtype=images.dtype,
            )

        coords = coords.reshape(B * N, 4)

        coord_feat = self.coord_embed(coords)

        # --------------------------------------------------
        # Merge streams
        # --------------------------------------------------

        feat = torch.cat([visual_feat, coord_feat], dim=1)

        feat = feat.reshape(B, N, -1)

        # --------------------------------------------------
        # Sequential integration
        # --------------------------------------------------

        _, hidden = self.gru(feat)

        hidden = hidden.squeeze(0)

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------

        logits_z = self.fc1(hidden)

        probs = torch.sigmoid(logits_z / self.temperature)

        if self.stochastic:
            h = torch.bernoulli(probs)
        else:
            h = probs

        logits = self.fc2(h)

        if return_rep:
            return logits, h, probs

        return logits