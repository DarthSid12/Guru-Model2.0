import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ModelCoords(nn.Module):
    def __init__(self, num_classes=128, pretrained=False, size=180, T=16.0):
        """
        ResNet18 backbone with two heads:
          • classifier head  → CE loss
          • projection head  → SupCon loss
        """
        super().__init__()

        # # --------- Backbone feature dim (ResNet18 → 512) ---------
        self.in_size = 512
        self.temperature = T
        self.stochastic = False  # training mode without sampling when stochastic=False vs sampling mode when stochastic=True

        # --------- Backbone (feature extractor) --------- ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        self.backbone = nn.Sequential(*(list(base.children())[:-2]))  # conv layers only
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------- Classification head --------- ResNet18

        # add coordinates of fixation crop as input
        # input: (x,y, dx, dy) - both actual coords and difference from previous fixation given; dx,dy are 0 for the first fixation
        self.coord_embed = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(self.in_size + 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x, coords=None, return_rep=False):
        """
        Args:
            x: (B,3,H,W)
            coords: (B,2), normalized fixation coordinates (x,y) in [0,1]
        """

        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map)

        feat = pooled.view(pooled.size(0), -1)

        # in case no fixation crop given for convolutional case; otherwise we pass in the fixation crop coords to forward
        if coords is None:
            coords = torch.zeros(
                feat.size(0),
                4,
                device=feat.device,
                dtype=feat.dtype,
            )

        coord_feat = self.coord_embed(coords)
        feat = torch.cat([feat, coord_feat], dim=1)
        logits_z = self.fc1(feat)

        # logistic units with temperature        
        probs = torch.sigmoid(logits_z / self.temperature)

        # ---------- stochastic vs deterministic ----------
        if self.stochastic:
            h = torch.bernoulli(probs)
        else:
            h = probs  # deterministic expectation during training

        logits = self.fc2(h)
        if return_rep:
            return logits, h, probs # return h for variance analysis

        return logits  
