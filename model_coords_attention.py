import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ModelCoordsAttention(nn.Module):

    def __init__(
        self,
        num_classes=128,
        pretrained=False,
        size=180,
        T=2.0,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        """
        ResNet18 + coordinate embedding + Transformer fixation integration.

        Input:
            images:
                (B,N,3,H,W)

            coords:
                (B,N,4)
                (x,y,dx,dy)

        Output:
            logits:
                (B,num_classes)
        """

        super().__init__()

        self.temperature = T
        self.stochastic = False

        # ---------------- Backbone ----------------

        weights = (
            ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        base = resnet18(weights=weights)

        self.backbone = nn.Sequential(
            *(list(base.children())[:-2])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        feature_dim = 512


        # ---------------- Coordinates ----------------

        self.coord_embed = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU()
        )


        # ---------------- Fixation token projection ----------------

        self.token_dim = 256

        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim + 64, self.token_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        # ---------------- Transformer ----------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        # ---------------- Attention pooling ----------------

        self.attention_pool = nn.Sequential(
            nn.Linear(self.token_dim,128),
            nn.Tanh(),
            nn.Linear(128,1)
        )


        # ---------------- Classifier ----------------

        self.fc1 = nn.Linear(
            self.token_dim,
            256
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )


    def forward(self, x, coords=None, return_rep=False):
        """
        x:
            (B,N,C,H,W)

        coords:
            (B,N,4)
        """

        B,N,C,H,W = x.shape


        # Flatten fixations through CNN

        x = x.reshape(
            B*N,
            C,
            H,
            W
        )

        feat_map = self.backbone(x)

        pooled = self.avgpool(feat_map)

        feat = pooled.flatten(1)
        # (B*N,512)


        # Coordinates

        if coords is None:

            coords = torch.zeros(
                B,
                N,
                4,
                device=feat.device,
                dtype=feat.dtype
            )


        coords = coords.reshape(
            B*N,
            4
        )

        coord_feat = self.coord_embed(coords)
        # (B*N,64)


        # Combine visual + location

        token = torch.cat(
            [
                feat,
                coord_feat
            ],
            dim=1
        )


        token = self.token_proj(token)

        token = token.reshape(
            B,
            N,
            self.token_dim
        )


        # -------------------------
        # Order-independent integration
        # -------------------------

        token = self.transformer(token)


        # -------------------------
        # Attention pooling
        # -------------------------

        weights = self.attention_pool(token)
        # (B,N,1)

        weights = torch.softmax(
            weights,
            dim=1
        )


        pooled_token = torch.sum(
            weights * token,
            dim=1
        )


        # -------------------------
        # Binary bottleneck
        # -------------------------

        logits_z = self.fc1(pooled_token)

        probs = torch.sigmoid(
            logits_z / self.temperature
        )


        if self.stochastic:
            h = torch.bernoulli(probs)
        else:
            h = probs


        logits = self.fc2(h)


        if return_rep:
            return logits, h, probs

        return logits