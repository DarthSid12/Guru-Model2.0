import torch
import torch.nn as nn
import torchvision.models as tvm

# Selectable conv backbones, ordered small -> large by param count. Each entry
# gives the torchvision constructor and how to peel off its classifier so what
# remains is a pure conv feature extractor emitting [B, C, h, w]. The trailing
# feature dim C is inferred with a dummy forward (see _build_backbone), so
# adding a backbone here needs no other change. resnet18 is the historical
# default (~11M params); mobilenet is the "smaller" run, resnet34/50 the
# "bigger" runs, convnext_tiny the modern-conv comparison.
# The two log-polar-aware variants. theta is the ROW axis of the log-polar
# image, so a rotation is a cyclic row-shift; see cylconv.py for why stock
# zero-padded, strided ResNets cannot exploit that.
#   resnet18_cyl       circular padding on theta (the seam fix, CyCNN)
#   resnet18_cylblur   that PLUS anti-aliased downsampling (Zhang 2019)
#   resnet18_blur      anti-aliased downsampling ONLY, zero padding kept and the
#                      blur reflect-padded: fixes aliasing without building the
#                      theta wrap (and with it rotation) into the architecture
# Both keep resnet18's parameter count -- BlurPool filters are fixed buffers.
BACKBONES = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, "features"),  # ~2.5M
    "resnet18":           (tvm.resnet18,           "resnet"),    # ~11M  (baseline)
    "resnet34":           (tvm.resnet34,           "resnet"),    # ~21M
    "resnet50":           (tvm.resnet50,           "resnet"),    # ~25M, C=2048
    "convnext_tiny":      (tvm.convnext_tiny,      "features"),  # ~28M, C=768
    "resnet18_cyl":       (tvm.resnet18,           "resnet"),    # + circular theta
    "resnet18_cylblur":   (tvm.resnet18,           "resnet"),    # + circular + blurpool
    "resnet18_blur":      (tvm.resnet18,           "resnet"),    # + blurpool only
    # Zhang 2019's own antialiased models (pip install antialiased-cnns), built
    # from the reference code so "antialiased ResNet-18 / VGG16-BN" means exactly
    # the published architecture: BlurPool after the ReLU, stem left strided
    # (pool_only=True), reflect padding, filter size 4 (the package default).
    "resnet18_aa":        (lambda weights=None: _aa("resnet18", weights), "resnet"),    # ~11M
    "vgg16_bn_aa":        (lambda weights=None: _aa("vgg16_bn", weights), "features"),  # ~15M conv, C=512
    "vgg16_bn_aa5":       (lambda weights=None: _aa("vgg16_bn", weights, filter_size=5), "features"),
    "vgg16_bn":           (tvm.vgg16_bn, "features"),
}


def _aa(arch, weights, filter_size=4):
    """An antialiased-cnns model, imported lazily so no other backbone needs it."""
    import antialiased_cnns
    return getattr(antialiased_cnns, arch)(pretrained=weights is not None,
                                         filter_size=filter_size)


def _build_backbone(name, pretrained):
    """Return (feature_extractor, out_channels) for a named torchvision model."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choices: {sorted(BACKBONES)}")
    ctor, kind = BACKBONES[name]
    base = ctor(weights="DEFAULT" if pretrained else None)
    if kind == "resnet":
        extractor = nn.Sequential(*list(base.children())[:-2])  # drop avgpool+fc
    else:  # mobilenet / convnext expose a conv stack as `.features`
        extractor = base.features
    if name in ("resnet18_cyl", "resnet18_cylblur"):
        from cylconv import cylindrify, antialias
        extractor = cylindrify(extractor)
        if name == "resnet18_cylblur":
            extractor = antialias(extractor)
    elif name == "resnet18_blur":
        from cylconv import antialias
        extractor = antialias(extractor, circular=False)
    with torch.no_grad():
        c = extractor(torch.zeros(1, 3, 180, 180)).shape[1]
    return extractor, c


class Model(nn.Module):
    def __init__(self, num_classes=128, pretrained=False, size=180, T=16.0, dropout=0.0,
                 backbone="resnet18"):
        """
        Conv backbone (selectable, see BACKBONES) feeding a 256-unit sigmoid
        bottleneck `h` (fc1 + temperature) and a linear classifier head fc2.
        The Yin/NIMBLE simulation reads the shared code `h`; the backbone width
        is the only thing that changes across --backbone runs.
        """
        super().__init__()

        self.temperature = T
        self.backbone_name = backbone
        self.stochastic = False  # training mode without sampling when stochastic=False vs sampling mode when stochastic=True

        # --------- Backbone (feature extractor) ---------
        self.backbone, self.in_size = _build_backbone(backbone, pretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --------- Classification head ---------
        self.fc1 = nn.Linear(self.in_size, 256)
        # dropout only on the classifier's input to fc2, not on the returned
        # binary code h itself (the Yin/NIMBLE simulation reads h at eval
        # time, when dropout is a no-op anyway, but keep it out of h in train
        # mode too so return_rep's "h" stays comparable across dropout settings)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x, return_rep=False):
        """
        Args:
            x : (B, 3, H, W)
        """
        feat_map = self.backbone(x)
        pooled = self.avgpool(feat_map)
        feat = pooled.view(pooled.size(0), -1)  # [B, in_size], in_size set by backbone
        # logistic units with temperature
        logits_z = self.fc1(feat)
        probs = torch.sigmoid(logits_z / self.temperature)

        # ---------- stochastic vs deterministic ----------
        if self.stochastic:
            h = torch.bernoulli(probs)
        else:
            h = probs  # deterministic expectation during training

        logits = self.fc2(self.dropout(h))
        if return_rep:
            return logits, h, probs # return h for variance analysis

        return logits  
