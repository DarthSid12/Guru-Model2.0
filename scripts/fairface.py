"""FairFace race classifier (Karkkainen & Joo, WACV 2021), shared by the VGGFace2
race-labelling scripts.

Weights: res34_fair_align_multi_7_20190809.pt, the official 7-race model, taken
from the anning01/fairface Hugging Face re-upload because the official Google
Drive folder refuses programmatic listing. It is a plain state dict (loaded with
weights_only=True, so it cannot execute code) that matches torchvision ResNet-34
with an 18-way head exactly: logits 0-6 race, 7-8 gender, 9-17 age.
sha256 37f2e74c0e7da196f8edd7dae0c88cffdbef7923beebebcc27381f0ad0fdb2c1.

Preprocessing follows FairFace's predict.py after its dlib chip step: resize to
224, ImageNet normalisation. We feed loose face crops rather than dlib-aligned
chips; validate_on_rfw() measures what that costs.
"""
import torch, torchvision
import torchvision.transforms as T

WEIGHTS = "data/vggface2_raw/fairface/res34_fair_align_multi_7_20190809.pt"
RACES = ["White", "Black", "Latino_Hispanic", "East Asian",
         "Southeast Asian", "Indian", "Middle Eastern"]
PREP = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                  T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load(device):
    m = torchvision.models.resnet34()
    m.fc = torch.nn.Linear(m.fc.in_features, 18)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    return m.to(device).eval()


@torch.no_grad()
def race_probs(model, pil_images, device):
    """(N, 7) softmax over the 7 race classes, in RACES order."""
    x = torch.stack([PREP(im.convert("RGB")) for im in pil_images]).to(device)
    return torch.softmax(model(x)[:, :7].float(), dim=1).cpu()
