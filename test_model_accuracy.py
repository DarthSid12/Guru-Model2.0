import torch

import torchvision.transforms.functional as TF
from PIL import Image

import numpy as np

from pathlib import Path
import os
import argparse

from model import Model
from utils import list_classes

import re # checking end of filepath

# image file possible ends - ONLY INCLUDE THE FIRST CROP to get the name of the single overall image only once
IMG_EXTS = ("_proc0.jpg", "_proc0.jpeg", "_proc0.png", "_proc0.bmp", "_proc0.webp")

DATA_PATH = "processed_data"

# maps each category to the folder in data associated with i
CAT_DIR_MAP = {
    "faces": "faces"
}

VARIANT = None # cnn or lp - determined by command line input

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_accuracy(model, image_map, category_map):
    """
    image map has image_pth to category structure
    category_map is really just a list - the jth logit corresponds to category j
    """

    correct = 0
    total = 0

    model.eval()
  
    with torch.no_grad():
        for i in image_map:
            path = [Path(str(i) + f"_proc{x}.png") for x in range(32)]
   
            imgs = torch.stack([
                TF.to_tensor(Image.open(p).convert("RGB"))
                for p in path
            ], dim=0)

            logits = sum(model(imgs.to(device))) # sum all the fixation crops / just one image for "cnn" case
            #print("logits:", logits, len(logits))
            #print("logits top ten:", torch.topk(logits, 10))

            pred = torch.argmax(logits)

            print("category:", path[0], pred, category_map[pred], image_map[i])

            if category_map[torch.argmax(logits)] == image_map[i]:
                print("correct")
                correct += 1
            else:
                print("incorrect")
                correct += 0

            total += 1

    return correct / total

# for consistency, order of categories list will have to be normalized
def create_maps(categories):
    image_map = {}
    category_map = []
    for c in categories:
        update_maps_with_category(c, image_map, category_map)
    return image_map, category_map

def update_maps_with_category(category, image_map, category_map):

    #print(category_map)

    cat_path = os.path.join(DATA_PATH, CAT_DIR_MAP[category], VARIANT, "valid") 
    # get path to data - test data for test accurcay, of course

    #print(cat_path)

    for identity in list_classes(cat_path):
        id_path = os.path.join(cat_path, identity)

        if os.path.isdir(id_path):    
            # update image_map
            for image in os.listdir(id_path):
                #print(image)   
                if image.lower().endswith(IMG_EXTS):          
                    img_path = os.path.join(id_path, str(image).split("_proc0")[0])
                    #print("img_path:", img_path)
                    image_map[img_path] = identity

            # update category_map - much easier
            category_map.append(identity)

def load_model(checkpoint_path, num_classes, temperature):
    model = Model(size=180, num_classes=num_classes, pretrained=False, T=temperature).to(device)
    result = model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)

    print(result)

    return model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--categories",
        nargs = '+',
        default=["faces"],
        help="categories of model we test on"
    )

    parser.add_argument(
        "--variant",
        default="lp",
        choices=["lp", "cnn"],
        help="type of image"
    )

    parser.add_argument(
        "--checkpoint",
        default="runs/FACES_Yin/pretrainTrue_RESNET18_FACES_LPNet_32_fixations_128_identities_40_epoch/resnet18_20260506_062829.pth",
        help="model weights"
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=128,
        help="number of identities"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="model temperature"
    )

    return parser.parse_args()
 
def main():
    args = parse_args()

    global VARIANT
    VARIANT = args.variant

    model = load_model(args.checkpoint, args.num_classes, args.temperature)
    image_map, category_map = create_maps(args.categories)

    print(get_accuracy(model, image_map, category_map))

if __name__ == "__main__":
    main()

