import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import warnings
import argparse
import os
import requests
from typing import Union
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

model_dict = {}
device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            "laion2B-s32B-b79K",
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=False,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )
        model_dict["model"] = model
        model_dict["preprocess_val"] = preprocess_val


initialize_model()
model = model_dict["model"]
preprocess_val = model_dict["preprocess_val"]

# check if the checkpoint exists
if not os.path.exists(root_path):
    os.makedirs(root_path)
cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map["v2.1"])

checkpoint = torch.load(cp, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
tokenizer = get_tokenizer("ViT-H-14")
model = model.to(device)
model.eval()


def score(
    img_path: Union[list, str, Image.Image],
    prompt: str,
) -> list:
    images = [preprocess_val(image) for image in img_path]
    images = torch.stack(images).to(device=device, non_blocking=True)
    text = tokenizer(prompt).to(device=device, non_blocking=True)
    with torch.cuda.amp.autocast():
        outputs = model(images, text)
        image_features, text_features = (
            outputs["image_features"],
            outputs["text_features"],
        )
        logits_per_image = image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return hps_score


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        nargs="+",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(root_path, "HPS_v2_compressed.pt"),
        help="Path to the model checkpoint",
    )

    args = parser.parse_args()

    hps_score = score(args.image_path, args.prompt, args.checkpoint)
    print("HPSv2 score:", hps_score)
