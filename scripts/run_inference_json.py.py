#!/usr/bin/env python3
"""
Run inference on an image directory to predict AIGC generation likelihood.

This script processes an input folder of images, performs preprocessing and 
model inference, and saves the confidence scores to a structured JSON file.

Parameters:
    --image_dir (str): 
        Path to the target directory containing images to evaluate (supports 
        .jpg, .jpeg, .png, .webp, .bmp).
    --output_json (str, optional): 
        Path where the output JSON file will be saved. Default: "predictions.json".
    --checkpoint (str, optional): 
        Path to the trained model checkpoint file (.pt or .pth). 
        Default: "checkpoints/baseline_best.pt".
    --batch_size (int, optional): 
        Number of images to process per batch during inference. Default: 32.
    --num_workers (int, optional): 
        Number of subprocesses to use for data loading. Default: 2.

Output:
    A JSON file containing a list of dictionaries, structured as:
    [
        {
            "image_path": str,  # Path to the processed image
            "pred": float       # AIGC probability/confidence score in range [0.0, 1.0]
        },
        ...
    ]
"""

import argparse
import json
import os
from pathlib import Path
import sys
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

# Ensure project root is available in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DirectoryImageDataset(Dataset):
    """Recursively scans and loads images from a target folder."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        # Collect all valid image paths recursively
        self.image_paths = sorted(
            [
                p
                for p in self.root_dir.rglob("*")
                if p.suffix.lower() in VALID_EXTENSIONS
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, str(img_path), True
        except Exception:
            # Fallback zero-tensor for corrupted or unreadable images
            return torch.zeros((3, 224, 224)), str(img_path), False


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device):
    """Loads detector architecture and checkpoint weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    # 1. Resolve model builder or class from evaluate.py if present
    model = None
    try:
        import evaluate

        for func_name in ["build_model", "get_model", "load_model"]:
            if hasattr(evaluate, func_name):
                builder = getattr(evaluate, func_name)
                model = (
                    builder(str(checkpoint_path))
                    if func_name == "load_model"
                    else builder()
                )
                break
    except Exception:
        pass

    # 2. Check if checkpoint is an exported nn.Module
    if model is None and isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif model is None:
        raise RuntimeError(
            "Could not automatically detect the model architecture. "
            "Please ensure evaluate.py is in the root directory or import your model class."
        )

    # 3. Load state_dict if not already wrapped
    if not isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Generate AIGC likelihood confidence scores for an image folder."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory containing images to evaluate",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="predictions.json",
        help="Path where the output JSON file will be saved",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/baseline_best.pt",
        help="Path to trained model weights (.pt or .pth)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader worker count",
    )
    args = parser.parse_args()

    input_path = Path(args.image_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Image directory does not exist: {input_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard preprocessing pipeline
    preprocess = T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    dataset = DirectoryImageDataset(input_path, transform=preprocess)
    print(f"Discovered {len(dataset)} valid images in {input_path}")

    if len(dataset) == 0:
        print("No images found. Exiting.")
        return

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = load_model_from_checkpoint(Path(args.checkpoint), device)

    predictions = []
    print("Running inference...")

    with torch.no_grad():
        for batch_images, batch_paths, valid_flags in loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)

            # Handle multi-output models (e.g., tuples containing auxiliary heads or gate weights)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Convert logits to probability scores: P(AIGC) in [0.0, 1.0]
            probs = torch.sigmoid(outputs).squeeze().cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]

            for path, prob, is_valid in zip(batch_paths, probs, valid_flags):
                score = round(float(prob), 5) if is_valid else 0.0
                predictions.append(
                    {
                        "image_path": path,
                        "pred": score,  # Confidence score indicating likelihood of being AIGC
                    }
                )

    # Save to requested JSON format
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Successfully generated predictions for {len(predictions)} images.")
    print(f"Output saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()