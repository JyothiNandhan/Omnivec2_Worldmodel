import os
import argparse
import requests
from io import BytesIO
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from config import IMG_SIZE, PATCH_SIZE
from model import OmniVec2Stage1
from rgb.patches import patchify, unpatchify

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_image(url):
    print(f"Downloading image from {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

def preprocess_image(img):
    # Standard nuScenes / ImageNet normalization could be applied if your dataloader does so.
    # But based on `visualize.py` the model expects inputs in [0, 1] range to patchify.
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(DEVICE) # (1, 3, 224, 224)

@torch.no_grad()
def main(args):
    # 1. Load model
    print(f"Loading Stage 1 model from {args.checkpoint}...")
    model = OmniVec2Stage1().to(DEVICE)
    
    # Load weights (handles both full bundles and raw state dicts)
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    # 2. Get image
    img_pil = download_image(args.url)
    img_tensor = preprocess_image(img_pil)
    
    # 3. Forward pass
    print(f"Running reconstruction with mask ratio {args.mask_ratio}...")
    pred, target, mask = model.forward_rgb(img_tensor, mask_ratio=args.mask_ratio)
    
    # 4. Decode
    raw_patches = patchify(img_tensor)
    mean = raw_patches.mean(dim=-1, keepdim=True)
    var = raw_patches.var(dim=-1, keepdim=True)
    pred_pixels = pred * (var + 1e-6).sqrt() + mean

    recon_patches = raw_patches.clone()
    mask_exp = mask.unsqueeze(-1).expand_as(recon_patches).bool()
    recon_patches[mask_exp] = pred_pixels[mask_exp]
    recon_img_tensor = unpatchify(recon_patches)[0] # (3, 224, 224)
    
    masked_patches = raw_patches.clone()
    masked_patches[mask_exp] = 0.5 # grey out masked patches
    masked_img_tensor = unpatchify(masked_patches)[0]
    
    # 5. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    axes[0].imshow(img_tensor[0].cpu().permute(1, 2, 0).numpy())
    axes[0].set_title("Original (Resized to 224x224)")
    axes[0].axis("off")
    
    # Masked
    axes[1].imshow(masked_img_tensor.cpu().permute(1, 2, 0).numpy())
    axes[1].set_title(f"Masked ({args.mask_ratio*100:.0f}%)")
    axes[1].axis("off")
    
    # Reconstruction
    axes[2].imshow(recon_img_tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    axes[2].set_title("Model Reconstruction")
    axes[2].axis("off")
    
    plt.suptitle("OmniVec2 Stage 1 — Web Image Reconstruction Test", fontsize=16)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True, help="URL of the image to download")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your trained Stage 1 .pth file")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Masking ratio (0.0 to 1.0)")
    parser.add_argument("--output", type=str, default="web_reconstruction.png", help="Output PNG path")
    args = parser.parse_args()
    main(args)
