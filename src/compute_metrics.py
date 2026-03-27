import argparse
from pathlib import Path
import numpy as np
from imageio import v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray

def to_gray_uint8(img):
    if img.ndim == 3:
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0
        g = rgb2gray(img)
        g = np.clip(g * 255.0, 0, 255).astype(np.uint8)
        return g
    elif img.ndim == 2:
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

def crop_border(img, b):
    if b <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * b or w <= 2 * b:
        return img
    return img[b:h-b, b:w-b]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr_dir", type=str, required=True)
    parser.add_argument("--sr_dir", type=str, required=True)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop_border", type=int, default=4)  
    args = parser.parse_args()

    hr_dir = Path(args.hr_dir)
    sr_dir = Path(args.sr_dir)

    hr_files = sorted([p for p in hr_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
    if not hr_files:
        print("No HR images found.")
        return

    psnr_list = []
    ssim_list = []

    print("Per-image metrics:")
    for hr_path in hr_files:
        stem = hr_path.stem
        sr_name = f"{stem}_x{args.scale}_SR.png"
        sr_path = sr_dir / sr_name

        if not sr_path.exists():
            print(f"[SKIP] SR not found for {hr_path.name} -> expected {sr_name}")
            continue

        hr = imageio.imread(hr_path)
        sr = imageio.imread(sr_path)

        hr = to_gray_uint8(hr)
        sr = to_gray_uint8(sr)

        h = min(hr.shape[0], sr.shape[0])
        w = min(hr.shape[1], sr.shape[1])
        hr = hr[:h, :w]
        sr = sr[:h, :w]

        hr = crop_border(hr, args.crop_border)
        sr = crop_border(sr, args.crop_border)

        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, data_range=255)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

        print(f"{hr_path.name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")

    if len(psnr_list) == 0:
        print("\nNo valid HR-SR pairs found. Check naming/paths.")
        return

    print("\nSummary:")
    print(f"Images evaluated: {len(psnr_list)}")
    print(f"Average PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    main()