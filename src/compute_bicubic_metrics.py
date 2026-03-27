from pathlib import Path
import numpy as np
from PIL import Image
from imageio import v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray

HR_DIR = Path("./data/SrBenchmark/MedDemo/HR")
LR_DIR = Path("./data/SrBenchmark/MedDemo/LR_bicubic/X4")
SCALE = 4
CROP = 4

def to_gray_u8(img):
    if img.ndim == 3:
        img = rgb2gray(img.astype(np.float32) / 255.0)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def crop(img, b):
    if b <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2*b or w <= 2*b:
        return img
    return img[b:h-b, b:w-b]

psnr_list, ssim_list = [], []
print("Bicubic baseline metrics:\n")

for hr_path in sorted(HR_DIR.glob("*")):
    if hr_path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp"]:
        continue

    stem = hr_path.stem
    lr_path = LR_DIR / f"{stem}x{SCALE}{hr_path.suffix}"
    if not lr_path.exists():
        print(f"[SKIP] LR not found for {hr_path.name} -> expected {lr_path.name}")
        continue

    hr = imageio.imread(hr_path)
    hr_h, hr_w = hr.shape[:2]

    lr_img = Image.open(lr_path).convert("RGB")
    bicubic = lr_img.resize((hr_w, hr_h), Image.BICUBIC)
    bicubic = np.array(bicubic)

    hr_g = to_gray_u8(hr)
    bi_g = to_gray_u8(bicubic)

    h = min(hr_g.shape[0], bi_g.shape[0])
    w = min(hr_g.shape[1], bi_g.shape[1])
    hr_g = crop(hr_g[:h, :w], CROP)
    bi_g = crop(bi_g[:h, :w], CROP)

    p = peak_signal_noise_ratio(hr_g, bi_g, data_range=255)
    s = structural_similarity(hr_g, bi_g, data_range=255)
    psnr_list.append(p)
    ssim_list.append(s)

    print(f"{hr_path.name}: PSNR={p:.4f}, SSIM={s:.4f}")

if psnr_list:
    print("\nSummary:")
    print(f"Images: {len(psnr_list)}")
    print(f"Average Bicubic PSNR: {np.mean(psnr_list):.4f}")
    print(f"Average Bicubic SSIM: {np.mean(ssim_list):.4f}")
else:
    print("No valid pairs found.")