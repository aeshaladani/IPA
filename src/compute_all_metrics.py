"""
Compute and Compare Metrics for All Pipelines

Evaluates three pipelines:
    1. Bicubic interpolation (baseline)
    2. HSPAN only
    3. Enhanced (CLAHE + HSPAN + Unsharp Mask)

Generates comprehensive comparison table with PSNR and SSIM metrics.

Usage:
    python compute_all_metrics.py --test_data MedDemo
"""

import argparse
from pathlib import Path
import numpy as np
from imageio import v2 as imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.color import rgb2gray


def to_gray_uint8(img):
    """Convert image to grayscale uint8."""
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
    """Crop border pixels."""
    if b <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * b or w <= 2 * b:
        return img
    return img[b:h-b, b:w-b]


def compute_metrics_for_pipeline(hr_dir, sr_dir, scale=4, crop_border_size=4):
    """
    Compute PSNR and SSIM for a single pipeline.
    
    Returns
    -------
    dict
        Dictionary mapping image names to (psnr, ssim) tuples
    """
    hr_dir = Path(hr_dir)
    sr_dir = Path(sr_dir)
    
    hr_files = sorted([
        p for p in hr_dir.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
        and not p.stem.endswith('_clahe')
        and not p.stem.endswith('_sharp')
    ])
    
    results = {}
    
    for hr_path in hr_files:
        stem = hr_path.stem
        sr_name = f"{stem}_x{scale}_SR.png"
        sr_path = sr_dir / sr_name
        
        if not sr_path.exists():
            print(f"[SKIP] SR not found for {hr_path.name} -> expected {sr_name}")
            continue
        
        # Load images
        hr = imageio.imread(hr_path)
        sr = imageio.imread(sr_path)
        
        # Convert to grayscale
        hr = to_gray_uint8(hr)
        sr = to_gray_uint8(sr)
        
        # Match sizes
        h = min(hr.shape[0], sr.shape[0])
        w = min(hr.shape[1], sr.shape[1])
        hr = hr[:h, :w]
        sr = sr[:h, :w]
        
        # Crop borders
        hr = crop_border(hr, crop_border_size)
        sr = crop_border(sr, crop_border_size)
        
        # Compute metrics
        psnr = peak_signal_noise_ratio(hr, sr, data_range=255)
        ssim = structural_similarity(hr, sr, data_range=255)
        
        results[hr_path.name] = (psnr, ssim)
    
    return results


def print_comparison_table(bicubic_results, hspan_results, enhanced_results):
    """
    Print a comprehensive comparison table.
    """
    # Get all image names
    all_images = sorted(set(bicubic_results.keys()) | set(hspan_results.keys()) | set(enhanced_results.keys()))
    
    print("COMPREHENSIVE PIPELINE COMPARISON")
    print()
    
    # Header
    header = f"{'Image Name':<15} | {'Bicubic PSNR':>12} | {'HSPAN PSNR':>11} | {'Enhanced PSNR':>13} | {'Δ(H-B)':>8} | {'Δ(E-B)':>8} | {'Δ(E-H)':>8}"
    print(header)
    
    bicubic_psnrs = []
    hspan_psnrs = []
    enhanced_psnrs = []
    
    # Per-image results
    for img_name in all_images:
        bicubic_psnr, _ = bicubic_results.get(img_name, (np.nan, np.nan))
        hspan_psnr, _ = hspan_results.get(img_name, (np.nan, np.nan))
        enhanced_psnr, _ = enhanced_results.get(img_name, (np.nan, np.nan))
        
        delta_hspan_bicubic = hspan_psnr - bicubic_psnr if not np.isnan(hspan_psnr) and not np.isnan(bicubic_psnr) else np.nan
        delta_enhanced_bicubic = enhanced_psnr - bicubic_psnr if not np.isnan(enhanced_psnr) and not np.isnan(bicubic_psnr) else np.nan
        delta_enhanced_hspan = enhanced_psnr - hspan_psnr if not np.isnan(enhanced_psnr) and not np.isnan(hspan_psnr) else np.nan
        
        print(f"{img_name:<15} | {bicubic_psnr:12.4f} | {hspan_psnr:11.4f} | {enhanced_psnr:13.4f} | "
              f"{delta_hspan_bicubic:+8.4f} | {delta_enhanced_bicubic:+8.4f} | {delta_enhanced_hspan:+8.4f}")
        
        if not np.isnan(bicubic_psnr):
            bicubic_psnrs.append(bicubic_psnr)
        if not np.isnan(hspan_psnr):
            hspan_psnrs.append(hspan_psnr)
        if not np.isnan(enhanced_psnr):
            enhanced_psnrs.append(enhanced_psnr)
    
    # Averages
    avg_bicubic = np.mean(bicubic_psnrs) if bicubic_psnrs else np.nan
    avg_hspan = np.mean(hspan_psnrs) if hspan_psnrs else np.nan
    avg_enhanced = np.mean(enhanced_psnrs) if enhanced_psnrs else np.nan
    
    avg_delta_hspan = avg_hspan - avg_bicubic if not np.isnan(avg_hspan) and not np.isnan(avg_bicubic) else np.nan
    avg_delta_enhanced_bicubic = avg_enhanced - avg_bicubic if not np.isnan(avg_enhanced) and not np.isnan(avg_bicubic) else np.nan
    avg_delta_enhanced_hspan = avg_enhanced - avg_hspan if not np.isnan(avg_enhanced) and not np.isnan(avg_hspan) else np.nan
    
    print(f"{'AVERAGE':<15} | {avg_bicubic:12.4f} | {avg_hspan:11.4f} | {avg_enhanced:13.4f} | "
          f"{avg_delta_hspan:+8.4f} | {avg_delta_enhanced_bicubic:+8.4f} | {avg_delta_enhanced_hspan:+8.4f}")
    
    # SSIM Table
    print("\nSSIM COMPARISON")
    
    header_ssim = f"{'Image Name':<15} | {'Bicubic SSIM':>12} | {'HSPAN SSIM':>11} | {'Enhanced SSIM':>13} | {'Δ(E-H)':>8}"
    print(header_ssim)
    
    bicubic_ssims = []
    hspan_ssims = []
    enhanced_ssims = []
    
    for img_name in all_images:
        _, bicubic_ssim = bicubic_results.get(img_name, (np.nan, np.nan))
        _, hspan_ssim = hspan_results.get(img_name, (np.nan, np.nan))
        _, enhanced_ssim = enhanced_results.get(img_name, (np.nan, np.nan))
        
        delta_enhanced_hspan_ssim = enhanced_ssim - hspan_ssim if not np.isnan(enhanced_ssim) and not np.isnan(hspan_ssim) else np.nan
        
        print(f"{img_name:<15} | {bicubic_ssim:12.4f} | {hspan_ssim:11.4f} | {enhanced_ssim:13.4f} | {delta_enhanced_hspan_ssim:+8.4f}")
        
        if not np.isnan(bicubic_ssim):
            bicubic_ssims.append(bicubic_ssim)
        if not np.isnan(hspan_ssim):
            hspan_ssims.append(hspan_ssim)
        if not np.isnan(enhanced_ssim):
            enhanced_ssims.append(enhanced_ssim)
    
    avg_bicubic_ssim = np.mean(bicubic_ssims) if bicubic_ssims else np.nan
    avg_hspan_ssim = np.mean(hspan_ssims) if hspan_ssims else np.nan
    avg_enhanced_ssim = np.mean(enhanced_ssims) if enhanced_ssims else np.nan
    avg_delta_ssim = avg_enhanced_ssim - avg_hspan_ssim if not np.isnan(avg_enhanced_ssim) and not np.isnan(avg_hspan_ssim) else np.nan
    
    print(f"{'AVERAGE':<15} | {avg_bicubic_ssim:12.4f} | {avg_hspan_ssim:11.4f} | {avg_enhanced_ssim:13.4f} | {avg_delta_ssim:+8.4f}")
    
    # Summary
    print("\nSUMMARY")
    print(f"Images evaluated: {len(bicubic_psnrs)}")
    print()
    print(f"Average PSNR improvement (HSPAN over Bicubic):    {avg_delta_hspan:+.4f} dB")
    print(f"Average PSNR improvement (Enhanced over Bicubic): {avg_delta_enhanced_bicubic:+.4f} dB")
    print(f"Average PSNR improvement (Enhanced over HSPAN):   {avg_delta_enhanced_hspan:+.4f} dB")
    print()
    print(f"Average SSIM improvement (Enhanced over HSPAN):   {avg_delta_ssim:+.4f}")
    print()
    
    print("Legend:")
    print("  Bicubic  = Baseline bicubic interpolation")
    print("  HSPAN    = Standard HSPAN super-resolution")
    print("  Enhanced = HSPAN + Unsharp Mask (Gonzalez methods)")
    print("  Δ(H-B)   = HSPAN improvement over Bicubic")
    print("  Δ(E-B)   = Enhanced improvement over Bicubic")
    print("  Δ(E-H)   = Enhanced improvement over HSPAN")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compute metrics for all pipelines")
    parser.add_argument('--test_data', type=str, default='MedDemo',
                        help='Test dataset name (e.g., MedDemo, Set5)')
    parser.add_argument('--scale', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--crop_border', type=int, default=4,
                        help='Number of border pixels to crop')
    
    args = parser.parse_args()
    
    # Setup paths
    hr_dir = Path('./data/SrBenchmark') / args.test_data / 'HR'
    results_base = Path('../experiment') / f'enhanced_results_{args.test_data}'
    
    bicubic_dir = results_base / 'results_bicubic'
    hspan_dir = results_base / 'results_hspan'
    enhanced_dir = results_base / 'results_enhanced'
    
    # Validate paths
    if not hr_dir.exists():
        print(f"ERROR: HR directory not found: {hr_dir}")
        return
    
    for dir_path in [bicubic_dir, hspan_dir, enhanced_dir]:
        if not dir_path.exists():
            print(f"ERROR: Results directory not found: {dir_path}")
            print("Run run_enhanced_pipeline.py first!")
            return
    
    # Compute metrics for each pipeline
    print("Computing metrics for Bicubic...")
    bicubic_results = compute_metrics_for_pipeline(hr_dir, bicubic_dir, args.scale, args.crop_border)
    
    print("Computing metrics for HSPAN...")
    hspan_results = compute_metrics_for_pipeline(hr_dir, hspan_dir, args.scale, args.crop_border)
    
    print("Computing metrics for Enhanced...")
    enhanced_results = compute_metrics_for_pipeline(hr_dir, enhanced_dir, args.scale, args.crop_border)
    
    # Print comparison table
    print_comparison_table(bicubic_results, hspan_results, enhanced_results)


if __name__ == "__main__":
    main()