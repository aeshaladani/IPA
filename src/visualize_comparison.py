"""
Visual Comparison of All Pipelines
===================================
Creates side-by-side comparison images showing:
    - LR input
    - Bicubic upscale
    - HSPAN only
    - Enhanced (CLAHE + HSPAN + Unsharp)
    - HR ground truth

Usage:
    python visualize_comparison.py --test_data MedDemo --image_name img001.png
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from PIL import Image
import numpy as np


def create_comparison_figure(lr_path, bicubic_path, hspan_path, enhanced_path, hr_path, output_path):
    """
    Create a side-by-side comparison figure.
    
    Parameters
    ----------
    lr_path : Path
        Path to LR image
    bicubic_path : Path
        Path to bicubic SR image
    hspan_path : Path
        Path to HSPAN SR image
    enhanced_path : Path
        Path to enhanced SR image
    hr_path : Path
        Path to HR ground truth
    output_path : Path
        Where to save the comparison figure
    """
    
    # Load images
    lr = Image.open(lr_path) if lr_path.exists() else None
    bicubic = Image.open(bicubic_path) if bicubic_path.exists() else None
    hspan = Image.open(hspan_path) if hspan_path.exists() else None
    enhanced = Image.open(enhanced_path) if enhanced_path.exists() else None
    hr = Image.open(hr_path) if hr_path.exists() else None
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Pipeline Comparison: {lr_path.name}', fontsize=16, fontweight='bold')
    
    # Plot images
    images = [
        (lr, 'LR Input', axes[0, 0]),
        (bicubic, 'Bicubic Upscale', axes[0, 1]),
        (hr, 'HR Ground Truth', axes[0, 2]),
        (hspan, 'HSPAN Only', axes[1, 0]),
        (enhanced, 'Enhanced\n(CLAHE + HSPAN + Unsharp)', axes[1, 1]),
    ]
    
    for img, title, ax in images:
        if img is not None:
            ax.imshow(np.array(img))
            ax.set_title(title, fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    # Add text annotation in the empty subplot
    axes[1, 2].text(0.5, 0.5, 
                    'Enhanced Pipeline:\n\n'
                    '1. CLAHE preprocessing\n'
                    '   (Gonzalez Ch. 3)\n\n'
                    '2. HSPAN SR\n\n'
                    '3. Unsharp masking\n'
                    '   (Gonzalez Ch. 3.6)',
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline comparison")
    parser.add_argument('--test_data', type=str, default='MedDemo',
                        help='Test dataset name')
    parser.add_argument('--image_name', type=str, required=True,
                        help='Image name to visualize (e.g., img001.png)')
    parser.add_argument('--scale', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for comparison images')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path('./data/SrBenchmark') / args.test_data
    results_base = Path('../experiment') / f'enhanced_results_{args.test_data}'
    
    stem = Path(args.image_name).stem
    
    # Input and output paths
    lr_path = data_dir / 'LR_bicubic' / f'X{args.scale}' / args.image_name.replace('.png', f'x{args.scale}.png')
    hr_path = data_dir / 'HR' / args.image_name
    
    bicubic_path = results_base / 'results_bicubic' / f"{stem}_x{args.scale}_SR.png"
    hspan_path = results_base / 'results_hspan' / f"{stem}_x{args.scale}_SR.png"
    enhanced_path = results_base / 'results_enhanced' / f"{stem}_x{args.scale}_SR.png"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_base / 'comparisons'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_comparison.png"
    
    # Create comparison
    create_comparison_figure(lr_path, bicubic_path, hspan_path, enhanced_path, hr_path, output_path)
    
    print(f"\n✓ Comparison saved to: {output_path}")


if __name__ == "__main__":
    main()