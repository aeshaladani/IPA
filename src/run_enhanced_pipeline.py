"""
Enhanced HSPAN Pipeline - HSPAN + Unsharp Mask
Complete pipeline for medical image super-resolution with postprocessing.

Pipeline stages:
    1. HSPAN super-resolution
    2. Unsharp mask postprocessing (edge sharpening)

Usage:
    python run_enhanced_pipeline.py --test_data MedDemo

This script runs THREE pipelines for comparison:
    - Baseline: Bicubic interpolation
    - HSPAN: Standard HSPAN only
    - Enhanced: HSPAN + Unsharp Mask
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

# Import HSPAN model components
import utility
import model as model_module

# Import postprocessing
from postprocessing.unsharp_mask import apply_unsharp_mask


def create_hspan_args(model_path, scale=4, device='cpu'):
    args = argparse.Namespace()
    args.model = 'HSPAN'
    args.scale = [scale]
    args.n_resgroups = 10
    args.n_resblocks = 4
    args.n_feats = 192
    args.reduction = 2
    args.topk = 128
    args.res_scale = 0.1
    args.rgb_range = 1
    args.n_colors = 3
    args.pre_train = str(model_path)
    args.cpu = (device == 'cpu')
    args.n_GPUs = 1
    args.self_ensemble = False
    args.chop = False
    args.precision = 'single'
    args.save_models = False
    args.resume = 0
    return args


def load_hspan_model(model_path, scale=4, device='cpu', base_args=None):
    """Load pretrained HSPAN model."""
    print(f"Loading HSPAN model from: {model_path}")

    if base_args is None:
        base_args = create_hspan_args(model_path, scale=scale, device=device)

    # Create dummy checkpoint
    class DummyCheckpoint:
        def __init__(self):
            self.dir = '.'
            self.log_file = sys.stdout
        def write_log(self, *args, **kwargs):
            pass

    checkpoint = DummyCheckpoint()

    # Load model
    _model = model_module.Model(base_args, checkpoint)
    _model.model.eval()

    return _model


def sr_with_hspan(lr_image, hspan_model, scale=4):
    """
    Super-resolve an image using HSPAN.
    
    Parameters
    lr_image : PIL.Image
        Low resolution input image
    hspan_model : Model
        Loaded HSPAN model
    scale : int
        Super-resolution scale factor
        
    Returns
    PIL.Image
        Super-resolved image
    """
    # Convert PIL to tensor
    lr_array = np.array(lr_image).astype(np.float32) / 255.0
    lr_tensor = torch.from_numpy(lr_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    # Run HSPAN
    with torch.no_grad():
        sr_tensor = hspan_model(lr_tensor, 0)  # idx_scale=0
        sr_tensor = sr_tensor.clamp(0, 1)
    
    # Convert back to PIL
    sr_array = sr_tensor.squeeze(0).permute(1, 2, 0).numpy()
    sr_array = (sr_array * 255.0).astype(np.uint8)
    sr_image = Image.fromarray(sr_array)
    
    return sr_image


def bicubic_upscale(lr_image, scale=4):
    """Baseline bicubic interpolation."""
    w, h = lr_image.size
    return lr_image.resize((w * scale, h * scale), Image.BICUBIC)


def run_pipeline(lr_dir, hr_dir, output_base_dir, model_path, scale=4, 
                 unsharp_amount=1.5, unsharp_radius=1.0):
    """
    Run all three pipelines and save results.
    
    Parameters
    lr_dir : Path
        Directory containing LR images
    hr_dir : Path
        Directory containing HR ground truth images
    output_base_dir : Path
        Base directory for saving results
    model_path : Path
        Path to HSPAN pretrained model
    scale : int
        Super-resolution scale factor
    unsharp_amount : float
        Unsharp mask strength
    unsharp_radius : float
        Unsharp mask radius
    """
    
    # Create output directories
    output_dirs = {
        'bicubic': output_base_dir / 'results_bicubic',
        'hspan': output_base_dir / 'results_hspan',
        'enhanced': output_base_dir / 'results_enhanced'
    }
    
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load HSPAN model
    device = 'cpu'  # Use CPU to avoid compatibility issues
    hspan_model = load_hspan_model(model_path, scale=scale, device=device)
    
    # Get LR images
    lr_files = sorted([f for f in lr_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    
    if not lr_files:
        print(f"ERROR: No LR images found in {lr_dir}")
        return
    
    print(f"\nFound {len(lr_files)} LR images")
    print(f"Pipeline: LR → HSPAN → UnsharpMask(amount={unsharp_amount}, radius={unsharp_radius}) → SR\n")
    
    # Process each image
    for lr_file in tqdm(lr_files, desc="Processing images"):
        stem = lr_file.stem.replace('x4', '').replace('x2', '').replace('x3', '')  # Remove scale suffix
        
        # Load LR image
        lr = Image.open(lr_file).convert('RGB')
        
        #PIPELINE 1: Bicubic Baseline
        sr_bicubic = bicubic_upscale(lr, scale=scale)
        sr_bicubic.save(output_dirs['bicubic'] / f"{stem}_x{scale}_SR.png")
        
        #PIPELINE 2: HSPAN Only
        sr_hspan = sr_with_hspan(lr, hspan_model, scale=scale)
        sr_hspan.save(output_dirs['hspan'] / f"{stem}_x{scale}_SR.png")
        
        #PIPELINE 3: Enhanced (HSPAN + Unsharp)
        # Super-resolve with HSPAN
        sr_enhanced = sr_with_hspan(lr, hspan_model, scale=scale)
        
        # Apply unsharp mask
        sr_enhanced = apply_unsharp_mask(sr_enhanced, amount=unsharp_amount, radius=unsharp_radius, threshold=0)
        
        sr_enhanced.save(output_dirs['enhanced'] / f"{stem}_x{scale}_SR.png")
    
    print(f"\n✓ Processing complete!")
    print(f"\nResults saved to:")
    print(f"  Bicubic:  {output_dirs['bicubic']}")
    print(f"  HSPAN:    {output_dirs['hspan']}")
    print(f"  Enhanced: {output_dirs['enhanced']}")
    
    print(f"\nNext step: Run compute_all_metrics.py to evaluate results")


def main():
    parser = argparse.ArgumentParser(description="Run enhanced HSPAN pipeline")
    parser.add_argument('--test_data', type=str, default='MedDemo', 
                        help='Test dataset name (e.g., MedDemo, Set5)')
    parser.add_argument('--scale', type=int, default=4, 
                        help='Super-resolution scale factor')
    parser.add_argument('--model_path', type=str, 
                        default='../experiment/HSPAN_x4/model/HSPAN_x4.pt',
                        help='Path to pretrained HSPAN model')
    parser.add_argument('--unsharp_amount', type=float, default=1.5,
                        help='Unsharp mask strength (0.5-3.0, default: 1.5)')
    parser.add_argument('--unsharp_radius', type=float, default=1.0,
                        help='Unsharp mask radius (0.5-2.0, default: 1.0)')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path('./data/SrBenchmark') / args.test_data
    lr_dir = data_dir / 'LR_bicubic' / f'X{args.scale}'
    hr_dir = data_dir / 'HR'
    output_base_dir = Path('../experiment') / f'enhanced_results_{args.test_data}'
    model_path = Path(args.model_path)
    
    # Validate paths
    if not lr_dir.exists():
        print(f"ERROR: LR directory not found: {lr_dir}")
        sys.exit(1)
    if not hr_dir.exists():
        print(f"ERROR: HR directory not found: {hr_dir}")
        sys.exit(1)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        output_base_dir=output_base_dir,
        model_path=model_path,
        scale=args.scale,
        unsharp_amount=args.unsharp_amount,
        unsharp_radius=args.unsharp_radius
    )


if __name__ == "__main__":
    main()