"""
Unsharp Masking for Edge Enhancement
Based on: Gonzalez & Woods - Digital Image Processing, Chapter 3.6
          "Sharpening Spatial Filters"

Purpose: Sharpen edges in super-resolved medical images to enhance
         anatomical boundaries and diagnostic features.

Algorithm:
    1. Blur the image with Gaussian filter (creates "unsharp" version)
    2. Subtract blurred from original to get edge mask
    3. Add weighted mask back to original: sharp = original + amount × mask

Usage:
    from postprocessing.unsharp_mask import apply_unsharp_mask
    sharpened_sr = apply_unsharp_mask(sr_image, amount=1.5, radius=1.0)
"""

import cv2
import numpy as np
from PIL import Image


def apply_unsharp_mask(image, amount=1.5, radius=1.0, threshold=0):
    """
    Apply unsharp masking to sharpen image edges.
    
    Parameters
    image : PIL.Image or np.ndarray
        Input image (RGB or grayscale)
    amount : float
        Strength of sharpening (default: 1.5)
        - 1.0 = subtle sharpening
        - 1.5 = moderate sharpening (recommended for medical images)
        - 2.0+ = aggressive sharpening (may introduce artifacts)
    radius : float
        Gaussian blur radius in pixels (default: 1.0)
        - Smaller = sharpen fine details
        - Larger = sharpen broader edges
    threshold : int
        Minimum brightness change to sharpen (default: 0)
        - 0 = sharpen all edges
        - Higher = only sharpen strong edges (reduces noise amplification)
        
    Returns
    PIL.Image
        Sharpened image
        
    Notes
    - For medical images, amount=1.5 and radius=1.0 provide good balance
    - Threshold > 0 prevents noise amplification in smooth regions
    - Works in RGB or grayscale
    
    References
    Gonzalez & Woods (2018), Section 3.6.2: Unsharp Masking and Highboost Filtering
    """
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image).astype(np.float32)
    else:
        img_array = image.astype(np.float32)
    
    # Create Gaussian blur (the "unsharp" version)
    # sigma = radius for Gaussian kernel
    sigma = radius
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)  # Auto-calculate kernel size
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
    
    # Calculate edge mask (difference between original and blurred)
    mask = img_array - blurred
    
    # Apply threshold if specified
    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0, mask)
    
    # Add weighted mask to original: sharpened = original + amount × mask
    sharpened = img_array + amount * mask
    
    # Clip to valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Convert back to PIL
    return Image.fromarray(sharpened)


def apply_unsharp_mask_batch(images, amount=1.5, radius=1.0, threshold=0):
    """
    Apply unsharp masking to a batch of images.
    
    Parameters
    images : list of PIL.Image
        List of input images
    amount : float
        Sharpening strength
    radius : float
        Blur radius
    threshold : int
        Edge threshold
        
    Returns
    list of PIL.Image
        List of sharpened images
    """
    return [apply_unsharp_mask(img, amount, radius, threshold) for img in images]


if __name__ == "__main__":
    # Test the unsharp mask
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python unsharp_mask.py <image_path> [amount] [radius]")
        print("Example: python unsharp_mask.py test_image.png 1.5 1.0")
        sys.exit(1)
    
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: Image not found: {img_path}")
        sys.exit(1)

    # Parse optional parameters
    amount = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    
    # Load image
    img = Image.open(img_path)
    print(f"Loaded: {img_path.name} ({img.size[0]}x{img.size[1]})")
    print(f"Applying unsharp mask: amount={amount}, radius={radius}")
    
    # Apply unsharp mask
    sharpened = apply_unsharp_mask(img, amount=amount, radius=radius)
    
    # Save result
    out_path = img_path.parent / f"{img_path.stem}_sharp{img_path.suffix}"
    sharpened.save(out_path)
    print(f"Saved sharpened image: {out_path}")