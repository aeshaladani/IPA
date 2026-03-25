from PIL import Image
import os

def main():
    # Paths relative to HSPAN/src/
    hr_dir = './data/SrBenchmark/Set5/HR'
    lr_dir = './data/SrBenchmark/Set5/LR_bicubic/X4'
    scale = 4
    
    # Create LR directory if it doesn't exist
    os.makedirs(lr_dir, exist_ok=True)
    
    print("=" * 60)
    print("Step 1: Converting HR images to RGB (removing alpha channel)")
    print("=" * 60)
    
    # First, convert all HR images to RGB
    for f in os.listdir(hr_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            hr_path = os.path.join(hr_dir, f)
            img = Image.open(hr_path)
            
            # Check if conversion needed
            if img.mode != 'RGB':
                print(f"Converting {f} from {img.mode} to RGB")
                img = img.convert('RGB')
                img.save(hr_path)
            else:
                print(f"✓ {f} already RGB")
    
    print("\n" + "=" * 60)
    print("Step 2: Creating LR images (downscaled by x{})".format(scale))
    print("=" * 60)
    
    # Create LR images
    for f in os.listdir(hr_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            hr_path = os.path.join(hr_dir, f)
            img = Image.open(hr_path).convert('RGB')
            
            w, h = img.size
            lr = img.resize((w // scale, h // scale), Image.BICUBIC)
            
            name, ext = os.path.splitext(f)
            lr_filename = name + 'x4' + ext
            lr_path = os.path.join(lr_dir, lr_filename)
            
            lr.save(lr_path)
            print(f"Created: {lr_filename} ({w}x{h} → {w//scale}x{h//scale})")
    
    print("\n" + "=" * 60)
    print(" Done! Set5 is ready for testing.")
    print("=" * 60)
    print("\nNext step: Run the test command")
    print("cd to src/ and run:")
    print("python main.py --dir_data ./data --data_test Set5 ...")

if __name__ == "__main__":
    main()