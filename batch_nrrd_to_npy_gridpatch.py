import os
import numpy as np
import SimpleITK as sitk
from glob import glob

def resample_to_isotropic(image, spacing=(2.0, 2.0, 2.0), interpolator=sitk.sitkNearestNeighbor):
    original_spacing = image.GetSpacing()
    print(f"Original Spacing: {original_spacing}")
    original_size = image.GetSize()
    print(f"Original Size: {original_size}")
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
    ]
    print(f"New Size: {new_size}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())

    return resampler.Execute(image)

def extract_patches(image_np, patch_size=(64, 64, 64), stride=(64, 64, 64)):
    patches = []
    H, W, D = image_np.shape
    pH, pW, pD = patch_size
    for h in range(0, H - pH + 1, stride[0]):
        for w in range(0, W - pW + 1, stride[1]):
            for d in range(0, D - pD + 1, stride[2]):
                patch = image_np[h:h + pH, w:w + pW, d:d + pD]
                patches.append(patch)
    return patches

# Enter your Paths, It is highly recommended to have input and output in different drives, based on experience...
input_folder = "C:\\NRRD\\train"
output_folder = "D:\\NPY\\train"
os.makedirs(output_folder, exist_ok=True)

# File list
print(f"Scanning folder: {input_folder}")
nrrd_files = glob(os.path.join(input_folder, "*.nrrd"))
print(f"Found {len(nrrd_files)} NRRD files.")
input("Press Enter to start processing...")

# Process each file
for nrrd_path in nrrd_files:
    image_sitk = sitk.ReadImage(nrrd_path) #H,W,D


    ###Enter your desired spacing down below function

    resampled = resample_to_isotropic(image_sitk, spacing=(2.0, 2.0, 2.0))

    # For some reason GetArrayFromImage converts H,W,D shape to D,H,W. However, Slicer also uses np arrays and this order so i will stick with it
    img = sitk.GetArrayFromImage(resampled)
    print(f"{os.path.basename(nrrd_path)} → shape: {img.shape}")

    # Crop the center 256x256 section in the (H, W) plane. If you want difference size simply change half_crop

    h, w, d = img.shape #here h,w,d is actually d,h,w but i am a little bit lazy to change it. also it may mess up the fx, so let it stay that way
    center_w, center_d = w // 2, d // 2
    half_crop = 128  

    # Ensure we don't go out of bounds
    start_w = max(center_w - half_crop, 0)
    end_w = start_w + half_crop*2
    start_d = max(center_d - half_crop, 0)
    end_d = start_d + half_crop*2

    # Crop only H and W, keep full depth, again w is h and d is w actually but thats okay

    img = img[:,start_w:end_w, start_d:end_d]
    print(f"Cropped image shape: {img.shape}")




    # Patch extraction, you can change your patch and stride size...
    patches = extract_patches(img, patch_size=(64, 64, 64), stride=(64, 64, 64))
    print(f"{os.path.basename(nrrd_path)}: {len(patches)} patches extracted.")

    base = os.path.splitext(os.path.basename(nrrd_path))[0]
    for i, patch in enumerate(patches):
        out_path = os.path.join(output_folder, f"{base}_patch_{i:03d}.npy")
        np.save(out_path, patch)

print(f"\n✅ All patches saved to: {output_folder}")
input("Press Enter to exit...")