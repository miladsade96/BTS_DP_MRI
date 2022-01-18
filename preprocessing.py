"""
    This prrprocessing script performs the followings:
        * Scaling all volumes using MinMaxScalar
        * Combining the two volumes(MD, rCBV) into single multi-channel volume
        * Saving volumes as numpy arrays(.npy)

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


# Initializing cli argument parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("-d", "--dataset", help="Path to .nii files directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Level of verbosity")
# Parsing the arguments
args = parser.parse_args()

# Defining MinMaxScaler
mm_scaler = MinMaxScaler()

# List of Images
MD_list = sorted(glob.glob(f"{args.dataset}/Train/*/MD*.nii"))
rCBV_list = sorted(glob.glob(f"{args.dataset}/Train/*/rCBV*.nii"))
mask_list = sorted(glob.glob(f"{args.dataset}/Train/*/mask*.nii"))


PATH = os.path.abspath(f"{args.dataset}/")      # Getting absolute path
os.chdir(PATH)  # Navigating to dataset
for item in range(len(t1_list)):    # Creating new specific directory for any sample in dataset
    os.makedirs(f"npy_files/{item}", exist_ok=True)
    if args.verbose:
        print(f"Directory number {item} created or already exists.")
os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Navigating back to project root directory

print("-*-" * 50)

for i, _ in enumerate(t1_list):
    # Adding six zero slices on axis 2 in order to feed .npy file into U-Net model
    # image files shape: (128, 128, 16, 3) and mask files shape: (128, 128, 16, 4)
    zeros = np.zeros((128, 128, 6))

    if args.verbose:
        print(f"Preparing image and mask number: {i}")

    temp_mask = nib.load(os.path.abspath(mask_list[i])).get_fdata()
    if args.verbose:
        print(f"Mask file for sample number {i} is loaded")
    temp_mask = np.append(temp_mask, zeros, axis=2)
    if args.verbose:
        print(f"Zeros added to mask file of sample number {i}")
    temp_mask = temp_mask.astype(np.uint8)
    if args.verbose:
        print(f"Mask for sample number {i} converted to uint8.")
    # Efficient crop
    flt = np.where(temp_mask[:, :, 4] != 0)  # flt --> filter
    x, y = flt[0][0], flt[1][0]
    x = min(x, 63)
    y = min(y, 63)
    if args.verbose:
        print(f"Coordinates for sample number {i}: x: {x}, y: {y}")
    temp_mask = temp_mask[x:x + 64, y:y + 64, :]
    if args.verbose:
        print(f"Mask for sample number {i} cropped")
        print(f"mask shape: {temp_mask.shape}")

    temp_image_t1 = nib.load(os.path.abspath(t1_list[i])).get_fdata()
    temp_image_t1 = mm_scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(
        temp_image_t1.shape)
    if args.verbose:
        print(f"T1 for sample number {i} Loaded and rescaled.")
    temp_image_t1 = np.append(temp_image_t1, zeros, axis=2)
    if args.verbose:
        print(f"Zeros added to T1 file of sample number {i}")
    temp_image_t1 = temp_image_t1[x:x + 64, y:y + 64, :]
    if args.verbose:
        print(f"T1 for sample number {i} cropped")
        print(f"Cropped T1 shape: {temp_image_t1.shape}")

    temp_image_MD = nib.load(os.path.abspath(MD_list[i])).get_fdata()
    temp_image_MD = mm_scaler.fit_transform(temp_image_MD.reshape(-1, temp_image_MD.shape[-1])).reshape(
        temp_image_MD.shape)
    if args.verbose:
        print(f"MD for sample number {i} Loaded and rescaled.")
    temp_image_MD = np.append(temp_image_MD, zeros, axis=2)
    if args.verbose:
        print(f"Zeros added to MD file of sample number {i}")
    temp_image_MD = temp_image_MD[x:x + 64, y:y + 64, :]
    if args.verbose:
        print(f"MD for sample number {i} cropped")
        print(f"Cropped MD shape: {temp_image_t1.shape}")

    temp_image_rCBV = nib.load(os.path.abspath(rCBV_list[i])).get_fdata()
    temp_image_rCBV = mm_scaler.fit_transform(temp_image_rCBV.reshape(-1, temp_image_rCBV.shape[-1])).reshape(
        temp_image_rCBV.shape)
    if args.verbose:
        print(f"rCBV for sample number {i} Loaded and rescaled.")
    temp_image_rCBV = np.append(temp_image_rCBV, zeros, axis=2)
    if args.verbose:
        print(f"Zeros added to rCBV file of sample number {i}")
    temp_image_rCBV = temp_image_rCBV[x:x + 64, y:y + 64, :]
    if args.verbose:
        print(f"rCBV for sample number {i} cropped")
        print(f"Cropped rCBV shape: {temp_image_rCBV.shape}")

    temp_combined_images = np.stack([temp_image_t1, temp_image_MD, temp_image_rCBV], axis=3)
    if args.verbose:
        print(f"T1, MD and rCBV volumes combined as a single MegaVolume.")
        print(f"MegaVolume shape: {temp_combined_images.shape}")

    output = os.path.abspath("dataset/npy_files/")
    val, counts = np.unique(temp_mask, return_counts=True)
    if args.verbose:
        print(f"Values and counts for sample number {i}: {val}, {counts}")

    percent = 1 - (counts[0] / counts.sum())
    percent = int(round(percent, 2) * 100)
    print(f"Percent for mask number{i}: {percent}%")
    if percent > 0.01:  # At least 1% useful volume with labels that are not 0
        print(f"Saving sample number {i}")
        temp_mask = to_categorical(temp_mask, num_classes=4)

        if os.path.isfile(f"{output}/{i}/image_" + str(i) + ".npy"):
            if args.verbose:
                print(f"Image file already exists.")
        else:
            np.save(f"{output}/{i}/image_" + str(i) + ".npy", temp_combined_images)
            if args.verbose:
                print(f"Number {i} image .npy files saved successfully.")

        if os.path.isfile(f"{output}/{i}/mask_" + str(i) + ".npy"):
            if args.verbose:
                print(f"Mask file already exists.")
        else:
            np.save(f"{output}/{i}/mask_" + str(i) + ".npy", temp_mask)
            if args.verbose:
                print(f"Number {i} mask .npy files saved successfully.")

    else:
        print(f"Sample number {i} is useless.")

    print("-*-" * 50)
# Remove empty folders in npy_files
for dir_path, dir_names, file_names in os.walk("dataset/npy_files"):
    if not dir_names and not file_names:
        os.rmdir(dir_path)
