"""
    This prrprocessing script performs the followings:
        * Scaling all volumes using MinMaxScalar
        * Combining the three volumes(T1, MD, rCBV) into single multi-channel volume
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
t1_list = sorted(glob.glob(f"{args.dataset}/Train/*/*T1.nii"))
MD_list = sorted(glob.glob(f"{args.dataset}/Train/*/*MD.nii"))
rCBV_list = sorted(glob.glob(f"{args.dataset}/Train/*/*rCBV.nii"))
mask_list = sorted(glob.glob(f"{args.dataset}/Train/*/*mask.nii"))


PATH = os.path.abspath(f"{args.dataset}/")      # Getting absolute path
os.chdir(PATH)  # Navigating to dataset
for item in range(len(t1_list)):    # Creating new specific directory for any sample in dataset
    os.makedirs(f"npy_files/{item}", exist_ok=True)
    if args.verbose:
        print(f"Directory number {item} created or already exists.")
os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Navigating back to project root directory


for i, _ in enumerate(t1_list):
    # Adding six zero slices on axis 2 in order to feed .npy file into U-Net model
    # image files shape: (128, 128, 16, 3) and mask files shape: (128, 128, 16, 4)
    zeros = np.zeros((128, 128, 6))

    if args.verbose:
        print("Now preparing image and masks number: ", i)

    temp_image_t1 = nib.load(os.path.abspath(t1_list[i])).get_fdata()
    temp_image_t1 = mm_scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(
        temp_image_t1.shape)
    temp_image_t1 = np.append(temp_image_t1, zeros, axis=2)
    if args.verbose:
        print(f"T1 for sample number {i} Loaded and rescaled.")
    temp_image_MD = nib.load(os.path.abspath(MD_list[i])).get_fdata()
    temp_image_MD = mm_scaler.fit_transform(temp_image_MD.reshape(-1, temp_image_MD.shape[-1])).reshape(
        temp_image_MD.shape)
    temp_image_MD = np.append(temp_image_MD, zeros, axis=2)
    if args.verbose:
        print(f"MD for sample number {i} Loaded and rescaled.")
    temp_image_rCBV = nib.load(os.path.abspath(rCBV_list[i])).get_fdata()
    temp_image_rCBV = mm_scaler.fit_transform(temp_image_rCBV.reshape(-1, temp_image_rCBV.shape[-1])).reshape(
        temp_image_rCBV.shape)
    temp_image_rCBV = np.append(temp_image_rCBV, zeros, axis=2)
    if args.verbose:
        print(f"rCBV for sample number {i} Loaded and rescaled.")
    temp_mask = nib.load(os.path.abspath(mask_list[i])).get_fdata()
    temp_mask = np.append(temp_mask, zeros, axis=2)
    temp_mask = temp_mask.astype(np.uint8)
    if args.verbose:
        print(f"Mask for sample number {i} Loaded and converted to uint8.")
    temp_combined_images = np.stack([temp_image_t1, temp_image_MD, temp_image_rCBV], axis=3)
    if args.verbose:
        print(f"T1, MD and rCBV volumes combined as a single MegaVolume.")

    output = os.path.abspath("dataset/npy_files/")
    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.05:  # At least 5% useful volume with labels that are not 0
        print("Save Me")
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
        print(f"Number {i} is useless.")
