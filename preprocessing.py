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
Ann_list = sorted(glob.glob(f"{args.dataset}/Train/*/*Ann.nii"))


PATH = os.path.abspath(f"{args.dataset}/")      # Getting absolute path
os.chdir(PATH)  # Navigating to dataset
for item in range(len(t1_list)):    # Creating new specific directory for any sample in dataset
    os.makedirs(f"npy_files/{item}", exist_ok=True)
    if args.verbose:
        print(f"Directory number {item} created or already exists.")
os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Navigating back to project root directory


for img in range(len(t1_list)):
    if args.verbose:
        print("Now preparing image and masks number: ", img)

    temp_image_t1 = nib.load(os.path.abspath(t1_list[img])).get_fdata()
    temp_image_t1 = mm_scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(
        temp_image_t1.shape)
    if args.verbose:
        print(f"T1 for sample number {img} Loaded and rescaled.")
    temp_image_MD = nib.load(os.path.abspath(MD_list[img])).get_fdata()
    temp_image_MD = mm_scaler.fit_transform(temp_image_MD.reshape(-1, temp_image_MD.shape[-1])).reshape(
        temp_image_MD.shape)
    if args.verbose:
        print(f"MD for sample number {img} Loaded and rescaled.")
    temp_image_rCBV = nib.load(os.path.abspath(rCBV_list[img])).get_fdata()
    temp_image_rCBV = mm_scaler.fit_transform(temp_image_rCBV.reshape(-1, temp_image_rCBV.shape[-1])).reshape(
        temp_image_rCBV.shape)
    if args.verbose:
        print(f"rCBV for sample number {img} Loaded and rescaled.")
    temp_Ann = nib.load(os.path.abspath(Ann_list[img])).get_fdata()
    temp_Ann = temp_Ann.astype(np.uint8)
    if args.verbose:
        print(f"Ann for sample number {img} Loaded and converted to uint8.")
    temp_combined_images = np.stack([temp_image_t1, temp_image_MD, temp_image_rCBV], axis=3)
    if args.verbose:
        print(f"T1, MD and rCBV volumes combined as a single MegaVolume.")
    val, counts = np.unique(temp_Ann, return_counts=True)

    output = os.path.abspath("dataset/npy_files/")

    temp_Ann = to_categorical(temp_Ann, num_classes=4)

    if os.path.isfile(f"{output}/{img}/image_" + str(img) + ".npy"):
        if args.verbose:
            print(f"Image file already exists.")
    else:
        np.save(f"{output}/{img}/image_" + str(img) + ".npy", temp_combined_images)
        if args.verbose:
            print(f"Number {img} image .npy files saved successfully.")

    if os.path.isfile(f"{output}/{img}/ann_" + str(img) + ".npy"):
        if args.verbose:
            print(f"Annotation file already exists.")
    else:
        np.save(f"{output}/{img}/ann_" + str(img) + ".npy", temp_Ann)
        if args.verbose:
            print(f"Number {img} annotation .npy files saved successfully.")
