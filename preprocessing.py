"""
    This prrprocessing script performs the followings:
        * Scaling all volumes using MinMaxScalar
        * Combining the three volumes(T1, MD, rCBV) into single multi-channel volume
        * Saving volumes as numpy arrays(.npy)

    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical


# Defining MinMaxScaler
mm_scaler = MinMaxScaler()

# List of Images
t1_list = sorted(glob.glob("dataset/Train/*/*T1.nii"))
MD_list = sorted(glob.glob("dataset/Train/*/*MD.nii"))
rCBV_list = sorted(glob.glob("dataset/Train/*/*rCBV.nii"))
Ann_list = sorted(glob.glob("dataset/Train/*/*Ann.nii"))
