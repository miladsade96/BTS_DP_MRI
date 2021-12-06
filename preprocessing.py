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
