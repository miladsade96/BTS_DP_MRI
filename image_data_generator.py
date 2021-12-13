"""
    Custom Image data Generator
    That loads the .npy file from local directory in batches
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import os
import glob
import numpy as np
from typing import List, Tuple


def _load_images(img_list: List) -> Tuple:
    """
    This private function loads the numpy array image files
    that are passed within input parameter and append them into a list
    :param img_list: List, Contains absolute paths to .npy files
    :return: Tuple, Loaded images
    """
    images = list()
    for image in img_list:
        if image.split(".")[1] == "npy":
            npy_image = np.load(image)
            images.append(npy_image)
    images = np.array(images)
    return tuple(images)
