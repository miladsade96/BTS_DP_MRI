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
    :param img_list: List, Contains absolute paths to image.npy files
    :return: Tuple, Loaded images
    """
    images = list()
    for image in img_list:
        if image.split(".")[1] == "npy":
            npy_image = np.load(image)
            images.append(npy_image)
    images = np.array(images)
    return tuple(images)


def _load_annotations(ann_list: List) -> Tuple:
    """
    This private function loads the numpy array annotation files
    that are passed within input parameter and append them into a list
    :param ann_list: List, Contains absolute paths to ann.npy files
    :return: Tuple, Loaded annotations
    """
    annotations = list()
    for annotation in ann_list:
        if annotation.split(".")[1] == "npy":
            npy_ann = np.load(annotation)
            annotations.append(npy_ann)
    annotations = np.array(annotations)
    return tuple(annotations)
