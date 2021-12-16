"""
    Custom Image data Generator
    That loads the .npy files from local directory in batches
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import os
import glob
import numpy as np
from typing import List, Generator


def _load_images(img_list: List) -> np.ndarray:
    """
    This private function loads the numpy array image files
    that are passed within input parameter and append them into a list
    :param img_list: List, Contains absolute paths to image.npy files
    :return: N-dimensional array, Loaded images
    """
    images = list()
    for image in img_list:
        if image.split(".")[1] == "npy":
            npy_image = np.load(image)
            images.append(npy_image)
    images = np.array(images)
    return images


def _load_masks(mask_list: List) -> np.ndarray:
    """
    This private function loads the numpy array mask files
    that are passed within input parameter and append them into a list
    :param mask_list: List, Contains absolute paths to mask.npy files
    :return: N-dimensional array, Loaded masks
    """
    masks = list()
    for mask in mask_list:
        if mask.split(".")[1] == "npy":
            npy_mask = np.load(mask)
            masks.append(npy_mask)
    masks = np.array(masks)
    return masks


def image_generator(path: str, batch_size: int) -> Generator:
    """
    This function gets .npy files path that are produced from preprocessing step and
    yields images and masks in batches
    :param path: String, Path to .npy files
    :param batch_size: Integer, Batch size
    :return: Generator, Loaded images and masks in batch
    """
    path = os.path.abspath(path)
    images_list = sorted(glob.glob(f"{path}/*/image_*.npy"))
    masks_list = sorted(glob.glob(f"{path}/*/mask_*.npy"))

    length = len(images_list)

    while True:
        batch_start_point = 0
        batch_end_point = batch_size
        while batch_start_point < length:
            limit = min(batch_end_point, length)
            x = _load_images(images_list[batch_start_point:limit])
            y = _load_masks(masks_list[batch_start_point:limit])
            yield x, y
            batch_start_point += batch_size
            batch_end_point += batch_size
