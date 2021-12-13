"""
    Custom Image data Generator
    That loads the .npy file from local directory in batches
    Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import os
import glob
import numpy as np
from typing import List, Tuple, Any, Generator


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


def image_generator(path: str, batch_size: int) -> Generator[Tuple[Any]]:
    """
    This function gets .npy files path that are produced from preprocessing step and
    yields images and annotations in batches
    :param path: String, Path to .npy files
    :param batch_size: Integer, Batch size
    :return: Generator, Loaded images and annotations in batch
    """
    path = os.path.abspath(path)
    images_list = sorted(glob.glob(f"{path}/*/image_*.npy"))
    annotations_list = sorted(glob.glob(f"{path}/*/ann_*.npy"))

    length = len(images_list)

    while True:
        batch_start_point = 0
        batch_end_point = batch_size
        while batch_start_point < length:
            limit = min(batch_end_point, length)
            x = _load_images(images_list[batch_start_point:limit])
            y = _load_annotations(annotations_list[batch_start_point:limit])
            yield x, y
            batch_start_point += batch_size
            batch_end_point += batch_size
