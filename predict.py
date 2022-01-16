"""
    Loading trained model and make a prediction on unseen data
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
from segmentation_models_3D.metrics import iou_score
from segmentation_models_3D.losses import DiceLoss, categorical_focal_loss
