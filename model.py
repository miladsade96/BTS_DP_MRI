"""
    Standard 3D U-Net Model
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.python.types.core import Tensor
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, MaxPooling3D, concatenate
