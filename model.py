"""
    Standard 3D U-Net Model
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.python.types.core import Tensor
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, Dropout, MaxPooling3D, concatenate


def _build_encoder_block(pl: Tensor, n_filters: int, k_size: Tuple = (3, 3, 3), af=relu, ki: str = "he_uniform",
                         drop_rate: float = 0.1, p_size: Tuple = (2, 2, 2)):
    """
    Encoder path convolution block builder
    :param pl: Tensor, Previous layer
    :param n_filters: Integer, Number of filters in convolution layer
    :param k_size: Tuple, Kernel size in convolution layer, default is (3, 3, 3)
    :param af: Activation function in convolution layer, default is relu
    :param ki: String, Layer weight initializer algorithm, default is he_uniform
    :param drop_rate: Float, Dropout layer rate, default is 0.1
    :param p_size: Tuple, Pool size in downsampling layer, default is (2, 2, 2)
    :return: Tensor
    """
    c_1 = Conv3D(filters=n_filters, kernel_size=k_size, kernel_initializer=ki, activation=af, padding="same")(pl)
    do = Dropout(rate=drop_rate)(c_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, kernel_initializer=ki, activation=af, padding="same")(do)
    mp = MaxPooling3D(pool_size=p_size)(c_2)
    return mp, c_2  # c_2 is skip connection that will be connected to corresponding upsampling block in decoder path
