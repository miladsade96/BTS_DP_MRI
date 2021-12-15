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


def _build_bridge_block(pl: Tensor, n_filters: int = 256, k_size: Tuple = (3, 3, 3),
                        af=relu, ki: str = "he_uniform", drop_rate: float = 0.3):
    """
    Bridge path convolution block builder
    :param pl: Tensor, Previous layer
    :param n_filters: Integer, Number of filters in convolution layer, default is 256
    :param k_size: Tuple, Kernel size in convolution layer, default is (3, 3, 3)
    :param af: Activation function in convolution layer, default is relu
    :param ki: String, Layer weight initializer algorithm, default is he_uniform
    :param drop_rate: Float, Dropout layer rate, default is 0.3
    :return: Tensor
    """
    c_1 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(pl)
    do = Dropout(rate=drop_rate)(c_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(do)
    return c_2


def _build_decoder_block(pl: Tensor, sc: Tensor, n_filters: int, drop_rate: float, k_size_tr: Tuple = (2, 2, 2),
                         k_size: Tuple = (3, 3, 3), stride: Tuple = (2, 2, 2), af=relu, ki: str = "he_uniform"):
    """
    Decoder path Upsampling block builder
    :param pl: Tensor, Previous layer
    :param sc: Tensor, Skip connection returned from corresponding downsampling block
    :param n_filters: Integer, Number of filters in Conv3DTranspose and Con3D
    :param k_size_tr: Tuple, Kernel size in Conv3DTranspose layer, default is (2, 2, 2)
    :param k_size: Tuple, Kernel size in Conv3D layer, default is (3, 3, 3)
    :param stride: Tuple, Strides size in Conv3DTranspose layer, default is (2, 2, 2)
    :param af: Activation function in Conv3D layer, default is relu
    :param ki: String, Layer weight initializer algorithm, default is he_uniform
    :param drop_rate: Float, Dropout layer rate
    :return: Tensor
    """
    ct = Conv3DTranspose(filters=n_filters, kernel_size=k_size_tr, strides=stride, padding="same")(pl)
    concat = concatenate([ct, sc], axis=4)
    c_1 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(concat)
    do = Dropout(rate=drop_rate)(c_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(do)
    return c_2
