"""
    Standard 3D U-Net Model
    @Author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from typing import Tuple
from tensorflow.keras.models import Model
from tensorflow.python.types.core import Tensor
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import (Input, Conv3D, Conv3DTranspose, Dropout,
                                     MaxPooling3D, concatenate, BatchNormalization)


def _build_encoder_block(pl, n_filters: int, k_size: Tuple = (3, 3, 3), af=relu, ki: str = "he_uniform",
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
    bn_1 = BatchNormalization()(c_1)
    do = Dropout(rate=drop_rate)(bn_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, kernel_initializer=ki, activation=af, padding="same")(do)
    bn_2 = BatchNormalization()(c_2)
    mp = MaxPooling3D(pool_size=p_size)(bn_2)
    return mp, c_2  # c_2 is skip connection that will be connected to corresponding upsampling block in decoder path


def _build_bridge_block(pl, n_filters: int = 256, k_size: Tuple = (3, 3, 3),
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
    bn_1 = BatchNormalization()(c_1)
    do = Dropout(rate=drop_rate)(bn_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(do)
    bn_2 = BatchNormalization()(c_2)
    return bn_2


def _build_decoder_block(pl, sc: Tensor, n_filters: int, drop_rate: float, k_size_tr: Tuple = (2, 2, 2),
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
    bn_1 = BatchNormalization()(c_1)
    do = Dropout(rate=drop_rate)(bn_1)
    c_2 = Conv3D(filters=n_filters, kernel_size=k_size, activation=af, kernel_initializer=ki, padding="same")(do)
    bn_2 = BatchNormalization()(c_2)
    return bn_2


def build_unet_model(img_height: int, img_width: int, img_depth: int, img_channels: int, num_classes: int):
    """
    U-Net model builder
    :param img_height: Integer, Input image height
    :param img_width: Integer, input image width
    :param img_depth: Integer, Input image depth
    :param img_channels: Integer, Input image channels
    :param num_classes: Integer, Number of classes in output layer
    :return: Model
    """
    # Defining model input layer
    in_layer = Input(shape=(img_height, img_width, img_depth, img_channels))

    # Encoder path downsampling blocks
    out_1, sc_1 = _build_encoder_block(pl=in_layer, n_filters=16)
    out_2, sc_2 = _build_encoder_block(pl=out_1, n_filters=32)
    out_3, sc_3 = _build_encoder_block(pl=out_2, n_filters=64)
    out_4, sc_4 = _build_encoder_block(pl=out_3, n_filters=128)
    # Bridge block
    bb = _build_bridge_block(pl=out_4)

    # Decoder path blocks
    db_1 = _build_decoder_block(pl=bb, sc=sc_4, n_filters=128, drop_rate=0.2)
    db_2 = _build_decoder_block(pl=db_1, sc=sc_3, n_filters=64, drop_rate=0.2)
    db_3 = _build_decoder_block(pl=db_2, sc=sc_2, n_filters=32, drop_rate=0.1)
    db_4 = _build_decoder_block(pl=db_3, sc=sc_1, n_filters=16, drop_rate=0.1)

    # Defining model output layer
    out_layer = Conv3D(filters=num_classes, kernel_size=(1, 1, 1), activation=softmax)(db_4)

    final_model = Model(inputs=[in_layer], outputs=[out_layer])
    final_model.summary()
    return final_model
