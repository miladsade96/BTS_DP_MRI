"""
    Training 3D U-Net Model
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


from model import build_unet_model
from tensorflow.keras.optimizers import Adam
from image_data_generator import image_generator
from segmentation_models_3D.metrics import IOUScore
from segmentation_models_3D.losses import DiceLoss, CategoricalFocalLoss


# Defining image data generator
train_data_generator = image_generator("dataset/npy_files", batch_size=2)

dice_loss = DiceLoss()
focal_loss = CategoricalFocalLoss()
# Combining loss functions in order to create better total loss function
total_loss = dice_loss + (1 * focal_loss)

# Setting accuracy and IntersectionOverUnion as metrics
metrics = ["accuracy", IOUScore(threshold=0.5)]

# Building the model
model = build_unet_model(128, 128, 16, 3, 4)
