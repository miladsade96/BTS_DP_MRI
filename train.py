"""
    Training 3D U-Net Model
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import os
import argparse
import matplotlib.pyplot as plt
from model import build_unet_model
from tensorflow.keras.optimizers import Adam
from image_data_generator import image_generator
from segmentation_models_3D.metrics import IOUScore
from segmentation_models_3D.losses import DiceLoss, CategoricalFocalLoss


# Initializing cli argument parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("-d", "--dataset", help="Path to .npy files directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Level of verbosity")
parser.add_argument("-l", "--learning_rate", help="Learning rate", type=float, default=0.0001)
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=2)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("-s", "--save", help="Path to save trained model", default=os.getcwd())

# Parsing the arguments
args = parser.parse_args()

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

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=total_loss, metrics=metrics)
# Setting training process
history = model.fit(
    train_data_generator,
    steps_per_epoch=34//2,
    epochs=10,
    verbose=1,
)

# Saving the trained model
model.save(filepath="./BTS_DP_MRI.hdf5", overwrite=True)

# Plotting model history
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
