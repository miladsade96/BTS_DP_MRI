"""
    Training 3D U-Net Model
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import build_unet_model
from tensorflow.keras.optimizers import Adam
from image_data_generator import image_generator
from tensorflow.keras.callbacks import TensorBoard
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
train_data_generator = image_generator(path=f"{args.dataset}/Train/", batch_size=args.batch_size)
valid_data_generator = image_generator(path=f"{args.dataset}/Valid/", batch_size=args.batch_size)

# Calculating class weights
columns = ["0", "1", "2", "3"]
df = pd.DataFrame(columns=columns)
mask_list = sorted(glob.glob(f"{args.dataset}/Train/*/mask_*.npy"))
for img in range(len(mask_list)):
    print(img)
    tmp_image = np.load(mask_list[img])
    tmp_image = np.argmax(tmp_image, axis=3)
    val, counts = np.unique(tmp_image, return_counts=True)
    zipped = zip(columns, counts)
    counts_dict = dict(zipped)
    df = df.append(counts_dict, ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['1'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4
wt0 = round((total_labels / (n_classes * label_0)), 2)
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

dice_loss = DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = CategoricalFocalLoss()
# Combining loss functions in order to create better total loss function
total_loss = dice_loss + (1 * focal_loss)

# Setting accuracy and IntersectionOverUnion as metrics
metrics = ["accuracy", IOUScore(threshold=0.5)]

# Building the model
model = build_unet_model(64, 64, 16, 3, 4)

# Defining callback objects
tensorboard_callback = TensorBoard(log_dir="./tb_logs", histogram_freq=1, write_graph=True,
                                   write_images=True, update_freq="epoch")

# Compiling the model
model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=total_loss, metrics=metrics)
# Setting training process
history = model.fit(
    train_data_generator,
    steps_per_epoch=26//2,
    validation_data=valid_data_generator,
    validation_steps=7//2,
    shuffle=True,
    epochs=args.epochs,
    verbose=args.verbose,
    callbacks=[tensorboard_callback]
)

# Saving the trained model
model.save(filepath=f"{args.save}/BTS_DP_MRI.hdf5", overwrite=True)

if args.verbose:
    # Plotting model history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
