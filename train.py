"""
    Training 3D U-Net Model
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import build_unet_model
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from image_data_generator import image_generator
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from segmentation_models_3D.metrics import IOUScore
from segmentation_models_3D.losses import DiceLoss, CategoricalFocalLoss


# Initializing cli argument parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("-d", "--dataset", help="Path to .npy files directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Level of verbosity")
parser.add_argument("-l", "--learning_rate", help="Learning rate", type=float, default=0.001)
parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=2)
parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("-s", "--save", help="Path to save trained model", default=os.getcwd())

# Parsing the arguments
args = parser.parse_args()


kf = KFold(n_splits=8)  # Configuring kfold cross validation
fold_counter = 1    # Initializing fold counter

for train, valid in kf.split(range(34)):    # 33 is the number of samples
    print(f"Fold Number {fold_counter}")
    train_data_generator = image_generator(path=args.dataset, indexes=train, batch_size=2)
    valid_data_generator = image_generator(path=args.dataset, indexes=valid, batch_size=2)

    # Calculating class weights
    columns = ["0", "1", "2"]
    df = pd.DataFrame(columns=columns)
    mask_list = list()
    for index in train:
        mask_list.append(f"{args.dataset}/{index}/mask_{index}.npy")
    for img in range(len(mask_list)):
        tmp_image = np.load(mask_list[img])
        tmp_image = np.argmax(tmp_image, axis=3)
        val, counts = np.unique(tmp_image, return_counts=True)
        zipped = zip(columns, counts)
        counts_dict = dict(zipped)
        df = df.append(counts_dict, ignore_index=True)

    label_0 = df['0'].sum()
    label_1 = df['1'].sum()
    label_2 = df['2'].sum()
    total_labels = label_0 + label_1 + label_2
    n_classes = 3
    wt0 = round((total_labels / (n_classes * label_0)), 2)
    wt1 = round((total_labels / (n_classes * label_1)), 2)
    wt2 = round((total_labels / (n_classes * label_2)), 2)

    dice_loss = DiceLoss(class_weights=np.array([wt0, wt1, wt2]))
    focal_loss = CategoricalFocalLoss()
    # Combining loss functions in order to create better total loss function
    total_loss = dice_loss + (1 * focal_loss)

    # Setting accuracy and IntersectionOverUnion as metrics
    metrics = ["accuracy", "TruePositives", "TrueNegatives", "FalsePositives", "FalseNegatives",
               "Precision", "Recall", IOUScore(threshold=0.5)]

    # Building the model
    model = build_unet_model(64, 64, 16, 2, 3)

    # Defining callback objects
    tensorboard_callback = TensorBoard(log_dir="./tb_logs", histogram_freq=1, write_graph=True,
                                       write_images=True, update_freq="epoch")
    # Defining logger callback
    logger_callback = CSVLogger("log_file.csv", separator=",", append=True)

    # Compiling the model
    model.compile(optimizer=Adam(learning_rate=args.learning_rate), loss=total_loss, metrics=metrics)
    n_training_samples = len(train)
    n_validating_samples = len(valid)
    # Setting training process
    history = model.fit(
        train_data_generator,
        steps_per_epoch=n_training_samples//2,
        validation_data=valid_data_generator,
        validation_steps=n_validating_samples//2,
        shuffle=True,
        epochs=args.epochs,
        verbose=args.verbose,
        callbacks=[tensorboard_callback, logger_callback]
    )

    # Saving the trained model
    model.save(filepath=f"{args.save}/BTS_DP_MRI_fold_0{fold_counter}.hdf5", overwrite=True)

    if args.verbose:
        # Plotting model history
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "y", label="Training Loss")
        plt.plot(epochs, val_loss, "r", label="Validation Loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(fname=f"./tv_loss_0{fold_counter}.png", dpi=960)
        plt.show()

        iou_score = history.history["iou_score"]
        val_iou_score = history.history["val_iou_score"]
        plt.plot(epochs, iou_score, 'y', label='Training IOU Score')
        plt.plot(epochs, val_iou_score, 'r', label='Validation IOU Score')
        plt.title('Training and validation IOU Score')
        plt.xlabel('Epochs')
        plt.ylabel('IOU Score')
        plt.legend()
        plt.savefig(fname=f"./tv_iou_score_0{fold_counter}.png", dpi=960)
        plt.show()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        plt.plot(epochs, acc, 'y', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(fname=f"./tv_acc_0{fold_counter}.png", dpi=960)
        plt.show()

        fold_counter += 1
