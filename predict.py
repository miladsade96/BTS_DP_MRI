"""
    Loading trained model and make a prediction on unseen data
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Initializing cli argument parser
parser = argparse.ArgumentParser()
# Adding the arguments
parser.add_argument("--model", help="Path to saved model")
parser.add_argument("--mask", help="Path to preprocessed .npy mask file")
parser.add_argument("--image", help="Path to preprocessed .npy image file")
parser.add_argument("-v", "--verbose", help="Level of verbosity", action="store_true")
parser.add_argument("-s", "--save_plot", help="Save Prediction plot", default=os.getcwd())
parser.add_argument("--slice", help="Predicted mask lice number in order to plot", type=int, default=6)
# Parsing the arguments
args = parser.parse_args()

# Loading the saved model
model = load_model(filepath=args.model, compile=False)

# Loading the image and mask
image = np.load(file=args.image)
mask = np.load(file=args.mask)

mask_argmax = np.argmax(mask, axis=3)   # Converting from categorical
img_input = np.expand_dims(image, axis=0)   # Expanding image dimension
prediction = model.predict(img_input)       # Making a prediction
prediction_argmax = np.argmax(prediction, axis=4)[0, :, :, :]


n_slice = args.slice    # slice number
# Plotting prediction
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Test Image')
plt.imshow(image[:, :, n_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Test Mask')
plt.imshow(mask_argmax[:, :, n_slice])
plt.subplot(233)
plt.title('Prediction')
plt.imshow(prediction_argmax[:, :, n_slice])
plt.savefig(f"{args.save_plot}prediction.png")
if args.verbose:
    plt.show()
