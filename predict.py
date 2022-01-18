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
parser.add_argument("--image", help="Path to preprocessed .npy image file")
parser.add_argument("--mask", help="Path to preprocessed .npy mask file")
parser.add_argument("--model", help="Path to saved model")
parser.add_argument("-v", "--verbose", help="Level of verbosity", action="store_true")
parser.add_argument("-s", "--save_plot", help="Save Prediction plot", default=os.getcwd())
# Parsing the arguments
args = parser.parse_args()
