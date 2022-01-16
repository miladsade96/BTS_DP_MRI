"""
    Loading trained model and make a prediction on unseen data
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""

import argparse
import numpy as np
import nibabel as nb
from glob import glob
from os.path import abspath
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
