"""
    Reporting results according to training log file
    @author: Milad Sadeghi DM - EverLookNeverSee@GitHub
"""


import argparse
import pandas as pd
from os import getcwd
from os.path import abspath


# Initializing cli argument parser
parser = argparse.ArgumentParser()
# Adding arguments
parser.add_argument("-f", "--file_path", help="Path to .csv file directory")
parser.add_argument("-v", "--verbose", action="store_true", help="Level of verbosity")
parser.add_argument("-s", "--save", help="Path to save report.csv file", default=getcwd())

# Parsing the arguments
args = parser.parse_args()

# Absolute path of csv file
path = abspath(args.file_path)

# Reading csv file with certain columns
if args.verbose:
    print("Reading log_file.csv")
df = pd.read_csv(filepath_or_buffer=f"{path}/log_file.csv", header=0, sep=",",
                 usecols=[
                         "epoch", "accuracy", "loss", "iou_score", "precision", "recall",
                         "val_accuracy", "val_loss", "val_iou_score", "val_precision", "val_recall"
                     ])
