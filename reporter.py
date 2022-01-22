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
