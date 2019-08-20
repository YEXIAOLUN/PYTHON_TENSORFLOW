import os
import sys
import argparse
import datetime
import numpy as np
import tensorflow as tf

parser=argparse.ArgumentParser()
parser.add_argument('--data_path',type=str,default=data_path,help='the path of the \
                    date for trainging and testing')
args=parser.parse_args()
