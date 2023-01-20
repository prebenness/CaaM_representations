""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

# Mean and STD for standard datasets:
MNIST_TRAIN_MEAN = (0.1306604762738431)
MNIST_TRAIN_STD = (0.3081078038564622)
MNIST_NUM_CLASSES = 10

CIFAR10_TRAIN_MEAN = (0.49139968, 0.48215827 ,0.44653124)
CIFAR10_TRAIN_STD = (0.24703233, 0.24348505, 0.26158768)
MNIST_NUM_CLASSES = 10

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
MNIST_NUM_CLASSES = 100

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'
DATA_DIR = 'data'
