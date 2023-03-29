from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models.classify import *
from engine import train_augmodel
from argparse import  ArgumentParser


# Training settings
parser = ArgumentParser(description='Train Augmented model')
parser.add_argument('--configs', type=str, default='./config/celeba/training_augmodel/ffhq.json')  
args = parser.parse_args()


if __name__ == '__main__':
    file = args.configs
    cfg = load_json(json_file=file)

    train_augmodel(cfg)