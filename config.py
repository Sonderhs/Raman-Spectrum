import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False,
                    help='CUDA training.')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')  # 训练轮次
parser.add_argument('--lr', type=float, default=0.005,
                    help='Learning rate.')  # 学习率
parser.add_argument('--wd', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')  # L2正则化
parser.add_argument('--hidden', type=int, default=64,
                    help='Dimension of representations')
parser.add_argument('--c', type=int, default=2,
                    help='Num of classes')
parser.add_argument('--d', type=int, default=1200,
                    help='Num of spectra dimension')
parser.add_argument('--model', type=str, default='GCN',
                    help='Model')  # GCN, CNN, LSTM
parser.add_argument('--task', type=str, default='classify',
                    help='Task')  # classify, predict


args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
set_seed(args.seed, args.cuda)