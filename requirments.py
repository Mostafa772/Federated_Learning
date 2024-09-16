import torch
import random
import torchvision
import mat73
import copy
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Union, Dict, List, OrderedDict
from torch import nn, optim, utils
from collections import OrderedDict
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold 
from sklearn.datasets import load_digits
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from itertools import chain
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm   
from torch.utils.data.sampler import SequentialSampler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
