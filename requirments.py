import torch
import random
from torch.utils.data import DataLoader, Dataset
from typing import Union, Dict, List, OrderedDict
from torch import nn, optim, utils
from collections import OrderedDict
import torch
import torch.nn.functional as F
import copy
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch import nn, optim
import random
import torchvision
import numpy as np
from sklearn.model_selection import StratifiedKFold 
from sklearn.datasets import load_digits
from torchvision.transforms import ToTensor
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import pandas as pd
import numpy as np
import copy
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as sio
import tqdm
import mat73
import copy
import sys
from itertools import chain
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm   
from torch import nn
from torch.utils.data.sampler import SequentialSampler
