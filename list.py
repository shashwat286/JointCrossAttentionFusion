import option
import numpy as np
from model import Model
import torch
from torch.utils.data import random_split
from dataset import Dataset
from torch.utils.data import DataLoader


args = option.parser.parse_args()

torch.cuda.empty_cache()