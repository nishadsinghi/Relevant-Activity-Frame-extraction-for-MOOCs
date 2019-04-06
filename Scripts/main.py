import numpy as np 
import torch.nn as nn
import torch
from model import ConvModel

device = torch.device('cpu')


M = 1
Tx = 50

model = ConvModel().to(device)
