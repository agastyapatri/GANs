#   Global Imports
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)

#   Custom Imports
from src.GAN import Discriminator

DISC_ARCH = [1, 20, 1] 
DISC = Discriminator(arch = DISC_ARCH)

