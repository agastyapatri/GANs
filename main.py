#   Global Imports
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)

#   Custom Imports
from src.GAN import Discriminator, Generator
from src.DGEN import DISC_DATA, GEN_DATA


"""--------------------------------------------------------------------------------0. METADATA--------------------------------------------------------------------------------"""
DISC_ARCH = [1, 20, 2] 
GEN_ARCH = [1, 20, 1] 



"""-----------------------------------------------------------------------1. Loading the Data---------------------------------------------------------------------------------"""
generator_data = GEN_DATA(dim = 1, len = 1000)


"""------------------------------------------------------------------2. Creating the Networks---------------------------------------------------------------------------------"""
DISC = Discriminator(disc_arch = DISC_ARCH)
GEN = Generator(gen_arch=GEN_ARCH)
