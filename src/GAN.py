import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
dtype = torch.float32 


class Discriminator(nn.Module):
    """
        Defining the Discriminator network. A simple MLP which labels a given sample into the actual distribution or the fake sample produced by G. 

        [args]:
            arch: a list defining the sizes of each layer in the network   
        
        [returns];
            a tensor of shape (1, 2)
    """

    def __init__(self, disc_arch:list) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(
            #   Input Layer
            nn.Linear(disc_arch[0], disc_arch[1], dtype=dtype),
            nn.Sigmoid(),
            
            #   Output Layer 
            nn.Linear(disc_arch[1], disc_arch[2], dtype=dtype),
            nn.Softmax()
        )

    def forward(self, x):
        #   Defining the forward pass of the network 
        output = self.discriminator(x)
        return output 



class Generator(nn.Module):
    """
        Defining the Generator Network. A simple MLP which produces samples that need to approximate a distribution. 
        
        [args]:
            arch: a list defining the sizes of each layer in the network   
        
        [returns];
            a tensor of shape (1, 2)
    """
    def __init__(self, gen_arch:list) -> None:
        super().__init__()
        self.generator = nn.Sequential(
            #   Input Layer
            nn.Linear(gen_arch[0], gen_arch[1], dtype=dtype),
            nn.ReLU(),
            
            #   Output Layer 
            nn.Linear(gen_arch[1], gen_arch[2], dtype=dtype),
            nn.ReLU()
        )

    def forward(self, x):
        #   Defining the forward pass of the network 
        output = self.generator(x)
        return output 



if __name__ == "__main__":
    #   Defining the metadata
    sample_data = torch.randn((50, 1), dtype=dtype)
    DISC_ARCH = [1, 20, 2] 
    GEN_ARCH = [1, 20, 1] 
    

    #   Defining the networks.
    DISC = Discriminator(disc_arch = DISC_ARCH)
    GEN = Generator(gen_arch = GEN_ARCH)
