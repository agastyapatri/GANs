import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
dtype = torch.float32 


class Discriminator(nn.Module):
    """
        Defining the Discriminator network. A simple MLP which labels a given sample into the actual distribution or the fake sample produced by G. 
        Outputs Log Probabilites, log[ D(x) ]
    """

    def __init__(self, arch) -> None:
        super().__init__()
        self.discriminator = nn.Sequential(
            #   Input Layer
            nn.Linear(arch[0], arch[1], dtype=dtype),
            nn.Sigmoid(),
            
            #   Output Layer 
            nn.Linear(arch[1], arch[2], dtype=dtype),
            nn.Softmax()
        )

    def forward(self, x):
        #   Defining the forward pass of the network 
        output = self.discriminator(x)
        return output 



class Generator(nn.Module):
    """
        Defining the Generator Network. A simple MLP which produces samples that need to approximate a distribution. 
    """
    def __init__(self, ) -> None:
        super().__init__()
        pass
    def forward(self, x):
        #   Defining the forward pass of the network 
        pass 
    pass



if __name__ == "__main__":

    sample_data = torch.randn((50, 1), dtype=dtype)

    DISC_ARCH = [1, 20, 1] 
    DISC = Discriminator(arch = DISC_ARCH)
    print(DISC)
