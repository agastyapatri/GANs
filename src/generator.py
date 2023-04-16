import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)


class DGEN:
    """
        DGEN is a class to generate the data for the generator and discriminator. Whether the 
    """
    def __init__(self, ) -> None:
        pass

    def __len__(self):
        pass 

    def __getitem__(self, i):
        return np.arange(1, 10, 1)[i] 

if __name__ == "__main__":
    data_generator = DGEN()
    print(data_generator[0], data_generator[1])