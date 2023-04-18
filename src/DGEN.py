import numpy as np
import scipy as sp
import torch
import torch.nn as nn
torch.manual_seed(0)
dtype = torch.float32


class DISC_DATA(torch.utils.data.Dataset):
    """
        Target Distribution for G, Training Data for D.  
    """
    def __init__(self, ) -> None:
        pass

    def __len__(self) -> int:
        pass 

    def __getitem__(self, i):
        return np.arange(1, 10, 1)[i] 


class GEN_DATA(torch.utils.data.Dataset):
    """
        The input data for G is random noise, of dimensionality which depends on the problem statement. 
    """
    def __init__(self, dim, len) -> None:
        self.len = len 
        self.dim = dim 

    def __len__(self) -> int:
        return self.len  
    
    def __getitem__(self, i):
        return torch.tensor(np.random.uniform(low=0, high=1, size=(self.len, self.dim)), dtype=dtype)[i] 


if __name__ == "__main__":
    disc_data = DISC_DATA()
    gen_data = GEN_DATA(dim = 2, len = 1000)


