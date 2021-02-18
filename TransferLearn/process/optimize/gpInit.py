# Build RNN-LSTM model with random initial weights 
import sklearn
import collections
from tqdm import tqdm
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .utils import predictions, expected_improvement, propose_location, best_location
from ..RNNTransferLearn import build_model


def gpInit(kernel, n_restarts_optimizer=1000, **kwargs): 
    
    '''    
    Run Gaussian Process optimization model for given number of iterations
    
    Accepts:
            kernel:                         (int) Number of LSTM layers required
            n_restarts_optimizer:           (list/array of size num_layers) lstm units for each layer
    Returns:
            gpmodel:                         compiled RNn-LSTM model
    
    '''

    # Instantiate a Gaussian Process model
    gpmodel = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)  
    
    return gpmodel

