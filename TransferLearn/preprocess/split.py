# Read files and convert into dataframes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split(data, test_size = 0.10, **kwargs):
    '''
    Read function for TransferLearn.
    
    If datatype is None, read the file as pandas dataframe (default)
    If datatype is 'df', read the file as pandas dataframe
    If datatype is 'np', read the file as numpy array
    
    Accepts:
            data:                  (dataframe/numpy array) specify dataset
            test_size:             (float) specify the fraction of test dataset (default: 0.01)
    
    Returns:
            train:                  training dataset
            test:                   test dataset
    
    '''
    train, test = train_test_split(data, test_size)
    
    return train, test