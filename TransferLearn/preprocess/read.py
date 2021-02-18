# Read files and convert into dataframes
import pathlib
from os.path import splitext
import numpy as np
import pandas as pd
import datetime

def read(path, datatype = None, **kwargs):
    '''
    Read function for TransferLearn.
    
    If datatype is None, read the file as pandas dataframe (default)
    If datatype is 'df', read the file as pandas dataframe
    If datatype is 'np', read the file as numpy array
    
    Accepts:
            path:                  (str) Path to a valid file (.txt, .csv or .xlsx)
            datatype:              (str) Specifies which format to read ('df' or 'np')
            nrows:                 (int) Number of rows to read 
    Returns:
            df:                     Dataframe/numpy array
    
    '''
    _,fform = splitext(path)
    if fform in ['.txt','.text','.csv','.xlsx']:
        if fform in ['.txt','.text','.csv']:
            df = pd.read_csv(path, **kwargs)
        else:
            df = pd.read_excel(path, **kwargs)
            
        if datatype == 'np':
            df = df.to_numpy()
        elif datatype not in ['df',None]:
            raise Exception("Datatype {} not recognized. Only dataframe (df) and numpy (np) are the possible choice".format(datatype))
    else:
        raise Exception("Unsupported file extension {}.  Currently supports one of these formats - .txt, .csv or .xlsx".format(fform))
    
    return df