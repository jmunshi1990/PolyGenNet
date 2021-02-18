# Build RNN-LSTM model with random initial weights 
import numpy as np
import sklearn
import collections
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .utils import predictions, expected_improvement, propose_location, best_location
from ..RNNTransferLearn import build_model


def gpOpt(gpmodel, X, y, X_train, Y_train, X_test, Y_test, bounds, num_iter = 10, epochs = 10, batch_size= 256, **kwargs): 
    
    '''    
    Run Gaussian Process optimization model for given number of iterations
    
    Accepts:
            gpmodel:                     (int) Number of LSTM layers required
            X:                           (list/array of size num_layers) lstm units for each layer
            y:                             (list/array of size num_layers) drop out rate units for each layer
            bounds:                    shape of one hot encoded SMILES strings
            num_iter:                     output dimension for softmax (vocabulary size)
            epochs:                     activation function for the final output layer (default: softmax)
            batch_size:                      optimizer for tensorflow model (default: adam)
    Returns:
            best:                          compiled RNN-LSTM model
    
    '''
    c=-1
    for i in range(num_iter):
        c=c+1
        prior_size = len(X)
        gpmodel.fit(X, y)
        X_next,muu = propose_location(expected_improvement, X, y, gpmodel, bounds)
        
        num_layer = int(len(X_next)/2)
        lstm = X_next[:num_layer]
        dp = X_next[num_layer:]
      
        model= build_model(num_layer, lstm, dp, input_shape, output_dim)
        history = model.fit(X_train,Y_train ,validation_data=(X_test, Y_test),
                            epochs = epochs, batch_size = batch_size)
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        #hist.head()
        
        mu= hist.iloc[-1,:].val_loss
        X=np.append(X,X_next)
        X=X.reshape(prior_size+c, 2*num_layer)
        y=np.append(y,mu)
        
    best,_ = best_location(predictions,X, gpmodel, bounds)   
    return best
