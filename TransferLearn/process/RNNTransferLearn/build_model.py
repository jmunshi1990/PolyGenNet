# Build RNN-LSTM model with random initial weights 
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.random import seed
import sklearn
import collections
import tensorflow as tf
from tensorflow import config
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
from tensorflow.keras import layers


def build_model(num_layers, lstm, dp, input_shape, output_dim, activation = 'softmax', optimizer='adam', loss='categorical_crossentropy', metrics = ['acc']):  
    '''    
    Create the LSTM model
    
    Accepts:
            num_layers:                     (int) Number of LSTM layers required
            lstm:                           (list/array of size num_layers) lstm units for each layer
            dp:                             (list/array of size num_layers) drop out rate units for each layer
            input_shape:                    shape of one hot encoded SMILES strings
            output_dim:                     output dimension for softmax (vocabulary size)
            activation:                     activation function for the final output layer (default: softmax)
            optimizer:                      optimizer for tensorflow model (default: adam)
            loss:                           compute loss metric for tensorflow model (default: categorical crossentropy)
            metric:                         compute metric for tensorflow model (default: accuracy) 
    Returns:
            model:                          compiled RNn-LSTM model
    
    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(lstm[0], input_shape=(None,input_shape[1]), return_sequences = True))
    model.add(tf.keras.layers.Dropout(dp[0]))
    
    assert(num_layers == len(lstm)), "Error: number of LSTM units must be equal to number of layers"
    assert(num_layers == len(dp)), "Error: number of drop out units must be equal to number of layers"
    
    for i in range(num_layers-1):
        model.add(tf.keras.layers.LSTM(lstm[i+1], return_sequences = True))
        model.add(tf.keras.layers.Dropout(dp[i+1]))

    model.add(tf.keras.layers.Dense(output_dim, activation=activation))
    model.compile(optimizer=optimizer, loss=loss, metrics = metrics)
    
    return model
