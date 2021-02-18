# Build tansfer-learning RNN-LSTM model with customizable weights
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.random import seed
import sklearn
import collections
import tensorflow as tf
from tensorflow import config
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(False)
from tensorflow.keras import layers


def tlmodel(model, num_layers, lstm, dp, input_shape, output_dim, activation = 'softmax', freeze_layer = False, fine_tune = False , lr =  1e-5, optimizer='adam', loss='categorical_crossentropy', metrics = ['acc']):  
    '''    
    Create the LSTM model
    
    Accepts:
            model:                          pre-trained RNN-LSTM model for transfer-learning
            num_layers:                     (int) Number of LSTM layers required
            lstm:                           (list/array of size num_layers) lstm units for each layer
            dp:                             (list/array of size num_layers) drop out rate units for each layer
            input_shape:                    shape of one hot encoded SMILES strings
            output_dim:                     output dimension for softmax (vocabulary size)
            freeze_layer:                   check if the pre-trained layers should be frozen (default: False)
            fine_tune:                      check if the fine-tuning is required (default: False)
            lr:                             learning rate for the model optimizer; required if fine_tune = True (default: 1e-05)
            activation:                     activation function for the final output layer (default: softmax)
            optimizer:                      optimizer for tensorflow model (default: adam)
            loss:                           compute loss metric for tensorflow model (default: categorical crossentropy)
            metric:                         compute metric for tensorflow model (default: accuracy) 
    Returns:
            transfermodel:                  compiled transfer learninng RNN-LSTM model
    
    '''
    transfermodel =tf.keras.Sequential([tf.keras.Input(shape=(None,input_shape[1]))])
    
    for i in range(len(model.layers)-1):
        transfermodel.add(model.layers[i])
        if len(model.layers[i].get_weights()) > 0:
            transfermodel.layers[i].set_weights(model.layers[i].get_weights())
        
    
    if freeze_layer:
        for layer in (transfermodel.layers):
            layer.trainable = False
    else:
        for layer in (transfermodel.layers):
            layer.trainable = True
            
    if fine_tune:
        if optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(lr)
        elif optimizer == 'adamax':
            optimizer = tf.keras.optimizers.Adamax(lr)
        elif optimizer == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(lr)
        elif optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr)
        else:
            raise Exception("{} is not a valid optimizer".format(optimizer))
    
    assert(num_layers == len(lstm)), "Error: number of LSTM units must be equal to number of layers"
    assert(num_layers == len(dp)), "Error: number of drop out units must be equal to number of layers"
    
    for i in range(num_layers):
        transfermodel.add(tf.keras.layers.LSTM(lstm[i], return_sequences = True))
        transfermodel.add(tf.keras.layers.Dropout(dp[i]))

    transfermodel.add(tf.keras.layers.Dense(output_dim, activation=activation))
    transfermodel.compile(optimizer=optimizer, loss=loss, metrics = metrics)
    
    return transfermodel
