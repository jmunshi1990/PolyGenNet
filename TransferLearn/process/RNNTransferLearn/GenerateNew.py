# Build RNN-LSTM model with random initial weights 
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.random import seed
import sklearn
import collections
import tensorflow as tf
from tensorflow import config
from .utils import sample, diversity


def GenerateNew(model, output_dim, char_to_int, int_to_char, embed, temp=0.5, num_mol = 100):  
    '''    
    Generate new samples
    
    '''
    
    
    for i in range(num_mol):
        #print('\n Generating Molecule: ', i, '\n') 
        seed = i+1
        _,text = sample(output_dim, model, char_to_int, int_to_char, seed, embed, temp)
        #print(text)
        print('\n')
    
