# Build RNN-LSTM model with random initial weights 
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.random import seed
import sklearn
import collections
import tensorflow as tf
#from tensorflow import config
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.debugging.set_log_device_placement(False)
from tensorflow.keras import layers



def diversity(preds, temperature):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample(n_x, SampleModel, char_to_ix, int_to_char, seed, embed, temp):
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    text    = []
    newline_character = char_to_ix['E']
    counter = 1
    X = np.zeros((1,1,n_x))
    zero = np.zeros((1,1,n_x))
    X[0,0,char_to_ix['!']]=1
    idx = -1
    
    while (idx != newline_character and counter <= embed):
        
        y = SampleModel.predict(X)
        y_hat=y.reshape((counter,44))
        
        #Overwrite last index
        seed += 1
        counter +=1
        #print(counter)
        np.random.seed(seed+counter)
        X=np.append(X,zero,axis=1)
        
        #print(y_hat.shape)
        i = y_hat.shape[0] - 1
        #print(y_hat[i])
        idx = diversity(y_hat[i], temp)
        X[0,i+1,idx] = 1
        
        if (counter == 463):
            indices.append(char_to_ix['E'])
    
    for i in range(X.shape[1]):
        idx = np.where(X[0,i]==1)
        txt = ''.join(int_to_char[idx[0][0]])
        text.append(txt)
        print ('%s' % (txt, ), end='')
    
    return indices,text[1:-1]
    
