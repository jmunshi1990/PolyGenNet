# Read files and convert into dataframes
import pathlib
from os.path import splitext
import numpy as np
import pandas as pd
import datetime

def OneHotVectorize(smiles, embed, vocab):
    '''
    One hot vectorization of SMILES strings.
    
    Accepts:
            smiles:              (str) array/dataframe of SMILES strings
            embed:               (int) Max length of SMILES strings
            dictionary:          (array/list) vocabulary for the chemical language
    Returns:
            Feature:             (numpy array) one hot encoded vectors as feature (RNN-LSTM)
            Label:               (numpy array) one hot encoded (shifted) vectors as labels (RNN-LSTM)
    
    '''
    one_hot =  np.zeros((smiles.shape[0], embed , len(vocab)),dtype=np.int8)
    char_to_int = dict((c,i) for i,c in enumerate(vocab))
    int_to_char = dict((i,c) for i,c in enumerate(vocab))
    
    for i,smile in enumerate(smiles):
        
        # print(smile)
        #encode the startchar
        pos = 1
        one_hot[i,0,char_to_int["!"]] = 1
        r_exist = False
        for j,c in enumerate(smile):
            if j < len(smile) - 1:
                if r_exist:
                    r_exist = False
                elif c=='B' and smile[j+1] == 'r':
                    one_hot[i,pos,char_to_int['Br']] = 1
                    r_exist = True
                    pos+=1
                elif c=='C' and smile[j+1] == 'l' :
                    one_hot[i,pos,char_to_int['Cl']] = 1
                    r_exist = True;
                    pos+=1
                elif c=='S' and smile[j+1] == 'i':
                    one_hot[i,pos,char_to_int['Si']] = 1
                    r_exist = True
                    pos+=1
                elif c=='S' and smile[j+1] == 'e':
                    one_hot[i,pos,char_to_int['Se']] = 1
                    r_exist = True;
                    pos+=1
                elif c=='G' and smile[j+1] == 'e':
                    one_hot[i,pos,char_to_int['Ge']] = 1
                    r_exist = True
                    pos+=1
                elif c=='T' and smile[j+1] == 'e':
                    one_hot[i,pos,char_to_int['Te']] = 1
                    r_exist = True
                    pos+=1
                else:
                    one_hot[i,pos,char_to_int[c]] = 1
                    pos+=1
            elif smile[j] == 'r' or smile[j] == 'l':
                break
            else:
                one_hot[i,pos,char_to_int[c]] = 1
                pos+=1
        one_hot[i,pos+1:,char_to_int["E"]] = 1
        
        feature = one_hot[:,0:-1,:]
        label = one_hot[:,1:,:]
        
    #Return two, one for input and the other for output
    return feature, label, char_to_int, int_to_char