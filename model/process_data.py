#-*- coding: utf-8 -*-
"""
what    : get batch
data    : 
"""
import numpy as np
import pickle
import random
import time

class ProcessData:

    def __init__(self, params):

        self.data_path = params.DATA_PATH
        self.params = params
        
        # load data
        self.train = self.load_data( self.data_path + self.params.DATA_TRAIN )
        self.dev   = self.load_data( self.data_path + self.params.DATA_DEV )
        self.test  = self.load_data( self.data_path + self.params.DATA_TEST )

        params.FEATURE_DIM = np.shape(self.train[0][0])[0]
        print ('[completed] load data, feature_dim: ', params.FEATURE_DIM)
        
        
    def load_data(self, file_path):
     
        print ('load data : ' + file_path)
        list_ret = []
        
        with open(file_path, 'rb') as f:
            list_data, list_label = pickle.load(f)
        
        print(len(list_data), len(list_label))
        
        # read & add data to list
        for data, label in zip(list_data, list_label):
            np_label = np.zeros(self.params.N_CLASS, dtype=np.float32)
            np_label[int(label)] = 1.0
            
            list_ret.append( [data, np_label] )
        
        return list_ret
        
    
    """
        inputs: 
            data         : data to be processed (train/dev/test)
            batch_size   : mini-batch size
            is_test      : True, inference stage (ordered input)  (default : False)
            start_index  : start index of mini-batch (will be used when is_test==True)

        return:
            list_q       : [batch, time_step(==MAX_LENGTH_Q)] questions
            list_a       : [batch, time_step(==MAX_LENGTH_A)] answers
            list_l        : [batch] labels
            
            list_len_q   : [batch] vaild sequecne length
            list_len_a   : [batch] vaild sequecne length
    """
    def get_batch(self, data, batch_size, is_test=False, start_index=0):

        list_d, list_l = [], []
        index = start_index
        
        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed

        for _ in range(batch_size):

            if is_test is False:
                # train case -  random sampling
                d, l = random.choice(data)
                
            else:
                # dev, test case = ordered data
                if index >= len(data):
                    d, l = data[0]  # dummy for batch - will not be evaluated
                    index += 1
                else: 
                    d, l = data[index]
                    index += 1

            list_d.append(d)
            list_l.append(l)
                
        return list_d, list_l