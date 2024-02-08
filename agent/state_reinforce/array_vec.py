'''
implementation of an expandable vector backed by a NumPy array
'''

import numpy as np

class ArrayVec:
    shape = None
    extend_len = None
    array = None
    index = None

    def __init__(self, shape, initial_size, extend_len):
        self.shape = shape
        self.extend_len = extend_len
        self.array = np.empty((initial_size,) + self.shape)
        self.index = 0
    
    def push(self, array):
        if self.index == self.array.shape[0]:
            self.array = np.append(self.array, np.empty((self.extend_len,) + self.shape), axis=0)
        self.array[self.index] = array
        self.index += 1
    
    def clear(self):
        self.index = 0
    
    def get_ref(self):
        return self.array[0:self.index]