'''
implementation of an expandable vector backed by a NumPy array
'''

import numpy as np

class ArrayVec:
    shape = None
    extend_len = None
    array = None
    index = None

    def __init__(self, shape, initial_len, extend_len):
        self.shape = shape
        self.extend_len = extend_len
        self.array = np.empty((initial_len,) + self.shape)
        self.index = 0
    
    def push(self, array):
        if self.index == self.array.shape[0]:
            self.array = np.append(self.array, np.empty((self.extend_len,) + self.shape), axis=0)
        self.array[self.index] = array
        self.index += 1
    
    def clear(self):
        self.index = 0

    def len(self):
        return self.index
    
    def get_item(self, index):
        return self.array[index]
    
    def get_view(self, start, end):
        if start >= self.index:
            return self.array[0:0]
        elif end > self.index:
            return self.array[start:self.index]
        return self.array[start:end]
    
    def get_ref(self):
        return self.array[0:self.index]