# neural network model class with 1 hidden layer and 1 output neuron and relu
# activation for the hidden layer and sigmoid activation for the output neuron

import numpy as np

class Model:
    input_size = None
    hidden_size = None
    learning_rate = None
    weights = None

    def __init__(self, input_size, hidden_size, learning_rate):
        # hyperparameters

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # initialize weights with Xavier initialization

        self.weights = [
            np.random.randn(hidden_size, input_size + 1) / np.sqrt(input_size + 1),
            np.random.randn(hidden_size + 1) / np.sqrt(hidden_size + 1),
        ]
