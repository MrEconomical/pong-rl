'''
stochastic gradient descent neural network model class with 1 hidden layer and 
1 output neuron using relu for the hidden layer and sigmoid for the output neuron
'''

import numpy as np

class Model:
    input_size = None
    hidden_size = None
    learning_rate = None
    weights = None

    # initialize parameters and weights

    def __init__(self, input_size, hidden_size, learning_rate):
        # set hyperparameters

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # initialize weights with Xavier initialization

        self.weights = [
            np.random.randn(hidden_size, input_size + 1) / np.sqrt(input_size + 1),
            np.random.randn(hidden_size + 1) / np.sqrt(hidden_size + 1),
        ]
    
    # calculate forward propagation result

    def forward(self, input_data):
        hidden_output = np.dot(self.weights[0][:, :-1], input_data) + self.weights[0][:, -1]
        np.maximum(hidden_output, 0, out=hidden_output) # relu activation
        output = np.dot(self.weights[1][:-1], hidden_output) + self.weights[1][-1]
        output = 1 / (1 + np.exp(-output)) # sigmoid activation

        return hidden_output, output

    # update weights with back propagation

    def back_prop(self, input_data, hidden_output, output, expected):
        # calculate gradients for output neuron using sigmoid derivative

        output_delta = (output - expected) * (output * (1 - output)) # using sigmoid derivative
        output_gradient = np.empty(self.hidden_size + 1)
        output_gradient[:-1] = output_delta * hidden_output # set output weight derivatives
        output_gradient[-1] = output_delta # bias is a fixed input of 1

        # calculate gradients for hidden neurons using relu derivative

        hidden_deltas = output_delta * self.weights[1][:-1] * (hidden_output > 0) # using relu derivative
        hidden_gradients = np.empty((self.hidden_size, self.input_size + 1))
        hidden_gradients[:, :-1] = np.outer(hidden_deltas, input_data) # set hidden weight derivatives
        hidden_gradients[:, -1:] = np.reshape(hidden_deltas, (self.hidden_size, 1)) # bias is a fixed input of 1

        # update model weights

        self.weights[0] -= self.learning_rate * hidden_gradients
        self.weights[1] -= self.learning_rate * output_gradient

        # return mean squared error

        return 0.5 * (expected - output) ** 2
