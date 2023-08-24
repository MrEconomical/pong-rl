# neural network model class with 1 hidden layer and 1 output neuron and relu
# activation for the hidden layer and sigmoid activation for the output neuron

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
        hidden_output = np.dot(self.weights[0][:, :-1], input_data) + self.weights[0][:, -1:].reshape(self.hidden_size)
        np.maximum(hidden_output, 0, out=hidden_output) # relu activation
        output = np.dot(self.weights[1][:-1], hidden_output) + self.weights[1][-1]
        output = 1 / (1 + np.exp(-output)) # sigmoid activation

        return hidden_output, output

    # update weights with back propagation

    def back_prop(self, input_data, hidden_output, output, expected):
        # calculate output delta and new output neuron weights

        output_delta = (output - expected) * (output * (1 - output)) # using sigmoid derivative
        updated_output_weights = np.empty(self.hidden_size + 1)
        updated_output_weights[:-1] = output_delta * hidden_output # set output weight derivatives
        updated_output_weights[-1] = output_delta # bias is a fixed input of 1

        updated_output_weights *= -self.learning_rate
        updated_output_weights += self.weights[1]

        # back propagate error to hidden layer

        hidden_deltas = output_delta * self.weights[1][:-1] * (hidden_output > 0) # using relu derivative
        updated_hidden_weights = np.empty((self.hidden_size, self.input_size + 1))
        updated_hidden_weights[:, :-1] = np.outer(hidden_deltas, input_data) # set hidden weight derivatives
        updated_hidden_weights[:, -1:] = np.reshape(hidden_deltas, (self.hidden_size, 1)) # bias is a fixed input of 1

        updated_hidden_weights *= -self.learning_rate
        updated_hidden_weights += self.weights[0]

        # set model weights

        self.weights[0] = updated_hidden_weights
        self.weights[1] = updated_output_weights

        # return mean squared error

        return 0.5 * (expected - output) ** 2
