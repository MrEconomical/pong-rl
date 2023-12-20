'''
traditional mean squared error neural network model class with 1 hidden layer and
1 output neuron using relu for the hidden layer and sigmoid for the output neuron
'''

import json
import numpy as np

class Model:
    input_size = None
    hidden_size = None
    learning_rate = None
    weights = None

    # set model data

    def __init__(self, input_size, hidden_size, learning_rate, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights = weights

    # create new model with He and Xavier initialization

    @classmethod
    def with_random_weights(self, input_size, hidden_size, learning_rate):
        hidden_weights = np.empty((hidden_size, input_size + 1))
        hidden_weights[:, :-1] = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size) # He initialization
        hidden_weights[:, -1] = 0

        output_weights = np.empty(hidden_size + 1)
        output_weights[:-1] = np.random.randn(hidden_size) * np.sqrt(1 / hidden_size) # Xavier initialization
        output_weights[-1] = 0

        return self(
            input_size,
            hidden_size,
            learning_rate,
            [hidden_weights, output_weights]
        )

    # load model from file

    @classmethod
    def from_save(self, file_name):
        model_data = None
        with open(file_name, "r") as file:
            model_data = json.load(file)
        return self(
            model_data["input_size"],
            model_data["hidden_size"],
            model_data["learning_rate"],
            [
                np.array(model_data["weights"][0]),
                np.array(model_data["weights"][1]),
            ]
        )
    
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

        return hidden_gradients, output_gradient
    
    # update model weights

    def update_gradients(self, hidden_gradients, output_gradient):
        self.weights[0] -= self.learning_rate * hidden_gradients
        self.weights[1] -= self.learning_rate * output_gradient
    
    # save model to file

    def save(self, file_path):
        serialized_model = json.dumps(
            {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "learning_rate": self.learning_rate,
                "weights": [
                    self.weights[0].tolist(),
                    self.weights[1].tolist(),
                ],
            },
            indent=4,
        )
        with open(file_path, "w") as file:
            file.write(serialized_model)
