'''
Deep Q-Network mean squared error model class with 1 hidden layer and 1 output
layer using relu for the hidden layer and linear activation for the output layer
with batched gradient updates
'''

import json
import numpy as np

class Model:
    input_size = None
    hidden_size = None
    learning_rate = None
    weights = None

    # set model data

    def __init__(self, input_size, hidden_size, output_size, learning_rate, discount_rate, explore_factor, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.explore_factor = explore_factor
        self.weights = weights

    # create new model with random small weights

    @classmethod
    def with_random_weights(self, input_size, hidden_size, output_size, learning_rate, discount_rate, explore_factor):
        hidden_weights = np.empty((hidden_size, input_size + 1))
        hidden_weights[:, :-1] = np.random.randn(hidden_size, input_size) / 50 # initialize with small weights
        hidden_weights[:, -1] = 0

        output_weights = np.empty((output_size, hidden_size + 1))
        output_weights[:, :-1] = np.random.randn(output_size, hidden_size) / 50 # initialize with small weights
        output_weights[:, -1] = 0

        return self(
            input_size,
            hidden_size,
            output_size,
            learning_rate,
            discount_rate,
            explore_factor,
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
            model_data["output_size"],
            model_data["learning_rate"],
            model_data["discount_rate"],
            model_data["explore_factor"],
            [
                np.array(model_data["weights"][0]),
                np.array(model_data["weights"][1]),
            ]
        )
    
    # calculate forward propagation result

    def forward(self, input_data):
        hidden_output = np.dot(self.weights[0][:, :-1], input_data) + self.weights[0][:, -1]
        np.maximum(hidden_output, 0, out=hidden_output) # relu activation
        output = np.dot(self.weights[1][:, :-1], hidden_output) + self.weights[1][:, -1]

        return hidden_output, output

    # calculate batched forward propagation result

    def batch_forward(self, input_batch):
        hidden_outputs = np.dot(self.weights[0][:, :-1], np.transpose(input_batch)) + self.weights[0][:, -1:]
        np.maximum(hidden_outputs, 0, out=hidden_outputs) # relu activation
        outputs = np.dot(self.weights[1][:, :-1], hidden_outputs) + self.weights[1][:, -1:]

        return hidden_outputs, outputs

    # calculate gradients with back propagation

    def back_prop(self, input_data, hidden_output, output, expected):
        # calculate gradients for output neuron using linear derivative

        output_deltas = output - expected # using linear derivative
        output_gradients = np.empty((self.output_size, self.hidden_size + 1))
        output_gradients[:, :-1] = np.outer(output_deltas, hidden_output) # set output weight derivatives
        output_gradients[:, -1:] = np.reshape(output_deltas, (self.output_size, 1)) # bias is a fixed input of 1

        # calculate gradients for hidden neurons using relu derivative

        hidden_predeltas = np.dot(output_deltas, self.weights[1][:, :-1]) # find total error per neuron
        hidden_deltas = hidden_predeltas * (hidden_output > 0) # using relu derivative
        hidden_gradients = np.empty((self.hidden_size, self.input_size + 1))
        hidden_gradients[:, :-1] = np.outer(hidden_deltas, input_data) # set hidden weight derivatives
        hidden_gradients[:, -1:] = np.reshape(hidden_deltas, (self.hidden_size, 1)) # bias is a fixed input of 1

        # return gradients and error

        difference = expected - output
        error = difference.dot(difference) / len(difference)
        return hidden_gradients, output_gradients, error
    
    # update weights with gradients

    def apply_gradients(self, hidden_gradients, output_gradients):
        self.weights[0] -= self.learning_rate * hidden_gradients
        self.weights[1] -= self.learning_rate * output_gradients
    
    # save model to file

    def save(self, file_path):
        serialized_model = json.dumps(
            {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "learning_rate": self.learning_rate,
                "discount_rate": self.discount_rate,
                "explore_factor": self.explore_factor,
                "weights": [
                    self.weights[0].tolist(),
                    self.weights[1].tolist(),
                ],
            },
            indent=4,
        )
        with open(file_path, "w") as file:
            file.write(serialized_model)
