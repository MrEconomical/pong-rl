'''
policy gradient model class with 1 hidden layer and 1 output layer using relu
for the hidden layer and implicit softmax activation for the output layer with
batched gradient updates
'''

import json
import numpy as np

class Model:
    input_size = None
    hidden_size = None
    output_size = None
    learning_rate = None
    discount_rate = None
    weights = None

    # set model data

    def __init__(self, input_size, hidden_size, output_size, learning_rate, discount_rate, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.weights = weights

    # create new model with He and Xavier initialization

    @classmethod
    def with_random_weights(self, input_size, hidden_size, output_size, learning_rate, discount_rate):
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
        np.exp(output - np.max(output), out=output) # softmax activation
        output /= np.sum(output)

        return hidden_output, output

    # calculate gradients with back propagation

    def back_prop(self, input_data, hidden_output, output, action, reward):
        # calculate gradients for output neuron using softmax derivative

        output_deltas = (output - action) * reward # using softmax derivative and policy gradient
        output_gradients = np.empty((self.output_size, self.hidden_size + 1))
        output_gradients[:, :-1] = np.outer(output_deltas, hidden_output) # set output weight derivatives
        output_gradients[:, -1:] = np.reshape(output_deltas, (self.output_size, 1)) # bias is a fixed input of 1

        # calculate gradients for hidden neurons using relu derivative

        hidden_predeltas = np.dot(output_deltas, self.weights[1][:, :-1]) # find total error per neuron
        hidden_deltas = hidden_predeltas * (hidden_output > 0) # using relu derivative
        hidden_gradients = np.empty((self.hidden_size, self.input_size + 1))
        hidden_gradients[:, :-1] = np.outer(hidden_deltas, input_data) # set hidden weight derivatives
        hidden_gradients[:, -1:] = np.reshape(hidden_deltas, (self.hidden_size, 1)) # bias is a fixed input of 1

        # return gradients

        return hidden_gradients, output_gradients
    
    def batch_back_prop(self, input_batch, hidden_outputs, outputs, actions, rewards):
        # TODO: investigate precision loss
        # calculate gradients for output neuron using softmax derivative

        batch_len = len(input_batch)
        output_deltas = (outputs - actions) * rewards.reshape(-1, 1) # using softmax derivative and policy gradient
        output_gradients = np.empty((batch_len, self.output_size, self.hidden_size + 1))
        output_gradients[:, :, :-1] = np.matmul(
            np.reshape(output_deltas, (batch_len, self.output_size, 1)), # stack output deltas for weight derivatives
            np.reshape(hidden_outputs, (batch_len, 1, self.hidden_size)) # stack hidden outputs
        )
        output_gradients[:, :, -1:] = np.reshape(output_deltas, (batch_len, self.output_size, 1)) # bias is a fixed input

        # calculate gradients for hidden neurons using relu derivative

        hidden_predeltas = np.dot(output_deltas, self.weights[1][:, :-1]) # find total error per neuron
        hidden_deltas = hidden_predeltas * (hidden_outputs > 0) # using relu derivative
        hidden_gradients = np.empty((batch_len, self.hidden_size, self.input_size + 1))
        hidden_gradients[:, :, :-1] = np.matmul(
            np.reshape(hidden_deltas, (batch_len, self.hidden_size, 1)), # stack hidden deltas for weight derivatives
            np.reshape(input_batch, (batch_len, 1, self.input_size)) # stack input batch
        )
        hidden_gradients[:, :, -1:] = np.reshape(hidden_deltas, (batch_len, self.hidden_size, 1)) # bias is a fixed input

        # return gradients

        return hidden_gradients, output_gradients
    
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
                "weights": [
                    self.weights[0].tolist(),
                    self.weights[1].tolist(),
                ],
            },
            indent=4,
        )
        with open(file_path, "w") as file:
            file.write(serialized_model)