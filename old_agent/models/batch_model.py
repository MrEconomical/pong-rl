'''
batched gradient descent neural network model class with 1 hidden layer and 
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

    def __init__(self, input_size, hidden_size, discount_rate, learning_rate, weights):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.weights = weights

    # create new model with He and Xavier initialization

    @classmethod
    def with_random_weights(self, input_size, hidden_size, discount_rate, learning_rate):
        hidden_weights = np.empty((hidden_size, input_size + 1))
        hidden_weights[:, :-1] = np.random.randn(hidden_size, input_size) * np.sqrt(2 / input_size) # He initialization
        hidden_weights[:, -1] = 0

        output_weights = np.empty(hidden_size + 1)
        output_weights[:-1] = np.random.randn(hidden_size) * np.sqrt(1 / hidden_size) # Xavier initialization
        output_weights[-1] = 0

        return self(
            input_size,
            hidden_size,
            discount_rate,
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
            model_data["discount_rate"],
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

    # calculate back propagation gradients for episode

    def back_prop(self, inputs, hidden_outputs, probs, advantages):
        # calculate gradients for output neuron using policy gradient theorem

        output_deltas = probs * advantages # using policy gradient theorem
        output_gradient = np.empty(self.hidden_size + 1)
        output_gradient[:-1] = np.dot(hidden_outputs.transpose(), output_deltas) # set output weight derivatives
        output_gradient[-1] = np.sum(output_deltas) # bias is a fixed input of 1

        # calculate gradients for hidden neurons using relu derivative

        hidden_deltas = np.outer(output_deltas, self.weights[1][:-1]) # calculate deltas for each input
        hidden_deltas[hidden_outputs <= 0] = 0 # apply relu derivative
        hidden_gradients = np.empty((self.hidden_size, self.input_size + 1))
        hidden_gradients[:, :-1] = np.dot(hidden_deltas.transpose(), inputs) # set hidden weight derivatives
        hidden_gradients[:, -1] = np.sum(hidden_deltas.transpose(), axis=1)

        return hidden_gradients, output_gradient
    
    # apply gradients and update weights

    def apply_gradients(self, hidden_gradients, output_gradient):
        self.weights[0] -= self.learning_rate * hidden_gradients
        self.weights[1] -= self.learning_rate * output_gradient
    
    # save model to file

    def save(self, file_path):
        serialized_model = json.dumps(
            {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "discount_rate": self.discount_rate,
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
    
    # calculate discounted rewards array with length

    def discount_rewards(self, final_reward, length):
        discounted_rewards = np.empty(length)
        reward = final_reward
        for r in reversed(range(length)):
            discounted_rewards[r] = reward
            reward *= self.discount_rate
        return discounted_rewards
