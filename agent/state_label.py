# labeled learning with full state of Pong environment

from mse_model import Model
import numpy as np
import pong_rl

# create or load model

load_model = False
checkpoint = 1
epoch_length = 50

model = None
if load_model:
    model = Model.from_save("agent/models/state_label/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        checkpoint
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        40, # hidden size
        0.01, # learning rate
    )
    print("created new model with parameters ({}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate
    ))