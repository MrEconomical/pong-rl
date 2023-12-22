'''
Deep Q-Network learning with full state of Pong environment
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.dqn_model import Model
import numpy as np
import pong_rl

# create or load model

load_model = False
checkpoint = 0
explore_decay = 0.999

model = None
if load_model:
    model = Model.from_save("agent/state_dqn/dqn_models/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
        model.explore_factor,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        40, # hidden size
        1, # output size
        0.001, # learning rate
        0.99, # discount rate
        0.8, # explore factor
    )
    print("created new model with parameters ({}, {}, {}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
        model.explore_factor,
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0
wins = 0
losses = 0