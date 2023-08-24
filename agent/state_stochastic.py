# stochastic gradient descent with full state of Pong environment

from stochastic_model import Model
import numpy as np
import pong_rl
import random

# create or load model

load_model = False
checkpoint = 1
checkpoint_episodes = 50

model = None
if load_model:
    model = Model.from_save("agent/models/state_stochastic_" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        checkpoint
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        10, # hidden size
        0.01 # learning rate
    )
    print("created new model with parameters ({}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate
    ))

# create Pong environment

pong = pong_rl.PongEnv.with_render()
pong.start()

while True:
    reward = pong.tick(0 if random.random() < 0.5 else 1)
    game_state = pong.get_game_state()
    print("game state:", game_state)
    if reward != 0:
        print("final reward:", reward)
        break