# stochastic gradient descent with full state of Pong environment

from stochastic_model import Model
import numpy as np
import pong_rl
import random

# create or load model

load_model = False
checkpoint = 1
epoch_length = 50

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
episode_num = 0

while True:
    # collect and train agent over epoch

    for e in range(epoch_length):
        episode_num += 1
        episode_states = []
        reward = 0

        pong.start()
        while reward == 0:
            game_state = pong.get_game_state()
            h, action_prob = model.forward(game_state)
            action = 1 if np.random.uniform() < action_prob else 0
            episode_states.append((game_state, action))
            reward = pong.tick(action)
        
        print("FINISHED EPISODE:")
        print(episode_states)
        print("final reward:", reward)
        exit()