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
        7, # input size
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

pong = pong_rl.PongEnv.with_render()
episode_num = 0
wins = 0
losses = 0

transitions = []
buffer_len = 10000
buffer_index = 0
explore_decay = 0.999

while True:
    # initialize episode data

    episode_num += 1
    pong.start()
    game_state = pong.get_normalized_state()
    final_reward = 0

    while final_reward == 0:
        # predict action

        action = None
        if np.random.uniform() < model.explore_factor:
            action = 1 if np.random.uniform() < 0.5 else 0
        else:
            h, up_value = model.forward(np.append(game_state, 1))
            h, down_value = model.forward(np.append(game_state, 0))
            action = 1 if up_value[0] >= down_value[0] else 0
        
        # advance game state
            
        final_reward = pong.tick(action)
        if final_reward == 0:
            final_reward = pong.tick(action)
        next_state = pong.get_normalized_state()

        # store state transition

        transition = None
        if final_reward == 0:
            transition = (game_state, action, final_reward, next_state)
        else:
            transition = (game_state, action, final_reward, None)
        
        if len(transitions) < buffer_len:
            transitions.append(transition)
        else:
            transitions[buffer_index] = transition
            buffer_index = (buffer_index + 1) % buffer_len
    
    # sample state transitions from replay buffer
            
    print(transitions)
    break