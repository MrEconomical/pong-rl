'''
Deep Q-Network learning with full state of Pong environment
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

import copy
from models.dqn_model import Model
import numpy as np
import pong_rl
import random

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
        6, # input size
        40, # hidden size
        2, # output size
        0.005, # learning rate
        0.98, # discount rate
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

# initialize training data

target_model = copy.deepcopy(model)
sync_interval = 200

transitions = []
buffer_len = 20000
buffer_index = 0

batch_size = 64
explore_decay = 0.995

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
            action = 0 if np.random.uniform() < 0.5 else 1
        else:
            h, action_values = model.forward(game_state)
            action = 0 if action_values[0] >= action_values[1] else 1
        
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
    
    # update stats counter
    
    if final_reward == -1:
        losses += 1
    else:
        wins += 1
    
    # train using random transitions from replay buffer
    
    train_sample = random.sample(transitions, min(batch_size, len(transitions)))
    hidden_batch = np.zeros((model.hidden_size, model.input_size + 1))
    output_batch = np.zeros((model.output_size, model.hidden_size + 1))
    total_error = 0

    for transition in train_sample:
        # calculate target values using target model

        target_values = np.full(2, transition[2], dtype=np.float64)
        if not (transition[3] is None):
            h, action_values = target_model.forward(transition[3])
            target_values += model.discount_rate * action_values

        # back propgate target values through model
        
        hidden_output, predicted_values = model.forward(transition[0])
        hidden_grad, output_grad, error = model.back_prop(
            action,
            hidden_output,
            predicted_values,
            target_values
        )

        hidden_batch += hidden_grad
        output_batch += output_grad
        total_error += error
    
    model.apply_gradients(hidden_batch, output_batch)

    # update target model

    if episode_num % sync_interval == 0:
        model.explore_factor *= explore_decay
        target_model = copy.deepcopy(model)
    
    # reset game environment
        
    pong.reset()
    
    if episode_num % 5000 == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        print("average error:", total_error / len(train_sample))

        wins = 0
        losses = 0
    
    if episode_num % 20000 == 0:
        checkpoint += 1
        model.save("agent/state_dqn/dqn_models/" + str(checkpoint) + ".json")