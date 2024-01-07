'''
Deep Q-Network learning with full state of Pong environment with an artifical
reward for hitting the ball
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
    model = Model.from_save("agent/state_hit_dqn/dqn_models/" + str(checkpoint) + ".json")
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
        50, # hidden size
        2, # output size
        0.001, # learning rate
        0.99, # discount rate
        1, # explore factor
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
sync_interval = 12

transitions = []
buffer_len = 50000
buffer_index = 0

batch_size = 32
explore_decay = 0.997
min_explore = 0.1

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
        
        # advance game state and add artificial reward
        
        prev_velocity = pong.get_normalized_state()[2]
        final_reward = pong.tick(action)

        if final_reward == 0:
            if prev_velocity < 0 and pong.get_normalized_state()[2] > 0:
                final_reward = 1
            else:
                prev_velocity = pong.get_normalized_state()[2]
                final_reward = pong.tick(action)
                if final_reward == 0:
                    if prev_velocity < 0 and pong.get_normalized_state()[2] > 0:
                        final_reward = 1
        
        next_state = pong.get_normalized_state()

        # store state transition

        transition = None
        if final_reward == 0:
            transition = (game_state, action, final_reward, next_state)
        else:
            transition = (game_state, action, final_reward, None)
        game_state = next_state
        
        if len(transitions) < buffer_len:
            transitions.append(transition)
            continue
        else:
            transitions[buffer_index] = transition
            buffer_index = (buffer_index + 1) % buffer_len
        
        # train using random transitions from replay buffer

        train_sample = random.sample(transitions, batch_size)
        hidden_batch = np.zeros((model.hidden_size, model.input_size + 1))
        output_batch = np.zeros((model.output_size, model.hidden_size + 1))
        total_error = 0

        for transition in train_sample:
            # calculate target value using target model

            target_value = transition[2]
            if not (transition[3] is None):
                h, action_values = target_model.forward(transition[3])
                best_value = max(action_values[0], action_values[1])
                target_value += model.discount_rate * best_value

            # back propagate target values through model
            
            hidden_output, predicted_values = model.forward(transition[0])
            target_values = np.copy(predicted_values)
            target_values[transition[1]] = target_value
            hidden_grad, output_grad, error = model.back_prop(
                transition[0],
                hidden_output,
                predicted_values,
                target_values
            )

            hidden_batch += hidden_grad
            output_batch += output_grad
            total_error += error
        
        model.apply_gradients(hidden_batch, output_batch)
    
    # update stats counter
    
    if final_reward == -1:
        losses += 1
    else:
        wins += 1

    # update target model

    if episode_num % sync_interval == 0:
        if model.explore_factor > min_explore:
            model.explore_factor *= explore_decay
        target_model = copy.deepcopy(model)
    
    # reset game environment
        
    pong.reset()
    
    if episode_num % 1000 == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        print("explore factor:", model.explore_factor)

        wins = 0
        losses = 0
    
    if episode_num % 2000 == 0:
        checkpoint += 1
        model.save("agent/state_hit_dqn/dqn_models/" + str(checkpoint) + ".json")