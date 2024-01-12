'''
Deep Q-Network learning with stacked Pong frame data
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

save_folder = "dqn_models"
load_model = False
checkpoint = 0
log_interval = 1000
save_interval = 1000
print("save folder: " + save_folder)

model = None
if load_model:
    model = Model.from_save("agent/frame_dqn/" + save_folder + "/" + str(checkpoint) + ".json")
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
        300, # hidden size
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
sync_interval = 8

transitions = []
buffer_len = 40000
buffer_index = 0

batch_size = 32
explore_decay = 0.99975
min_explore = 0.1

while True:
    # initialize episode data

    episode_num += 1
    pong.start()
    final_reward = 0

    prev_frame = pong.get_normalized_frame()
    stacked_frame = np.concatenate((prev_frame, prev_frame))

    while final_reward == 0:
        # predict action

        action = None
        if np.random.uniform() < model.explore_factor:
            action = 0 if np.random.uniform() < 0.5 else 1
        else:
            h, action_values = model.forward(stacked_frame)
            action = 0 if action_values[0] >= action_values[1] else 1
        
        # advance game state
        
        final_reward = pong.tick(action)
        if final_reward == 0:
            final_reward = pong.tick(action)
        next_frame = pong.get_normalized_frame()
        next_stacked_frame = np.concatenate((prev_frame, next_frame))

        # store state transition

        transition = None
        if final_reward == 0:
            transition = (stacked_frame, action, final_reward, next_stacked_frame)
        else:
            transition = (stacked_frame, action, final_reward, None)
        prev_frame = next_frame
        stacked_frame = next_stacked_frame
        
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

        # batch calculate target values using target model

        train_next = np.array([(np.zeros(model.input_size) if t[3] is None else t[3]) for t in train_sample])
        h, action_values = target_model.batch_forward(train_next)
        target_values = np.max(action_values, axis=0) * model.discount_rate
        for t in range(batch_size):
            if not (train_sample[t][3] is None):
                target_values[t] += train_sample[t][2]
            else:
                target_values[t] = train_sample[t][2]
        
        # batch back propagate target values through model
        
        train_current = np.array([t[0] for t in train_sample])
        hidden_outputs, predicted_values = model.batch_forward(train_current)
        update_values = np.copy(np.transpose(predicted_values), order="C")
        for t in range(batch_size):
            update_values[t][train_sample[t][1]] = target_values[t]
        
        hidden_grads, output_grads, e = model.batch_back_prop(
            train_current,
            hidden_outputs,
            predicted_values,
            update_values
        )
        model.apply_gradients(
            np.sum(hidden_grads, axis=0),
            np.sum(output_grads, axis=0)
        )
    
    # update stats counter
    
    if final_reward == -1:
        losses += 1
    else:
        wins += 1

    # decay explore rate and update target model

    if len(transitions) == buffer_len:
        if model.explore_factor > min_explore:
            model.explore_factor *= explore_decay
        if episode_num % sync_interval == 0:
            target_model = copy.deepcopy(model)
    
    # reset game environment
        
    pong.reset()
    
    if episode_num % log_interval == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        print("explore factor:", model.explore_factor)

        wins = 0
        losses = 0
    
    if episode_num % save_interval == 0:
        checkpoint += 1
        model.save("agent/frame_dqn/" + save_folder + "/" + str(checkpoint) + ".json")