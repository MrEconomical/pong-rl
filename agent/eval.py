'''
evaluate model performance in Pong environment
'''

from models.stochastic_model import Model as StochasticModel
from models.batch_model import Model as BatchModel
from models.dqn_model import Model as DQNModel
from models.policy_model import Model as PolicyModel
import numpy as np
import pong_rl

# test parameters

Model = PolicyModel
model_type = "state_policy"
folder_path = "agent/state_reinforce/reinforce_models"
checkpoint = 50
num_trials = 20
trial_len = 500

# create game environment

pong = pong_rl.PongEnv.without_render()
pong.start()

# load model from file

model = Model.from_save(folder_path + "/" + str(checkpoint) + ".json")
print("loaded model with parameters ({}, {}, {})".format(
    model.input_size,
    model.hidden_size,
    model.output_size,
))
records = []

for t in range(num_trials):
    # run trial and record wins and losses

    record = [0, 0]
    
    for e in range(trial_len):
        prev_frame = pong.get_normalized_frame()
        reward = 0

        while reward == 0:
            # process game state
            
            game_state = pong.get_normalized_state()
            current_frame = pong.get_normalized_frame()

            # select action

            action = None
            if model_type == "state_label":
                h, action_prob = model.forward(game_state)
                action = 1 if np.random.uniform() < action_prob[0] else 0
            elif model_type == "direct_frame_label":
                h, action_prob = model.forward(current_frame)
                action = 1 if np.random.uniform() < action_prob[0] else 0
            elif model_type == "frame_label":
                stacked_frame = np.concatenate((prev_frame, current_frame))
                h, action_prob = model.forward(stacked_frame)
                action = 1 if np.random.uniform() < action_prob[0] else 0
            elif model_type == "state_dqn":
                h, action_values = model.forward(game_state)
                action = 0 if action_values[0] >= action_values[1] else 1
            elif model_type == "state_policy":
                h, action_probs = model.forward(game_state)
                action = np.random.choice(action_probs.size, p=action_probs)

            # advance game with action
            
            reward = pong.tick(action)
            if reward == 0:
                reward = pong.tick(action)
            prev_frame = current_frame
        
        # record final result
        
        pong.reset()
        if reward == 1:
            record[0] += 1
        else:
            record[1] += 1
    
    records.append(record)
    print("finished trial:", t + 1)

# process results

win_rates = [r[0] * 100 / (r[0] + r[1]) for r in records]
mean_win_rate = sum(win_rates) / len(win_rates)
deviation = np.std(win_rates)

print("win rates:", win_rates)
print("mean win rate:", mean_win_rate)
print("standard deviation:", deviation)