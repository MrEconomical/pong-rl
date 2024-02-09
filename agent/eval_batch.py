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
checkpoint_range = [1, 50]
num_trials = 3
trial_len = 100
print("save folder:", folder_path)

# create game environment

pong = pong_rl.PongEnv.without_render()
pong.start()

for checkpoint in range(checkpoint_range[0], checkpoint_range[1] + 1):
    # load model from file

    model = Model.from_save(folder_path + "/" + str(checkpoint) + ".json")
    records = []

    for t in range(num_trials):
        # run trial and record wins and losses

        record = [0, 0]
        for e in range(trial_len):
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
            
            # record final result
            
            pong.reset()
            if reward == 1:
                record[0] += 1
            else:
                record[1] += 1
        
        records.append(record)

    # process results

    win_rates = [r[0] * 100 / (r[0] + r[1]) for r in records]
    mean_win_rate = sum(win_rates) / len(win_rates)
    deviation = np.std(win_rates)
    print(str(checkpoint) + ":", mean_win_rate, "±", deviation)