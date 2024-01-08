'''
evaluate model performance in Pong environment
'''

'''
5 trials of 100 games (with frame skipping)
results:
state label stochastic checkpoint 8: 50.60 ± 3.72
state label batch checkpoint 8: 53.00 ± 2.90
direct frame label stochastic checkpoint 8: 52.60 ± 5.04
direct frame label batch checkpoint 8: 55.8 ± 5.88
state hit dqn checkpoint 8: 85.40 ± 1.50
state dqn checkpoint 22: 86.60 ± 1.85
'''

from models.stochastic_model import Model as StochasticModel
from models.batch_model import Model as BatchModel
from models.dqn_model import Model as DQNModel
import numpy as np
import pong_rl

# test parameters

Model = StochasticModel
model_type = "state_dqn"
file_path = "agent/state_dqn/dqn_models/22.json"

# load model from file

model = Model.from_save(file_path)
print("loaded model with parameters ({}, {}, {})".format(
    model.input_size,
    model.hidden_size,
    model.output_size,
))

# create game environment

pong = pong_rl.PongEnv.without_render()
pong.start()

num_trials = 5
trial_len = 100
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
    print("finished trial:", t + 1)

# process results

win_rates = [r[0] * 100 / (r[0] + r[1]) for r in records]
mean_win_rate = sum(win_rates) / len(win_rates)
deviation = np.std(win_rates)

print("records:", records)
print("win rates:", win_rates)
print("mean win rate:", mean_win_rate)
print("standard deviation:", deviation)