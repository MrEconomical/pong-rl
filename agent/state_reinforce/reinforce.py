'''
REINFORCE policy gradient with full state of Pong environment
'''

from pathlib import Path
import sys
sys.path.insert(0, str(Path(Path(__file__).parent.absolute()).parent.absolute()))

from models.array_vec import ArrayVec
from models.policy_model import Model
import numpy as np
import pong_rl

# create or load model

save_folder = "reinforce_models"
load_model = True
checkpoint = 27
log_interval = 8000
save_interval = 16000
print("save folder: " + save_folder)

model = None
if load_model:
    model = Model.from_save("agent/state_reinforce/" + save_folder + "/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        600, # hidden size
        2, # output size
        0.0001, # learning rate
        0.99, # discount rate
    )
    print("created new model with parameters ({}, {}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        model.discount_rate,
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0
wins = 0
losses = 0

# initialize training data

batch_size = 1600
sample_size = 800
sample_split = 40000
initial_len = 200000
extend_len = 20000

sample_states = ArrayVec((model.input_size,), initial_len, extend_len)
sample_hidden_outputs = ArrayVec((model.hidden_size,), initial_len, extend_len)
sample_probs = ArrayVec((model.output_size,), initial_len, extend_len)
sample_rewards = []

hidden_batch = np.zeros((model.hidden_size, model.input_size + 1))
output_batch = np.zeros((model.output_size, model.hidden_size + 1))

while True:
    # initialize episode data

    pong.start()
    episode_num += 1
    num_states = 0
    final_reward = 0

    while final_reward == 0:
        # predict action

        game_state = pong.get_normalized_state()
        hidden_output, action_probs = model.forward(game_state)
        action = np.random.choice(action_probs.size, p=action_probs)

        # store state and action data

        num_states += 1
        sample_states.push(game_state)
        sample_hidden_outputs.push(hidden_output)
        action_probs[action] -= 1
        sample_probs.push(action_probs)

        # advance game state

        final_reward = pong.tick(action)
        if final_reward == 0:
            final_reward = pong.tick(action)

    # calculate discounted rewards

    for s in range(num_states):
        reward = final_reward * (model.discount_rate ** (num_states - s - 1))
        sample_rewards.append(reward)

    if episode_num % sample_size == 0:
        # normalize sample rewards

        sample_rewards = np.array(sample_rewards)
        sample_rewards -= np.mean(sample_rewards)
        sample_rewards /= np.std(sample_rewards)

        # calculate sample policy gradients

        for s in range(0, sample_states.len(), sample_split):
            hidden_grads, output_grads = model.batch_back_prop(
                sample_states.get_view(s, s + sample_split),
                sample_hidden_outputs.get_view(s, s + sample_split),
                sample_probs.get_view(s, s + sample_split),
                sample_rewards[s:s + sample_split],
            )
            hidden_batch += np.sum(hidden_grads, axis=0)
            output_batch += np.sum(output_grads, axis=0)

        # reset sample data

        sample_states.clear()
        sample_hidden_outputs.clear()
        sample_probs.clear()
        sample_rewards = []

    if episode_num % batch_size == 0:
        # apply policy gradients

        model.apply_gradients(hidden_batch, output_batch)
        hidden_batch.fill(0)
        output_batch.fill(0)

    # update stats counter
    
    if final_reward == -1:
        losses += 1
    else:
        wins += 1

    # reset game environment
        
    pong.reset()
    
    if episode_num % log_interval == 0:
        print("FINISHED EPISODE:", episode_num)
        print("wins and losses:", wins, losses)
        wins = 0
        losses = 0
    
    if episode_num % save_interval == 0:
        checkpoint += 1
        model.save("agent/state_reinforce/" + save_folder + "/" + str(checkpoint) + ".json")