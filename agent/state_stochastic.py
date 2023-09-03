# stochastic gradient descent with full state of Pong environment

from stochastic_model import Model
import numpy as np
import pong_rl

# create or load model

load_model = False
checkpoint = 0
epoch_length = 100

model = None
if load_model:
    model = Model.from_save("agent/models/state_stochastic/" + str(checkpoint) + ".json")
    print("loaded model with parameters ({}, {}, {}) from checkpoint {}".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
        checkpoint,
    ))
else:
    model = Model.with_random_weights(
        6, # input size
        20, # hidden size
        0.99, # reward discount rate
        0.01, # learning rate
    )
    print("created new model with parameters ({}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate,
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0
wins = 0
losses = 0

while True:
    # collect and train agent over epoch

    for e in range(epoch_length):
        # initialize epoch data

        pong.start()
        episode_num += 1
        episode_states = []
        episode_hidden_outputs = []
        episode_probs = []
        final_reward = 0
        
        while final_reward == 0:
            # predict action

            game_state = pong.get_normalized_state()
            hidden_output, action_prob = model.forward(game_state)
            action = 1 if np.random.uniform() < action_prob else 0

            # store action data and get reward

            episode_states.append(game_state)
            episode_hidden_outputs.append(hidden_output)
            episode_probs.append(action - action_prob)
            final_reward = pong.tick(action)
        
        if final_reward == -1:
            losses += 1
        else:
            wins += 1
        
        # back propagate discounted rewards through model
        
        discounted_rewards = model.discount_rewards(final_reward, len(episode_states))
        model.back_prop(
            np.array(episode_states),
            np.array(episode_hidden_outputs),
            np.array(episode_probs),
            discounted_rewards,
        )
        
        # reset game environment

        pong.reset()

        if episode_num % 1000 == 0:
            print("FINISHED EPISODE:", episode_num)
            print("wins and losses:", wins, losses)
            print(model.weights[1][0:5])
        if episode_num % 100000 == 0:
            checkpoint += 1
            model.save("agent/models/state_stochastic/" + str(checkpoint) + ".json")