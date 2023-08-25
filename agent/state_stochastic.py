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
        0.99, # reward discount rate
        0.005, # learning rate
    )
    print("created new model with parameters ({}, {}, {})".format(
        model.input_size,
        model.hidden_size,
        model.learning_rate
    ))

# create Pong environment

pong = pong_rl.PongEnv.without_render()
episode_num = 0

while True:
    # collect and train agent over epoch

    for e in range(epoch_length):
        # initialize epoch data

        pong.start()
        episode_num += 1
        episode_states = []
        final_reward = 0
        
        while final_reward == 0:
            # predict action

            game_state = pong.get_game_state()
            hidden_output, action_prob = model.forward(game_state)
            action = 1 if np.random.uniform() < action_prob else 0

            # store action and get reward

            episode_states.append((
                game_state,
                hidden_output,
                action_prob,
                action,
            ))
            final_reward = pong.tick(action)
        
        # calculate discounted rewards
        
        discounted_rewards = model.discount_rewards(final_reward, len(episode_states))

        # back propagate rewards through model
        # todo: batch with numpy

        for s in range(len(episode_states)):
            model.back_prop(
                episode_states[s][0],
                episode_states[s][1],
                episode_states[s][2],
                episode_states[s][3],
                discounted_rewards[s],
            )
        
        # reset game environment

        pong.reset()
        
        if episode_num % 100 == 0:
            print("FINISHED EPISODE:", episode_num)
            print("final reward:", final_reward)
            print(model.weights[1])