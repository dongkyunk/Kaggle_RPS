import random
import numpy as np
from model.dumb_model import DumbModel  

def make_random_move(my_actions):
    my_action=random.randint(0, 2)
    my_actions=np.append(my_actions, my_action)
    return my_actions, my_action


def update_score(reward, my_actions, op_actions):
    winner=int((3 + my_actions[-1] - op_actions[-1]) % 3)
    if winner == 1:
        # Player won last game
        reward=reward + 1
    elif winner == 2:
        # Opponent won last game
        reward=reward - 1
    return reward


my_actions=np.empty((0, 0), dtype = int)
op_actions=np.empty((0, 0), dtype = int)
reward=0
model=DumbModel()

def sparring_rps_agent(observation, configuration):
    '''
        Rock paper scissor agent

        Objective : Make an agent that successfully predicts opponent's next move based on past interactions
    '''
    global my_actions, op_actions, reward, tactic, model

    # Random moves for first game
    if observation.step == 0:
        my_actions, my_action=make_random_move(my_actions)
        return my_action

    # Update Info
    op_actions=np.append(op_actions, observation.lastOpponentAction)
    update_score(reward, my_actions, op_actions)

    # Train Model
    model.train(my_actions, op_actions, reward)

    # Make Prediction
    my_action=model.action()

    # Update actions
    my_actions=np.append(my_actions, my_action)
    
    return my_action