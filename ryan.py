import random
import numpy as np
from sklearn.metrics import accuracy_score
my_actions = np.empty((0, 0), dtype=int)
op_actions = np.empty((0, 0), dtype=int)
tmp_actions = []
reward = 0
tactic = 0
num_round = 0

def make_random_move(my_actions):
    my_action = random.randint(0, 2)
    my_actions = np.append(my_actions, my_action)
    return my_action

def update_score(reward, my_actions, op_actions):
    winner = int((3 + my_actions[-1] - op_actions[-1]) % 3)
    if winner == 1:
        # Player won last game
        reward = reward + 1
    elif winner == 2:
        # Opponent won last game
        reward = reward - 1
    return reward

def agent(observation, configuration):
    ''' 
        Rock paper scissor agent 
        Objective : Make an agent that successfully predicts opponent's next move based on past interactions
    '''
    global my_actions, op_actions, reward, tactic, step, num_round, tmp_actions, num_round
    if observation.step == 0:
        return make_random_move(my_actions)

    op_actions = np.append(op_actions, observation.lastOpponentAction)
    tmp_actions.append(observation.lastOpponentAction)

    if observation.step % 15 == 0:
        my_action = 0
        if num_round != 0:
            cnt = [0, 0, 0]
            for a in tmp_actions:
                cnt[a]+=1
            my_action = np.argmax(cnt)
            if my_action == 0:
                return 1
            elif my_action == 1:
                return 2
            else:
                return 0
        tmp_actions.clear()
        num_round+=1
        return my_action
    else:
        return make_random_move(my_actions)

