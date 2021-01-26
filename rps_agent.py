from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats as s
import random
import numpy as np
from abc import abstractmethod


class Model():
    def __init__(self):
        self.tactic = 0

    @abstractmethod
    def train(self, my_actions, op_actions, reward):
        pass

    def action(self):
        return self.tactic


class DecisionTreeModel(Model):
    def __init__(self, frequency=15):
        self.tactic = 0
        self.min_samples = 30
        self.score = 0

    def train(self, my_actions, op_actions, reward):
        if len(my_actions) < 30:
            self.tactic = random.randint(0, 2)
        else:
            # Make training data
            X_train = np.vstack([my_actions[:-1], op_actions[:-1]]).T
            y_train = np.roll(op_actions, -1)[:-1].T

            # Set the history period. Long chains here will need a lot of time
            if len(X_train) > 25:
                random_window_size = 10 + random.randint(0, 10)
                X_test = X_train[-2*random_window_size:]
                y_test = y_train[-2*random_window_size:]
                X_train = X_train[-random_window_size:]
                y_train = y_train[-random_window_size:]

            # Train a classifier model
            model = RandomForestClassifier(n_estimators=25)
            model.fit(X_train, y_train)

            # Calculate Score
            y_pred = model.predict(X_test)
            self.score = accuracy_score(y_test, y_pred)

            # Use prediction if accuracy high
            if self.score >= 0.5:
                curr = np.empty((0, 0), dtype=int)
                curr = np.append(curr, [my_actions[-1], op_actions[-1]])
                prediction = model.predict(curr.reshape(1, -1))
                # print("prediction:", prediction)
                self.tactic = int((prediction + 1) % 3)
            else:
                self.tactic = random.randint(0, 2)

    def action(self):
        # print(self.score)    
        return self.tactic

class YapSapModel(Model):
    def __init__(self, frequency=15):
        self.init_frequency = frequency
        self.frequency = frequency
        self.curr_freq = 0
        self.tactic = 0
        self.model = DecisionTreeModel()
        
    def _frequency_updater(self, reward):
        if reward <= -20:
            self.frequency = self.init_frequency - random.randint(8, 12)
        elif reward <= -10:
            self.frequency = self.init_frequency - random.randint(3, 7)
        else:
            self.frequency = self.init_frequency

    def train(self, my_actions, op_actions, reward):
        # Update current freq
        self.curr_freq = self.curr_freq + 1
        
        if self.curr_freq == self.frequency:
            self._frequency_updater(reward)
            # Train submodel
            self.model.train(my_actions, op_actions, reward)
            self.tactic = self.model.action()
            self.curr_freq = 0
        else:
            self.tactic = random.randint(0, 2)

        # print("self.curr_freq:", self.curr_freq)

    def action(self):
        return self.tactic

    def action(self):
        return self.tactic


def make_random_move(my_actions):
    my_action = random.randint(0, 2)
    my_actions = np.append(my_actions, my_action)
    return my_actions, my_action


def update_score(reward, my_actions, op_actions):
    winner = int((3 + my_actions[-1] - op_actions[-1]) % 3)
    if winner == 1:
        # Player won last game
        reward = reward + 1
    elif winner == 2:
        # Opponent won last game
        reward = reward - 1
    return reward


my_actions = np.empty((0, 0), dtype=int)
op_actions = np.empty((0, 0), dtype=int)
reward = 0
model = YapSapModel()


def rps_agent(observation, configuration):
    '''
        Rock paper scissor agent

        Objective : Make an agent that successfully predicts opponent's next move based on past interactions
    '''
    global my_actions, op_actions, reward, tactic, model

    # Random moves for first game
    if observation.step == 0:
        my_actions, my_action = make_random_move(my_actions)
        return my_action

    # Update Info
    op_actions = np.append(op_actions, observation.lastOpponentAction)
    update_score(reward, my_actions, op_actions)

    # Train Model
    model.train(my_actions, op_actions, reward)

    # Make Prediction
    my_action = model.action()

    # Update actions
    my_actions = np.append(my_actions, my_action)

    return my_action
