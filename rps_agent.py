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

    def train(self, my_actions, op_actions, reward, step):
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
        self.strategy = "YapSap"
        self.init_frequency = frequency
        self.frequency = frequency
        self.iteration = 0
        self.tactic = 0
        self.model = DecisionTreeModel()
        self.reward_lst = list()

    def _get_zone(self, reward):
        if reward <= -25:
            return "Severe"
        if reward <= -15:
            return "Very Dangerous"
        elif reward <= -10:
            return "Dangerous"
        elif reward <= 10:
            return "Ok"
        elif reward <= 20:
            return "Try Harder"
        elif reward < 25:
            return "Almost There"
        elif reward <= 30:
            return "Pretty Good"
        else:
            return "Relax"

    def _update_frequency(self, reward, step):
        zone = self._get_zone(reward)
        print(zone)
        if zone in ["Severe", "Almost There"]:
            self.frequency = self.init_frequency - random.randint(10, 12)
        elif zone in ["Very Dangerous", "Dangerous", "Try Harder"]:
            self.frequency = self.init_frequency - random.randint(4, 6)
        elif zone == "Ok":
            self.frequency = self.init_frequency
        elif zone == "Pretty Good":
            self.frequency = self.init_frequency + random.randint(4, 6)

    def _update_strategy(self, reward, step):
        self.reward_lst.append(reward)
        if len(self.reward_lst) > self.frequency:
            self.reward_lst = self.reward_lst[-self.frequency:]

        zone = self._get_zone(reward)
        if zone in ["Relax"]:
            self.strategy = "Random"
            return

        # Switch Strategies when keep losing
        if len(self.reward_lst) == self.frequency:
            decreasing = (
                self.reward_lst[self.frequency-1] - self.reward_lst[0]) < 0
            if decreasing and zone in ["Severe", "Very Dangerous"] and step < 800:
                if self.strategy == "Random":
                    self.strategy = "YapSap"
                else:
                    self.strategy = "Random"
                self.reward_lst = list()
                return
            if zone not in ["Severe", "Very Dangerous"]:
                self.strategy = "YapSap"

    def train(self, my_actions, op_actions, reward, step):
        self._update_strategy(reward, step)

        self.tactic = random.randint(0, 2)

        if self.strategy == "Random":
            return

        # Update iteration
        self.iteration = self.iteration + 1

        if self.iteration >= self.frequency:
            # Train and predict
            self.model.train(my_actions, op_actions, reward, step)
            self.tactic = self.model.action()
            # Update values
            self.iteration = 0
            self._update_frequency(reward, step)

    def action(self):
        print("Iteration: {}, Strategy: {}, Frequency: {}".format(
            self.iteration, self.strategy, self.frequency))
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
    reward = update_score(reward, my_actions, op_actions)

    # Train Model
    model.train(my_actions, op_actions, reward, observation.step)

    # Make Prediction
    my_action = model.action()

    # Update actions
    my_actions = np.append(my_actions, my_action)

    return my_action
