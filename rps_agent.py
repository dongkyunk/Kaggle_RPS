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
    def __init__(self, frequency=10, rolling_window_size=5):
        self.tactic = 0
        self.min_samples = 40
        self.score = 0
        self.rolling_window_size = rolling_window_size

    def train(self, my_actions, op_actions, reward, step):
        if len(my_actions) < self.min_samples:
            self.tactic = random.randint(0, 2)
        else:
            # Make One hot encoding
            onehot_my_acts = np.zeros([len(my_actions), 3])
            onehot_op_acts = np.zeros([len(my_actions), 3])
            for i in range(len(my_actions)):
                onehot_my_acts[i][my_actions[i]] = 1
                onehot_op_acts[i][op_actions[i]] = 1

            # Rolling window
            X_train = np.hstack([onehot_my_acts, onehot_op_acts])
            for i in range(1, self.rolling_window_size+1):
                X_train = np.hstack((X_train, np.roll(X_train, i, axis=0)))

            # Prepare data
            curr = X_train[-1]
            X_train = np.roll(X_train, 1, axis=0)
            y_train = onehot_op_acts

            # Set the history period. Long chains here will need a lot of time
            random_window_size = 10 + random.randint(0, 10)
            X_test = X_train[-2*random_window_size:]
            y_test = y_train[-2*random_window_size:]
            X_train = X_train[-random_window_size:]
            y_train = y_train[-random_window_size:]

            # Train a classifier model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Calculate accuracy
            y_pred = model.predict(X_test)
            self.score = accuracy_score(y_test, y_pred)

            # Use prediction if accuracy high
            if self.score >= 0.5:
                prediction = model.predict(curr.reshape(1, -1))
                if 1 in prediction[0].tolist():
                    prediction_proba = model.predict_proba(curr.reshape(1, -1))
                    prediction = [1-prediction_proba[i][0][0]
                                  for i in range(3)]
                    r, p, s = prediction[0], prediction[1], prediction[2]
                    self.tactic = int(np.argmax(np.array([s-p, r-s, p-r])))
            else:
                self.tactic = random.randint(0, 2)

    def action(self):
        return self.tactic


class TransitionMatrix():
    def __init__(self):
        self.tactic = 0
        self.T = np.zeros((3, 3))
        self.P = np.zeros((3, 3))
        self.a1, self.a2 = None, None

    def train(self, my_actions, op_actions, reward, step):
        self.a1 = op_actions[-1]
        self.T[self.a2, self.a1] += 1
        self.P = np.divide(self.T, np.maximum(
            1, self.T.sum(axis=1)).reshape(-1, 1))
        self.a2 = self.a1
        self.tactic = int(np.random.randint(3))
        if np.sum(self.P[self.a1, :]) == 1:
            # prediction = self.P[self.a1, :]
            # r, p, s = prediction[0], prediction[1], prediction[2]
            # self.tactic = int(np.argmax(np.array({s-p, r-s, p-r})))
            self.tactic = int((np.random.choice(
                [0, 1, 2],
                p=self.P[self.a1, :]
            ) + 1) % 3)

    def action(self):
        return self.tactic


class YapSapModel(Model):
    def __init__(self, frequency=5):
        self.strategy = "DT"
        self.init_frequency = frequency
        self.frequency = frequency
        self.iteration = 0
        self.tactic = 0
        self.model = DecisionTreeModel()
        self.prev_reward, self.curr_reward = 0, 0
        self.strategy_score = {"TM": [], "DT": []}

    def _get_zone(self, reward):
        if reward <= -25:
            return "Severe"
        if reward <= -15:
            return "Very Dangerous"
        elif reward <= -10:
            return "Dangerous"
        elif reward <= 10:
            return "Ok"
        elif reward <= 15:
            return "Try Harder"
        elif reward < 23:
            return "Almost There"
        elif reward <= 30:
            return "Pretty Good"
        else:
            return "Relax"

    def _update_frequency(self, reward, step):
        zone = self._get_zone(reward)
        print(zone)
        if zone in ["Severe", "Almost There"]:
            self.frequency = self.init_frequency - random.randint(3, 5)
        elif zone in ["Very Dangerous", "Dangerous", "Try Harder"]:
            self.frequency = self.init_frequency - random.randint(1, 3)
        elif zone in ["Ok"]:
            self.frequency = self.init_frequency
        elif zone in ["Pretty Good", "Relax"]:
            self.frequency = self.init_frequency + random.randint(3, 5)

    def _update_strategy(self, reward, step):
        if len(self.strategy_score[self.strategy]) < 5:
            return
        good_strategy = sum(self.strategy_score[self.strategy][-5:]) > 1
        bad_strategy = sum(self.strategy_score[self.strategy][-5:]) < -1
        print(sum(self.strategy_score[self.strategy][-5:]))

        # Switch strategy
        if bad_strategy:
            if self.strategy == "DT":
                self.strategy = "TM"
                self.model = TransitionMatrix()
            elif self.strategy == "TM":
                self.strategy = "DT"
                self.model = DecisionTreeModel()
        # More aggressive
        if good_strategy:
            self.frequency = self.init_frequency - random.randint(3, 5)

    def train(self, my_actions, op_actions, reward, step):
        import random
        random = random.SystemRandom()

        # Update strategy after every freq over
        if self.iteration == 0:
            self.curr_reward = reward
            self.strategy_score[self.strategy].append(
                self.curr_reward - self.prev_reward)
            self._update_strategy(reward, step)

        # Update iteration
        self.iteration = self.iteration + 1
        if self.strategy == "TM":
            self.model.train(my_actions, op_actions, reward, step)

        if self.iteration >= self.frequency:
            # Train and predict
            if self.strategy == "DT":
                self.model.train(my_actions, op_actions, reward, step)
            self.tactic = self.model.action()
            # Update values
            self._update_frequency(reward, step)
            self.iteration = 0
            self.prev_reward = reward
        else:
            self.tactic = random.randint(0, 2)

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
