from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import collections
import numpy as np
import random
from model.model import Model
from scipy import stats as s
from statistics import mode


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
