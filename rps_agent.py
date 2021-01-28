from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from scipy import stats as s
import random
import numpy as np
from abc import abstractmethod
import math
import secrets
from xgboost import XGBClassifier
import pandas as pd

class Model():
    def __init__(self):
        self.tactic = 0

    @abstractmethod
    def train(self, my_actions, op_actions, reward):
        pass

    def action(self):
        return self.tactic

class Xgboost(Model):
    def __init__(self):
        self.numTurnsPredictors = 5 #number of previous turns to use as predictors
        self.minTrainSetRows = 10 #only start predicting moves after we have enough data
        self.myLastMove = None
        self.mySecondLastMove = None
        self.opponentLastMove = None
        self.numDummies = 2 #how many dummy vars we need to represent a move
        self.predictors = pd.DataFrame(columns=[str(x) for x in range(self.numTurnsPredictors * 2 * self.numDummies)]).astype("int")
        self.opponentsMoves = [0] * 1000
        self.roundHistory = [None] * 1000
        self.dummies = [[[0,0,0,0], [0,1,0,0], [1,0,0,0]], [[0,0,0,1], [0,1,0,1], [1,0,0,1]], [[0,0,1,0], [0,1,1,0], [1,0,1,0]]]
        self.clf = XGBClassifier(n_estimators=10)

    def updateFeatures(self, rounds):
        self.predictors.loc[len(self.predictors)] = sum(rounds, [])

    def fitAndPredict(self, x, y, newX):
        self.clf.fit(x.values, y)
        return int(self.clf.predict(np.array(newX).reshape((1,-1)))[0])

    def train(self, my_actions, op_actions, reward, step, configuration):
        T = step
        A = op_actions[-1]
        S = configuration.signs
        self.myLastMove = A
        self.roundHistory[T-1] = self.dummies[self.myLastMove][A]
        if T == 1:
            self.myLastMove = secrets.randbelow(S)
        else:
            self.opponentsMoves[T-2] = A
            if T > self.numTurnsPredictors:
                self.updateFeatures(self.roundHistory[:T][-self.numTurnsPredictors - 1: -1])

            if len(self.predictors) > self.minTrainSetRows:
                predictX = sum(self.roundHistory[:T][-self.numTurnsPredictors:], []) #data to predict next move
                predictedMove = self.fitAndPredict(self.predictors, self.opponentsMoves[:T-1][(self.numTurnsPredictors-1):], predictX)
                self.myLastMove = (predictedMove + 1) % S
            else:
                self.myLastMove = secrets.randbelow(S)

    def action(self):
        return int(self.myLastMove)          

class PatternAggressive(Model):
    def __init__(self):
        self.Jmax = 2
        self.J = self.Jmax - \
            int(math.sqrt(secrets.randbelow((self.Jmax+1)**2)))
        self.Dmin = 2
        self.Dmax = 5
        self.Hash = []
        self.Map = []
        self.MyMap = []
        for D in range(self.Dmin, self.Dmax+1):
            self.Hash.append([0, 0, 0])
            self.Map.append([{}, {}, {}])
            self.MyMap.append([{}, {}, {}])
        self.G = 2
        self.R = 0.4
        self.V = 0.8
        self.VM = 0.95
        self.tactic = 0
        self.DT = 200

    def add(self, map1, hash1, A, T):
        if hash1 not in map1:
            map1[hash1] = {'S': []}
        d = map1[hash1]
        if A not in d:
            d[A] = [T]
        else:
            d[A].append(T)
        d['S'].append(T)

    def rank(self, A, T):
        return len([a for a in A if a > T - self.DT])

    def match(self, map1, hash1, S, T):
        if hash1 not in map1:
            return
        d = map1[hash1]
        if self.rank(d['S'], T) >= self.G:
            for A in range(S):
                if A in d and (self.rank(d[A], T) >= self.rank(d['S'], T) * self.R + (1-self.R) * self.G) and secrets.randbelow(1001) < 1000 * self.V:
                    if secrets.randbelow(1001) < 1000 * self.VM:
                        self.tactic = (A+1) % S
                    else:
                        self.tactic = A % S
                    self.J = self.Jmax - \
                        int(math.sqrt(secrets.randbelow((self.Jmax+1)**2)))

    def train(self, my_actions, op_actions, reward, step, configuration):
        T = step
        A = op_actions[-1]
        S = configuration.signs
        BA = (self.tactic+1) % S
        self.tactic = secrets.randbelow(S)
        for D in range(self.Dmin, self.Dmax+1):
            if T > D:
                self.add(self.Map[D-self.Dmin][0],
                         self.Hash[D-self.Dmin][0], A, T)
                self.add(self.Map[D-self.Dmin][1],
                         self.Hash[D-self.Dmin][1], A, T)
                self.add(self.Map[D-self.Dmin][2],
                         self.Hash[D-self.Dmin][2], A, T)
                self.add(self.MyMap[D-self.Dmin][0],
                         self.Hash[D-self.Dmin][0], BA, T)
                self.add(self.MyMap[D-self.Dmin][1],
                         self.Hash[D-self.Dmin][1], BA, T)
                self.add(self.MyMap[D-self.Dmin][2],
                         self.Hash[D-self.Dmin][2], BA, T)
            if T > 0:
                self.Hash[D-self.Dmin][0] = self.Hash[D -
                                                      self.Dmin][0] // S**2 + (A + S*self.tactic) * S**(2*D-1)
                self.Hash[D-self.Dmin][1] = self.Hash[D -
                                                      self.Dmin][1] // S + A * S**(D-1)
                self.Hash[D-self.Dmin][2] = self.Hash[D -
                                                      self.Dmin][2] // S + self.tactic * S**(D-1)
            if self.J == 0:
                self.match(self.Map[D-self.Dmin][0],
                           self.Hash[D-self.Dmin][0], S, T)
                self.match(self.Map[D-self.Dmin][1],
                           self.Hash[D-self.Dmin][1], S, T)
                self.match(self.Map[D-self.Dmin][2],
                           self.Hash[D-self.Dmin][2], S, T)
            if self.J == 0:
                self.match(self.MyMap[D-self.Dmin][0],
                           self.Hash[D-self.Dmin][0], S, T)
                self.match(self.MyMap[D-self.Dmin][1],
                           self.Hash[D-self.Dmin][1], S, T)
                self.match(self.MyMap[D-self.Dmin][2],
                           self.Hash[D-self.Dmin][2], S, T)
        if self.J > 0:
            self.J -= 1

    def action(self):
        return int(self.tactic)


class NumpyPatterns:
    K = 20

    def __init__(self):
        self.tactic = 0
        # Jitter - steps before next non-random move
        self.Jmax = 2
        self.J2 = (self.Jmax+1)**2
        self.J = self.Jmax - int(math.sqrt(secrets.randbelow(self.J2)))
        # Depth - number of previous steps taken into consideration
        self.Dmin = 1
        self.Dmax = 3
        self.DL = self.Dmax-self.Dmin+1
        self.HL = 3
        self.HText = ['Opp',  'Me', 'Score']
        self.Depth = np.arange(self.DL)
        self.Hash = np.zeros((self.HL, self.DL), dtype=int)
        self.G = 2
        self.R = 0.4
        self.RG = (1-self.R) * self.G
        self.Threshold = 0.4

        S = 3
        B, HL, DL, Dmin, Dmax = self.tactic, self.HL, self.DL, self.Dmin, self.Dmax
        SD = S**self.DL

        self.Map = np.zeros((S, SD**2, HL, HL, DL))
        self.SList = np.arange(S)[:, None, None, None]
        self.Predicts = np.full((HL, HL, DL), S, dtype=int)
        self.Attempts = np.zeros((HL, HL, DL), dtype=int)
        self.Scores = np.zeros((S, HL, HL, DL))
        self.OrgID = np.ogrid[:S, :HL, :HL, :DL]
        self.Hash2 = self.Hash[None, :] + SD*self.Hash[:, None]

    def get_score(self, S, A1, A2):
        return (S + A1 - A2 + 1) % S - 1

    def split_idx(self, idx):
        d = idx % self.DL
        idx //= self.DL
        h2 = idx % self.HL
        idx //= self.HL
        h1 = idx % self.HL
        idx //= self.HL
        return d, h1, h2, idx

    def train(self, my_actions, op_actions, reward, step, configuration):
        T = step
        A = op_actions[-1]
        S = configuration.signs
        B, HL, DL, Dmin, Dmax = self.tactic, self.HL, self.DL, self.Dmin, self.Dmax
        SD = S**self.DL

        C = self.get_score(S, A, B) + 1
        ABC = np.array([A, B, C])[:, None]
        Depth, Hash, Hash2, Map, SList, OrgID, Predicts, Attempts, Scores = self.Depth, self.Hash, self.Hash2, self.Map, self.SList, self.OrgID, self.Predicts, self.Attempts, self.Scores
        # Update Moves Map by previous move and previous Hash
        Map *= 0.995
        Map[OrgID[0], Hash2, OrgID[1], OrgID[2], OrgID[3]
            ] += (T > Depth + Dmin) * (SList == A)
        # Update Hash by previous move
        Hash[:] //= S
        Hash[:] += ABC[:HL] * S**Depth
        Hash2[:] = Hash[None, :] + SD*Hash[:, None]

        # Update prediction scores by previous move
        PB = Predicts < S
        Attempts[:] = Attempts + PB
        Scores[:] += PB * self.get_score(S, Predicts + SList, A)
        # print(T, Scores.T[0])
        # Update prediction scores by previous move
        PR = Map[OrgID[0], Hash2, OrgID[1], OrgID[2], OrgID[3]]
        Sum = np.sum(PR, axis=0)
        Predicts[:] = (np.max((Sum >= self.G) * (PR >= Sum *
                                                 self.R + self.RG) * (SList + 1), axis=0) - 1) % (S + 1)

        self.tactic = np.random.choice(S)
        if self.J > 0:
            self.J -= 1
        else:
            sc = np.where(self.Predicts < S, self.Scores /
                          (self.Attempts + 2), 0).ravel()
            idx = np.argmax(sc)
            if sc[idx] > self.Threshold:
                Raw = self.Predicts.ravel()
                L = len(Raw)
                self.tactic = (Raw[idx % L] + idx // L) % S
                self.J = self.Jmax - int(math.sqrt(secrets.randbelow(self.J2)))
                # parts = self.split_idx(idx)
                # print(T, f'{parts[0]+self.Dmin}: {self.HText[parts[1]]}-{self.HText[parts[2]]}+{parts[3]}', self.Scores[:, parts[1], parts[2], parts[0]], self.B)

    def action(self):
        return int(self.tactic)


class DecisionTreeModel(Model):
    def __init__(self, min_samples=50, rolling_window_size=5):
        self.tactic = 0
        self.min_samples = min_samples
        self.score = 0
        self.rolling_window_size = rolling_window_size

    def train(self, my_actions, op_actions, reward, step, configuration):
        if len(my_actions) < self.min_samples:
            self.tactic = random.randint(0, 2)
        else:
            # Make One hot encoding
            onehot_acts = np.zeros(
                [len(my_actions)-self.rolling_window_size+1, 30])
            y_train = np.zeros([len(my_actions)-self.rolling_window_size, 3])
            for i in range(self.rolling_window_size, len(my_actions)+1):
                for j in range(1, self.rolling_window_size+1):
                    onehot_acts[i-self.rolling_window_size][my_actions[i-j] +
                                                            self.rolling_window_size+1*(j-1)] = 1
                    onehot_acts[i-self.rolling_window_size][op_actions[i-j] +
                                                            self.rolling_window_size+1*(j-1)+3] = 1
                if i != len(my_actions):
                    y_train[i-self.rolling_window_size][op_actions[i]] = 1
            X_train = onehot_acts

            # Prepare data
            curr = X_train[-1:]
            X_train = X_train[:-1]

            # Set the history period. Long chains here will need a lot of time
            random_window_size = 10 + random.randint(0, 10)
            X_test = X_train[-2*random_window_size:]
            y_test = y_train[-2*random_window_size:]
            X_train = X_train[-random_window_size:]
            y_train = y_train[-random_window_size:]

            # Train a classifier model
            model = DecisionTreeClassifier()
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
        # print(self.score)
        return self.tactic


class TransitionMatrix():
    def __init__(self):
        self.tactic = 0
        self.T = np.zeros((3, 3))
        self.P = np.zeros((3, 3))
        self.a1, self.a2 = None, None

    def train(self, my_actions, op_actions, reward, step, configuration):
        self.a1 = op_actions[-1]
        self.T[self.a2, self.a1] += 1
        self.P = np.divide(self.T, np.maximum(
            1, self.T.sum(axis=1)).reshape(-1, 1))
        self.a2 = self.a1
        self.tactic = int(np.random.randint(3))
        if np.sum(self.P[self.a1, :]) == 1:
            prediction = self.P[self.a1, :]
            r, p, s = prediction[0], prediction[1], prediction[2]
            self.tactic = int(np.argmax(np.array({s-p, r-s, p-r})))
            # self.tactic = int((np.random.choice(
            #     [0, 1, 2],
            #     p=self.P[self.a1, :]
            # ) + 1) % 3)

    def action(self):
        return self.tactic


class YapSapModel(Model):
    def __init__(self, frequency=10):
        self.strategy = "DT"
        self.init_frequency = frequency
        self.frequency = frequency
        self.iteration = 0
        self.tactic = 0
        self.prev_reward, self.curr_reward = 0, 0
        self.strategy_history = {"TM": [], "DT": [], "MP": [], "PA": [], "XG" : []}
        self.strategy_score = {"TM": [], "DT": [], "MP": [], "PA": [], "XG" : []}
        self.strategy_models = {"TM": TransitionMatrix(
        ), "DT": DecisionTreeModel(), "MP": NumpyPatterns(), "PA": PatternAggressive(), "XG" : Xgboost()}

    def _get_zone(self, reward, step):
        if step < 900:
            if reward <= -20:
                return "Severe"
            if reward <= -15:
                return "Very Dangerous"
            elif reward <= -10:
                return "Dangerous"
            elif reward <= 10:
                return "Ok"
            elif reward <= 15:
                return "Try Harder"
            elif reward < 20:
                return "Almost There"
            elif reward <= 30:
                return "Pretty Good"
            else:
                return "Relax"
        else:
            if reward <= -20:
                return "Severe"
            if reward <= -15:
                return "Very Dangerous"
            elif reward <= -10:
                return "Dangerous"
            elif reward <= 0:
                return "Try Harder"
            elif reward <= 15:
                return "Almost There"
            elif reward < 20:
                return "Almost There"
            elif reward <= 30:
                return "Pretty Good"
            else:
                return "Relax"

    def _update_frequency(self, reward, step):
        zone = self._get_zone(reward, step)
        print(zone)
        if zone in ["Severe", "Almost There"]:
            self.frequency = self.init_frequency - random.randint(7, 10)
        elif zone in ["Very Dangerous", "Dangerous", "Try Harder"]:
            self.frequency = self.init_frequency - random.randint(4, 7)
        elif zone in ["Ok"]:
            self.frequency = self.init_frequency
        elif zone in ["Pretty Good", "Relax"]:
            self.frequency = self.init_frequency + random.randint(7, 10)

    @staticmethod
    def _measure_performance(strategy_history, op_actions):
        reward = 0
        for i in range(len(strategy_history)):
            winner = int((3 + strategy_history[i] - op_actions[i]) % 3)
            if winner == 1:
                # Player won last game
                reward = reward + 1
            elif winner == 2:
                # Opponent won last game
                reward = reward - 1
        return reward

    def _update_strategy(self, op_actions, step):
        if self.strategy is not None and len(self.strategy_history[self.strategy]) < 10:
            return
        for strategy in self.strategy_history.keys():
            print(strategy, self.strategy_score[strategy])
            self.strategy_score[strategy] = self._measure_performance(
                self.strategy_history[strategy][-5:], op_actions[1:][-5:])

        self.strategy = max(self.strategy_score, key=self.strategy_score.get)

        if self.strategy is not None:
            if self.strategy_score[self.strategy] < 1 and step < 800:
                self.strategy = None
            elif self.strategy_score[self.strategy] == 4:
                # More aggressive
                self.frequency = self.init_frequency - random.randint(7, 9)
            elif self.strategy_score[self.strategy] == 5:
                # More aggressive
                self.frequency = 0

    def train(self, my_actions, op_actions, reward, step, configuration):
        import random
        random = random.SystemRandom()

        # Update strategy based on history
        self._update_strategy(op_actions, step)

        # Update models and record their predictions
        for strategy in self.strategy_models.keys():
            self.strategy_models[strategy].train(
                my_actions, op_actions, reward, step, configuration)
            self.strategy_history[strategy].append(
                self.strategy_models[strategy].action())

        if self.iteration >= self.frequency:
            if self.strategy is None:
                return
            # Train and predict
            self.tactic = self.strategy_models[self.strategy].action()
            # Update values
            self.iteration = 0
            self.prev_reward = reward
            self._update_frequency(reward, step)
        else:
            self.tactic = random.randint(0, 2)
            self.iteration = self.iteration + 1

    def action(self):
        print("Iteration: {}, Strategy: {}, Frequency: {}".format(
            self.iteration, self.strategy, self.frequency))
        return self.tactic


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
        my_action = random.randint(0, 2)
        my_actions = np.append(my_actions, my_action)
        return my_action

    # Update Info
    op_actions = np.append(op_actions, observation.lastOpponentAction)
    reward = update_score(reward, my_actions, op_actions)

    # Train Model
    model.train(my_actions, op_actions, reward,
                observation.step, configuration)

    # Make Prediction
    my_action = model.action()

    # Update actions
    my_actions = np.append(my_actions, my_action)

    return my_action
