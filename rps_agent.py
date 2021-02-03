from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy import stats as s
import random
import numpy as np
from abc import abstractmethod
import math
import secrets
from xgboost import XGBClassifier
import pandas as pd
import time
import collections
import cmath
from typing import List
import traceback
import sys
import operator


class Model():
    def __init__(self):
        self.tactic = 0

    @abstractmethod
    def train(self, my_actions, op_actions, reward):
        pass

    def action(self):
        return self.tactic


basis = np.array(
    [1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)]
)


HistMatchResult = collections.namedtuple("HistMatchResult", "idx length")


def find_all_longest(seq, max_len=None) -> List[HistMatchResult]:
    """
    Find all indices where end of `seq` matches some past.
    """
    result = []

    i_search_start = len(seq) - 2

    while i_search_start > 0:
        i_sub = -1
        i_search = i_search_start
        length = 0

        while i_search >= 0 and seq[i_sub] == seq[i_search]:
            length += 1
            i_sub -= 1
            i_search -= 1

            if max_len is not None and length > max_len:
                break

        if length > 0:
            result.append(HistMatchResult(i_search_start + 1, length))

        i_search_start -= 1

    result = sorted(result, key=operator.attrgetter("length"), reverse=True)

    return result


def probs_to_complex(p):
    return p @ basis


def _fix_probs(probs):
    """
    Put probs back into triangle. Sometimes this happens due to rounding errors or if you
    use complex numbers which are outside the triangle.
    """
    if min(probs) < 0:
        probs -= min(probs)

    probs /= sum(probs)

    return probs


def complex_to_probs(z):
    probs = (2 * (z * basis.conjugate()).real + 1) / 3
    probs = _fix_probs(probs)
    return probs


def z_from_action(action):
    return basis[action]


def sample_from_z(z):
    probs = complex_to_probs(z)
    return np.random.choice(3, p=probs)


def bound(z):
    return probs_to_complex(complex_to_probs(z))


def norm(z):
    return bound(z / abs(z))


class Pred:
    def __init__(self, *, alpha):
        self.offset = 0
        self.alpha = alpha
        self.last_feat = None

    def train(self, target):
        if self.last_feat is not None:
            offset = target * self.last_feat.conjugate()   # fixed

            self.offset = (1 - self.alpha) * self.offset + self.alpha * offset

    def predict(self, feat):
        """
        feat is an arbitrary feature with a probability on 0,1,2
        anything which could be useful anchor to start with some kind of sensible direction
        """
        feat = norm(feat)

        # offset = mean(target - feat)
        # so here we see something like: result = feat + mean(target - feat)
        # which seems natural and accounts for the correlation between target and feat
        # all RPSContest bots do no more than that as their first step, just in a different way

        result = feat * self.offset

        self.last_feat = feat

        return result


class BaseAgent:
    def __init__(self):
        self.my_hist = []
        self.opp_hist = []
        self.my_opp_hist = []
        self.outcome_hist = []
        self.step = None

    def __call__(self, my_actions, op_actions, reward, step, configuration):
        try:
            self.step = step

            opp = op_actions[-1]
            my = my_actions[-1]

            self.my_opp_hist.append((my, opp))
            self.opp_hist.append(opp)

            outcome = {0: 0, 1: 1, 2: -1}[(my - opp) % 3]
            self.outcome_hist.append(outcome)

            action = self.action()

            self.my_hist.append(action)

            return action
        except Exception:
            traceback.print_exc(file=sys.stderr)
            raise

    def action(self):
        pass


class Agent(BaseAgent):
    def __init__(self, alpha=0.01):
        super().__init__()

        self.predictor = Pred(alpha=alpha)

    def action(self):
        self.train()

        pred = self.preds()

        return_action = sample_from_z(pred)

        return return_action

    def train(self):
        last_beat_opp = z_from_action((self.opp_hist[-1] + 1) % 3)
        self.predictor.train(last_beat_opp)

    def preds(self):
        hist_match = find_all_longest(self.my_opp_hist, max_len=20)

        if not hist_match:
            return 0

        feat = z_from_action(self.opp_hist[hist_match[0].idx])

        pred = self.predictor.predict(feat)

        return pred


class Geometry(Model):
    def __init__(self, anti=False):
        self.model = Agent()
        self.anti = anti

    def train(self, my_actions, op_actions, reward, step, configuration):
        if self.anti:
            my_actions, op_actions = op_actions, my_actions
        self.tactic = self.model(
            my_actions, op_actions, reward, step, configuration)
        if self.anti:
            self.tactic = (self.tactic + 1) % 3

    def action(self):
        return int(self.tactic)


class IOU2:
    def __init__(self, anti=False):
        self.num_predictor = 27
        self.len_rfind = [20]
        self.limit = [10, 20, 60]
        self.beat = {"R": "P", "P": "S", "S": "R"}
        self.not_lose = {"R": "PPR", "P": "SSP", "S": "RRS"}  # 50-50 chance
        self.my_his = ""
        self.your_his = ""
        self.both_his = ""
        self.list_predictor = [""]*self.num_predictor
        self.length = 0
        self.temp1 = {"PP": "1", "PR": "2", "PS": "3",
                      "RP": "4", "RR": "5", "RS": "6",
                      "SP": "7", "SR": "8", "SS": "9"}
        self.temp2 = {"1": "PP", "2": "PR", "3": "PS",
                      "4": "RP", "5": "RR", "6": "RS",
                      "7": "SP", "8": "SR", "9": "SS"}
        self.who_win = {"PP": 0, "PR": 1, "PS": -1,
                        "RP": -1, "RR": 0, "RS": 1,
                        "SP": 1, "SR": -1, "SS": 0}
        self.score_predictor = [0]*self.num_predictor
        self.output = random.choice("RPS")
        self.predictors = [self.output]*self.num_predictor
        self.anti = anti

    def train(self, my_actions, op_actions, reward, step, configuration):
        if self.anti:
            my_actions, op_actions = op_actions, my_actions

        T = step
        A = op_actions[-1]
        S = configuration.signs

        to_char = ["R", "P", "S"]
        from_char = {"R": 0, "P": 1, "S": 2}
        if T == 0:
            return from_char[self.output]
        input = to_char[A]

        if len(self.list_predictor[0]) < 5:
            front = 0
        else:
            front = 1
        for i in range(self.num_predictor):
            if self.predictors[i] == input:
                result = "1"
            else:
                result = "0"
            # only 5 rounds before
            self.list_predictor[i] = self.list_predictor[i][front:5]+result
        # history matching 1-6
        self.my_his += self.output
        self.your_his += input
        self.both_his += self.temp1[input+self.output]
        self.length += 1
        for i in range(1):
            len_size = min(self.length, self.len_rfind[i])
            j = len_size
            # self.both_his
            while j >= 1 and not self.both_his[self.length-j:self.length] in self.both_his[0:self.length-1]:
                j -= 1
            if j >= 1:
                k = self.both_his.rfind(
                    self.both_his[self.length-j:self.length], 0, self.length-1)
                self.predictors[0+6*i] = self.your_his[j+k]
                self.predictors[1+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[0+6*i] = random.choice("RPS")
                self.predictors[1+6*i] = random.choice("RPS")
            j = len_size
            # self.your_his
            while j >= 1 and not self.your_his[self.length-j:self.length] in self.your_his[0:self.length-1]:
                j -= 1
            if j >= 1:
                k = self.your_his.rfind(
                    self.your_his[self.length-j:self.length], 0, self.length-1)
                self.predictors[2+6*i] = self.your_his[j+k]
                self.predictors[3+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[2+6*i] = random.choice("RPS")
                self.predictors[3+6*i] = random.choice("RPS")
            j = len_size
            # self.my_his
            while j >= 1 and not self.my_his[self.length-j:self.length] in self.my_his[0:self.length-1]:
                j -= 1
            if j >= 1:
                k = self.my_his.rfind(
                    self.my_his[self.length-j:self.length], 0, self.length-1)
                self.predictors[4+6*i] = self.your_his[j+k]
                self.predictors[5+6*i] = self.beat[self.my_his[j+k]]
            else:
                self.predictors[4+6*i] = random.choice("RPS")
                self.predictors[5+6*i] = random.choice("RPS")

        for i in range(3):
            temp = ""
            search = self.temp1[(self.output+input)]  # last round
            for start in range(2, min(self.limit[i], self.length)):
                if search == self.both_his[self.length-start]:
                    temp += self.both_his[self.length-start+1]
            if(temp == ""):
                self.predictors[6+i] = random.choice("RPS")
            else:
                # take win/lose from opponent into account
                collectR = {"P": 0, "R": 0, "S": 0}
                for sdf in temp:
                    next_move = self.temp2[sdf]
                    if(self.who_win[next_move] == -1):
                        collectR[self.temp2[sdf][1]] += 3
                    elif(self.who_win[next_move] == 0):
                        collectR[self.temp2[sdf][1]] += 1
                    elif(self.who_win[next_move] == 1):
                        collectR[self.beat[self.temp2[sdf][0]]] += 1
                max1 = -1
                p1 = ""
                for key in collectR:
                    if(collectR[key] > max1):
                        max1 = collectR[key]
                        p1 += key
                self.predictors[6+i] = random.choice(p1)
        for i in range(9, 27):
            self.predictors[i] = self.beat[self.beat[self.predictors[i-9]]]
        len_his = len(self.list_predictor[0])
        for i in range(self.num_predictor):
            sum = 0
            for j in range(len_his):
                if self.list_predictor[i][j] == "1":
                    sum += (j+1)*(j+1)
                else:
                    sum -= (j+1)*(j+1)
            self.score_predictor[i] = sum
        max_score = max(self.score_predictor)
        if max_score > 0:
            predict = self.predictors[self.score_predictor.index(max_score)]
        else:
            predict = random.choice(self.your_his)
        self.output = random.choice(self.not_lose[predict])
        self.tactic = from_char[self.output]
        if self.anti:
            self.tactic = (self.tactic + 1) % 3

    def action(self):
        return int(self.tactic)


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


class NumpyPatterns(Model):
    def __init__(self, anti=False):
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
        self.anti = anti

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
        if self.anti:
            my_actions, op_actions = op_actions, my_actions
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
        if self.anti:
            self.tactic = (self.tactic+1) % 3

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
            # model = RandomForestClassifier(n_estimators=25)
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
    def __init__(self, anti=False):
        self.tactic = 0
        self.T = np.zeros((3, 3))
        self.P = np.zeros((3, 3))
        self.a1, self.a2 = None, None
        self.anti = anti

    def train(self, my_actions, op_actions, reward, step, configuration):
        if self.anti:
            my_actions, op_actions = op_actions, my_actions
        self.a1 = op_actions[-1]
        self.T[self.a2, self.a1] += 1
        self.P = np.divide(self.T, np.maximum(
            1, self.T.sum(axis=1)).reshape(-1, 1))
        self.a2 = self.a1
        self.tactic = int(np.random.randint(3))
        if np.sum(self.P[self.a1, :]) == 1:
            self.tactic = int((np.random.choice(
                [0, 1, 2],
                p=self.P[self.a1, :]
            ) + 1) % 3)
        if self.anti:
            self.tactic = (self.tactic+1) % 3

    def action(self):
        return self.tactic


class YapSapModel(Model):
    def __init__(self, frequency=0):
        self.strategy = "IO"
        self.init_frequency = frequency
        self.frequency = frequency
        self.iteration = 0
        self.tactic = 0
        self.strategy_history = {
            "TM": [], "DT": [], "MP": [], "PA": [], "ATM": [], "AMP": [], "IO": [], "AIO": [], "GEO": [], "AGEO": []}
        self.strategy_score = {"TM": [], "DT": [],
                               "MP": [], "PA": [], "ATM": [], "AMP": [], "IO": [], "AIO": [], "GEO": [], "AGEO": []}
        self.strategy_models = {"TM": TransitionMatrix(
        ), "DT": DecisionTreeModel(), "MP": NumpyPatterns(), "PA": PatternAggressive(), "ATM": TransitionMatrix(anti=True),
            "AMP": NumpyPatterns(anti=True), "IO": IOU2(), "AIO": IOU2(anti=True), "GEO": Geometry(), "AGEO": Geometry(anti=True)}

    def _get_zone(self, reward, step):
        # if step < 900:
        # if reward <= -20:
        #     return "Severe"
        # if reward <= -15:
        #     return "Very Dangerous"
        # elif reward <= -10:
        #     return "Dangerous"
        if reward <= -20 and step < 900:
            return "Severe"
        elif reward < 20:
            return "Ok"
        elif reward <= 30:
            return "Pretty Good"
        else:
            return "Relax"
        # else:
        #     if reward <= -20:
        #         return "Severe"
        #     if reward <= -15:
        #         return "Very Dangerous"
        #     elif reward <= -10:
        #         return "Dangerous"
        #     elif reward <= 0:
        #         return "Try Harder"
        #     elif reward <= 15:
        #         return "Almost There"
        #     elif reward < 20:
        #         return "Almost There"
        #     elif reward <= 30:
        #         return "Pretty Good"
        #     else:
        #         return "Relax"

    def _update_frequency(self, reward, step):
        zone = self._get_zone(reward, step)
        # # print(zone)
        if zone in ["Severe"]:
            self.frequency = self.init_frequency - random.randint(5, 7)
        # elif zone in ["Very Dangerous", "Dangerous", "Try Harder"]:
        #     self.frequency = self.init_frequency + random.randint(1, 2)
        if zone in ["Ok"]:
            self.frequency = self.init_frequency - random.randint(0, 1)
        elif zone in ["Pretty Good"]:
            self.frequency = self.init_frequency + random.randint(8, 10)
        elif zone in ["Relax"]:
            self.frequency = self.init_frequency + random.randint(13, 15)

    @staticmethod
    def _measure_performance(strategy_history, op_actions):
        reward_lst = []
        for i in range(len(strategy_history)):
            winner = int((3 + strategy_history[i] - op_actions[i]) % 3)
            if winner == 1:
                # Player won last game
                reward_lst.append(1)
            elif winner == 2:
                # Opponent won last game
                reward_lst.append(-1)
            else:
                reward_lst.append(0)
        reward = np.ma.average(reward_lst, weights=[
                               1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        return reward

    def _update_strategy(self, op_actions, step):
        if self.strategy is not None and len(self.strategy_history[self.strategy]) < 15:
            return
        for strategy in self.strategy_history.keys():
            self.strategy_score[strategy] = self._measure_performance(
                self.strategy_history[strategy][-10:], op_actions[-10:])

            print(strategy, self.strategy_score[strategy])

        self.strategy = max(self.strategy_score, key=self.strategy_score.get)
        # print(strategy, self.strategy_score[self.strategy])

        if self.strategy is not None:
            if self.strategy_score[self.strategy] < 0.2 and step < 900:
                self.strategy = None
            elif self.strategy_score[self.strategy] > 0.7:
                # More aggressive
                self.frequency = self.init_frequency - random.randint(3, 5)
            elif self.strategy_score[self.strategy] > 0.9:
                # More aggressive
                self.frequency = 0
            # else:
            #     self.frequency = self.init_frequency

    def train(self, my_actions, op_actions, reward, step, configuration):
        import random
        random = random.SystemRandom()

        # Update strategy based on history
        self._update_frequency(reward, step)
        self._update_strategy(op_actions, step)

        # Update models and record their predictions
        for strategy in self.strategy_models.keys():
            self.strategy_models[strategy].train(
                my_actions, op_actions, reward, step, configuration)
            self.strategy_history[strategy].append(
                self.strategy_models[strategy].action())

        if self.iteration >= self.frequency:
            if self.strategy is None:
                self.tactic = random.randint(0, 2)
                self.iteration = 0
                return
            # Train and predict
            self.tactic = self.strategy_models[self.strategy].action()
            # Update values
            self.iteration = 0
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
