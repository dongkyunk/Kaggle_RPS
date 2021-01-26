import operator
import sys
import traceback
from collections import namedtuple
from typing import List
import cmath
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import collections
import pandas as pd
import random
my_last_action = 0


def hit_the_last_own_action(observation, configuration):
    global my_last_action
    my_last_action = (my_last_action + 1) % 3

    return my_last_action


def copy_opponent(observation, configuration):
    if observation.step > 0:
        return observation.lastOpponentAction
    else:
        return random.randrange(0, configuration.signs)


last_react_action = None


def reactionary(observation, configuration):
    global last_react_action
    if observation.step == 0:
        last_react_action = random.randrange(0, configuration.signs)
    elif get_score(last_react_action, observation.lastOpponentAction) <= 1:
        last_react_action = (
            observation.lastOpponentAction + 1) % configuration.signs

    return last_react_action


last_counter_action = None


def counter_reactionary(observation, configuration):
    global last_counter_action
    if observation.step == 0:
        last_counter_action = random.randrange(0, configuration.signs)
    elif get_score(last_counter_action, observation.lastOpponentAction) == 1:
        last_counter_action = (last_counter_action + 2) % configuration.signs
    else:
        last_counter_action = (
            observation.lastOpponentAction + 1) % configuration.signs

    return last_counter_action


action_histogram = {}


def statistical(observation, configuration):
    global action_histogram
    if observation.step == 0:
        action_histogram = {}
        return
    action = observation.lastOpponentAction
    if action not in action_histogram:
        action_histogram[action] = 0
    action_histogram[action] += 1
    mode_action = None
    mode_action_count = None
    for k, v in action_histogram.items():
        if mode_action_count is None or v > mode_action_count:
            mode_action = k
            mode_action_count = v
            continue

    return (mode_action + 1) % configuration.signs


def nash_equilibrium(observation, configuration):
    return random.randint(0, 2)


def markov_agent(observation, configuration):
    k = 2
    global table, action_seq
    if observation.step % 250 == 0:  # refresh table every 250 steps
        action_seq, table = [], collections.defaultdict(lambda: [1, 1, 1])
    if len(action_seq) <= 2 * k + 1:
        action = int(np.random.randint(3))
        if observation.step > 0:
            action_seq.extend([observation.lastOpponentAction, action])
        else:
            action_seq.append(action)
        return action
    # update table
    key = ''.join([str(a) for a in action_seq[:-1]])
    table[key][observation.lastOpponentAction] += 1
    # update action seq
    action_seq[:-2] = action_seq[2:]
    action_seq[-2] = observation.lastOpponentAction
    # predict opponent next move
    key = ''.join([str(a) for a in action_seq[:-1]])
    if observation.step < 500:
        next_opponent_action_pred = np.argmax(table[key])
    else:
        scores = np.array(table[key])
        # add stochasticity for second part of the game
        next_opponent_action_pred = np.random.choice(3, p=scores/scores.sum())
    # make an action
    action = (next_opponent_action_pred + 1) % 3
    # if high probability to lose -> let's surprise our opponent with sudden change of our strategy
    if observation.step > 900:
        action = next_opponent_action_pred
    action_seq[-1] = action
    return int(action)


def find_pattern(memory_patterns, memory, memory_length):
    """ find appropriate pattern in memory """
    for pattern in memory_patterns:
        actions_matched = 0
        for i in range(memory_length):
            if pattern["actions"][i] == memory[i]:
                actions_matched += 1
            else:
                break
        # if memory fits this pattern
        if actions_matched == memory_length:
            return pattern
    # appropriate pattern not found
    return None


def get_step_result_for_my_agent(my_agent_action, opp_action):
    """ 
        get result of the step for my_agent
        1, 0 and -1 representing win, tie and lost results of the game respectively
    """
    if my_agent_action == opp_action:
        return 0
    elif (my_agent_action == (opp_action + 1)) or (my_agent_action == 0 and opp_action == 2):
        return 1
    else:
        return -1


# maximum steps in the pattern
steps_max = 3
# minimum steps in the pattern
steps_min = 3
# maximum amount of steps until reassessment of effectiveness of current memory patterns
max_steps_until_memory_reassessment = random.randint(80, 120)

# current memory of the agent
current_memory = []
# list of 1, 0 and -1 representing win, tie and lost results of the game respectively
# length is max_steps_until_memory_reassessment
results = []
# current best sum of results
best_sum_of_results = 0
# how many times each action was performed by opponent
opponent_actions_count = [0, 0, 0]
# memory length of patterns in first group
# steps_max is multiplied by 2 to consider both my_agent's and opponent's actions
group_memory_length = steps_max * 2
# list of groups of memory patterns
groups_of_memory_patterns = []
for i in range(steps_max, steps_min - 1, -1):
    groups_of_memory_patterns.append({
        # how many steps in a row are in the pattern
        "memory_length": group_memory_length,
        # list of memory patterns
        "memory_patterns": []
    })
    group_memory_length -= 2


def memory_patterns_agent(obs, conf):
    """ your ad here """
    global results
    global best_sum_of_results
    # action of my_agent
    my_action = None

    # if it's not first step, add opponent's last action to agent's current memory
    # and reassess effectiveness of current memory patterns
    if obs.step > 0:
        # count opponent's actions
        opponent_actions_count[obs.lastOpponentAction] += 1
        # add opponent's last step to current_memory
        current_memory.append(obs.lastOpponentAction)
        # previous step won or lost
        results.append(get_step_result_for_my_agent(
            current_memory[-2], current_memory[-1]))

        # if there is enough steps added to results for memery reassessment
        if len(results) == max_steps_until_memory_reassessment:
            results_sum = sum(results)
            # if effectiveness of current memory patterns has decreased significantly
            if results_sum < (best_sum_of_results * 0.5):
                # flush all current memory patterns
                best_sum_of_results = 0
                results = []
                for group in groups_of_memory_patterns:
                    group["memory_patterns"] = []
            else:
                # if effectiveness of current memory patterns has increased
                if results_sum > best_sum_of_results:
                    best_sum_of_results = results_sum
                del results[:1]

    # search for my_action in memory patterns
    for group in groups_of_memory_patterns:
        # if length of current memory is bigger than necessary for a new memory pattern
        if len(current_memory) > group["memory_length"]:
            # get momory of the previous step
            previous_step_memory = current_memory[:group["memory_length"]]
            previous_pattern = find_pattern(
                group["memory_patterns"], previous_step_memory, group["memory_length"])
            if previous_pattern == None:
                previous_pattern = {
                    "actions": previous_step_memory.copy(),
                    "opp_next_actions": [
                        {"action": 0, "amount": 0, "response": 1},
                        {"action": 1, "amount": 0, "response": 2},
                        {"action": 2, "amount": 0, "response": 0}
                    ]
                }
                group["memory_patterns"].append(previous_pattern)
            # if such pattern already exists
            for action in previous_pattern["opp_next_actions"]:
                if action["action"] == obs.lastOpponentAction:
                    action["amount"] += 1
            # delete first two elements in current memory (actions of the oldest step in current memory)
            del current_memory[:2]

            # if action was not yet found
            if my_action == None:
                pattern = find_pattern(
                    group["memory_patterns"], current_memory, group["memory_length"])
                # if appropriate pattern is found
                if pattern != None:
                    my_action_amount = 0
                    for action in pattern["opp_next_actions"]:
                        # if this opponent's action occurred more times than currently chosen action
                        # or, if it occured the same amount of times and this one is choosen randomly among them
                        if (action["amount"] > my_action_amount or
                                (action["amount"] == my_action_amount and random.random() > 0.5)):
                            my_action_amount = action["amount"]
                            my_action = action["response"]

    # if no action was found
    if my_action == None:
        # choose action randomly
        my_action = random.randint(0, 2)

    current_memory.append(my_action)
    return my_action


T = np.zeros((3, 3))
P = np.zeros((3, 3))

# a1 is the action of the opponent 1 step ago
# a2 is the action of the opponent 2 steps ago
a1, a2 = None, None


def transition_agent(observation, configuration):
    global T, P, a1, a2
    if observation.step > 1:
        a1 = observation.lastOpponentAction
        T[a2, a1] += 1
        P = np.divide(T, np.maximum(1, T.sum(axis=1)).reshape(-1, 1))
        a2 = a1
        if np.sum(P[a1, :]) == 1:
            return int((np.random.choice(
                [0, 1, 2],
                p=P[a1, :]
            ) + 1) % 3)
        else:
            return int(np.random.randint(3))
    else:
        if observation.step == 1:
            a2 = observation.lastOpponentAction
        return int(np.random.randint(3))


def construct_local_features(rollouts):
    features = np.array([[step % k for step in rollouts['steps']]
                         for k in (2, 3, 5)])
    features = np.append(features, rollouts['steps'])
    features = np.append(features, rollouts['actions'])
    features = np.append(features, rollouts['opp-actions'])
    return features


def construct_global_features(rollouts):
    features = []
    for key in ['actions', 'opp-actions']:
        for i in range(3):
            actions_count = np.mean([r == i for r in rollouts[key]])
            features.append(actions_count)

    return np.array(features)


def construct_features(short_stat_rollouts, long_stat_rollouts):
    lf = construct_local_features(short_stat_rollouts)
    gf = construct_global_features(long_stat_rollouts)
    features = np.concatenate([lf, gf])
    return features


def predict_opponent_move(train_data, test_sample):
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(train_data['x'], train_data['y'])
    return classifier.predict(test_sample)


def update_rollouts_hist(rollouts_hist, last_move, opp_last_action):
    rollouts_hist['steps'].append(last_move['step'])
    rollouts_hist['actions'].append(last_move['action'])
    rollouts_hist['opp-actions'].append(opp_last_action)
    return rollouts_hist


def warmup_strategy(observation, configuration):
    global rollouts_hist, last_move
    action = int(np.random.randint(3))
    if observation.step == 0:
        last_move = {'step': 0, 'action': action}
        rollouts_hist = {'steps': [], 'actions': [], 'opp-actions': []}
    else:
        rollouts_hist = update_rollouts_hist(
            rollouts_hist, last_move, observation.lastOpponentAction)
        last_move = {'step': observation.step, 'action': action}
    return int(action)


def init_training_data(rollouts_hist, k):
    for i in range(len(rollouts_hist['steps']) - k + 1):
        short_stat_rollouts = {
            key: rollouts_hist[key][i:i+k] for key in rollouts_hist}
        long_stat_rollouts = {
            key: rollouts_hist[key][:i+k] for key in rollouts_hist}
        features = construct_features(short_stat_rollouts, long_stat_rollouts)
        data['x'].append(features)
    test_sample = data['x'][-1].reshape(1, -1)
    data['x'] = data['x'][:-1]
    data['y'] = rollouts_hist['opp-actions'][k:]
    return data, test_sample


def decison_tree_agent(observation, configuration):
    # hyperparameters
    k = 5
    min_samples = 25
    global rollouts_hist, last_move, data, test_sample
    if observation.step == 0:
        data = {'x': [], 'y': []}
    # if not enough data -> randomize
    if observation.step <= min_samples + k:
        return warmup_strategy(observation, configuration)
    # update statistics
    rollouts_hist = update_rollouts_hist(
        rollouts_hist, last_move, observation.lastOpponentAction)
    # update training data
    if len(data['x']) == 0:
        data, test_sample = init_training_data(rollouts_hist, k)
    else:
        short_stat_rollouts = {
            key: rollouts_hist[key][-k:] for key in rollouts_hist}
        features = construct_features(short_stat_rollouts, rollouts_hist)
        data['x'].append(test_sample[0])
        data['y'] = rollouts_hist['opp-actions'][k:]
        test_sample = features.reshape(1, -1)

    # predict opponents move and choose an action
    next_opp_action_pred = predict_opponent_move(data, test_sample)
    action = int((next_opp_action_pred + 1) % 3)
    last_move = {'step': observation.step, 'action': action}
    return action


basis = np.array(
    [1, cmath.exp(2j * cmath.pi * 1 / 3), cmath.exp(2j * cmath.pi * 2 / 3)]
)


HistMatchResult = namedtuple("HistMatchResult", "idx length")


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
        # which seem natural and accounts for the correlation between target and feat
        # all RPSContest bots do no more than that, just in a hidden way

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

    def __call__(self, obs, conf):
        try:
            if obs.step == 0:
                action = np.random.choice(3)
                self.my_hist.append(action)
                return action

            self.step = obs.step

            opp = int(obs.lastOpponentAction)
            my = self.my_hist[-1]

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


agent = Agent()


def geometric_agent(obs, conf):
    return agent(obs, conf)
