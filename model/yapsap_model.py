import random
from model.model import Model
from scipy import stats as s
from statistics import mode
from model.decision_tree_model import DecisionTreeModel

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
            # most_freq = int(s.mode(my_actions[-self.frequency:])[0])
            # pred_opp = (most_freq + 1) % 3
            # self.tactic = (pred_opp + 1) % 3
            
            # Train submodel
            self.model.train(my_actions, op_actions, reward)
            self.tactic = self.model.action()
            self.curr_freq = 0
            
            # print(my_actions[-self.frequency:])
            # print("most_freq: ", most_freq)
            # print("pred_opp: ", pred_opp)
            # print("tactic: ", self.tactic)
        else:
            self.tactic = random.randint(0, 2)

        # print("self.curr_freq:", self.curr_freq)

    def action(self):
        return self.tactic
