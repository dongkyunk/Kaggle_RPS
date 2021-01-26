import random
from model.model import Model
from scipy import stats as s

class DumbModel(Model):
    def __init__(self, frequency=15):
        self.curr_freq = 0
        self.tactic = 0
        self.frequency = frequency

    def train(self, my_actions, op_actions, reward):            
        self.curr_freq = self.curr_freq + 1
        if self.curr_freq > self.frequency:
            self.tactic=(int(s.mode(my_actions[-self.frequency:])[0]) + 1) % 3
            # print("Dumb last few::", op_actions[-self.frequency:])
            # print("Dumb most_freq::", int(s.mode(my_actions[-self.frequency:])[0]))
        else:
            self.tactic=random.randint(0, 2)

    def action(self):
        return self.tactic 