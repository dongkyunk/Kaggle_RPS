from abc import abstractmethod

class Model():
    def __init__(self):
        self.tactic = 0

    @abstractmethod
    def train(self, my_actions, op_actions, reward):
        pass

    def action(self):
        return self.tactic