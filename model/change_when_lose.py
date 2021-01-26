from model.model import BaselineModel

class BaselineModel(Model):
    def __init__(self):
        self.tactic = 0

    def train(self, my_actions, op_actions, reward):
        if (int((3 + my_actions[-1] - op_actions[-1]) % 3) == 2):
            choice_lst = [0, 1, 2]
            choice_lst.remove(self.tactic)
            seed = random.randint(0, 100)
            random.seed(seed)
            self.tactic = random.choice(choice_lst)

    def action(self):
        return self.tactic