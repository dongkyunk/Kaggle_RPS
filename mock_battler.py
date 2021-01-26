from sparring_rps_agent import sparring_rps_agent
from rps_agent import rps_agent
import sparring_partner
import matplotlib.pyplot as plt 

class Observation():
    def __init__(self):
        self.step = 0
        self.lastOpponentAction = 0
        self.reward = 0

    def set_step(self, step):
        self.step = step

    def set_lastOpponentAction(self, lastOpponentAction):
        self.lastOpponentAction = lastOpponentAction

    def set_reward(self, action1, action2):
        winner = int((3 + action1 - action2) % 3)
        if winner == 1:
            self.reward = self.reward + 1
        elif winner == 2:
            # Opponent won last game
            self.reward = self.reward - 1


class MockBattler():
    def __init__(self, agent1, agent2):
        self.agent1, self.agent2 = agent1, agent2
        self.observation1, self.observation2 = Observation(), Observation()
        self.step = 0
        self.reward = 0

    def run(self):
        num_to_rps = {
            0: "Rock",
            1: "Paper",
            2: "Scissor"
        }
        action1, action2 = 0, 0
        player1_reward_lst = list()
        for i in range(0, 1000):
            self.step = i
            self.observation1.set_step(self.step)
            self.observation2.set_step(self.step)
            action1 = self.agent1(self.observation1, None)
            action2 = self.agent2(self.observation2, None)
            self.observation1.set_lastOpponentAction(action2)
            self.observation2.set_lastOpponentAction(action1)
            self.observation1.set_reward(action1, action2)
            self.observation2.set_reward(action2, action1)
            action1, action2 = num_to_rps[action1], num_to_rps[action2]
            player1_reward_lst.append(self.observation1.reward)
            # print("agent1 : {} / agent2 : {}".format(action1, action2))
            print("agent1 reward: {} / agent2 reward: {}".format(
                self.observation1.reward, self.observation2.reward))
        print("agent1 reward: {} / agent2 reward: {}".format(
            self.observation1.reward, self.observation2.reward))

        plt.plot(player1_reward_lst)
        plt.ylabel('reward')
        plt.show()


mb = MockBattler(rps_agent, sparring_partner.transition_agent)
mb.run()
