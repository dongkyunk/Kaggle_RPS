from rps_agent import rps_agent
import sparring_partner.white_belt.mirror_shift_1 as mirror_shift
import sparring_partner.white_belt.counter_reactionary as counter_reactionary
import sparring_partner.blue_belt.transition_matrix as transition_matrix
import sparring_partner.black_belt.greenberg as greenberg
import sparring_partner.black_belt.iocane_powder as iocane_powder
import sparring_partner.black_belt.dllu1 as dllu1
import sparring_partner.black_belt.IOU2 as IOU2
import test_partner
import matplotlib.pyplot as plt
import ryan

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

class Config():
    def __init__(self, signs):
        self.signs = signs

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
            action1 = self.agent1(self.observation1, Config(3))
            action2 = self.agent2(self.observation2, Config(3))
            self.observation1.set_lastOpponentAction(action2)
            self.observation2.set_lastOpponentAction(action1)
            self.observation1.set_reward(action1, action2)
            self.observation2.set_reward(action2, action1)
            # print(action1, action2)
            action1, action2 = num_to_rps[action1], num_to_rps[action2]
            player1_reward_lst.append(self.observation1.reward)
            print(
                "step : {} / agent1 : {} / agent2 : {}".format(self.step, action1, action2))
            print("agent1 reward: {} / agent2 reward: {}".format(
                self.observation1.reward, self.observation2.reward))
        print("agent1 reward: {} / agent2 reward: {}".format(
            self.observation1.reward, self.observation2.reward))

        plt.plot(player1_reward_lst)
        plt.ylabel('reward')
        plt.savefig('data/mock_battle.png', dpi=300)


# #rps_agent
mb = MockBattler(rps_agent, IOU2.agent)
#counter_reactionary.counter_reactionary
#transition_matrix.transition_agent
mb.run()
# def get_result(match_settings):
#     start = datetime.now()
#     outcomes = kaggle_environments.evaluate(
#         'rps', [match_settings[0], match_settings[1]], num_episodes=match_settings[2])
#     won, lost, tie, avg_score = 0, 0, 0, 0.
#     for outcome in outcomes:
#         score = outcome[0]
#         if score > 0: won += 1
#         elif score < 0: lost += 1
#         else: tie += 1
#         avg_score += score
#     elapsed = datetime.now() - start
#     return match_settings[1], won, lost, tie, elapsed, float(avg_score) / float(match_settings[2])

# import os
# import pandas as pd
# import kaggle_environments
# from datetime import datetime
# import multiprocessing as pymp
# from tqdm import tqdm
# import ray.util.multiprocessing as raymp

# get_result(['sparring_partner/white_belt/all_paper.py', 'sparring_partner/white_belt/all_paper.py', 1000])
