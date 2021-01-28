import os
import pandas as pd
import kaggle_environments
from datetime import datetime
import multiprocessing as pymp
from tqdm import tqdm
import ray.util.multiprocessing as raymp


# function to return score
def get_result(match_settings):
    start = datetime.now()
    outcomes = kaggle_environments.evaluate(
        'rps', [match_settings[0], match_settings[1]], num_episodes=match_settings[2])
    won, lost, tie, avg_score = 0, 0, 0, 0.
    for outcome in outcomes:
        score = outcome[0]
        if score > 0: won += 1
        elif score < 0: lost += 1
        else: tie += 1
        avg_score += score
    elapsed = datetime.now() - start
    return match_settings[1], won, lost, tie, elapsed, float(avg_score) / float(match_settings[2])


def eval_agent_against_baselines(agent, baselines, num_episodes=10, use_ray=False):
    df = pd.DataFrame(
        columns=['wins', 'loses', 'ties', 'total time', 'avg. score'],
        index=baselines
    )
    
    if use_ray:
        pool = raymp.Pool()
    else:
        pool = pymp.Pool()
    matches = [[agent, baseline, num_episodes] for baseline in baselines]
    
    results = []
    for content in tqdm(pool.imap_unordered(get_result, matches), total=len(matches)):
        results.append(content)
    

    for baseline_agent, won, lost, tie, elapsed, avg_score in results:
        df.loc[baseline_agent, 'wins'] = won
        df.loc[baseline_agent, 'loses'] = lost
        df.loc[baseline_agent, 'ties'] = tie
        df.loc[baseline_agent, 'total time'] = elapsed
        df.loc[baseline_agent, 'avg. score'] = avg_score
        
    return df

white_belt_agents = [os.path.join('white_belt', agent) for agent in os.listdir('white_belt')]
blue_belt_agents = [os.path.join('blue_belt', agent) for agent in os.listdir('blue_belt')]
black_belt_agents = [os.path.join('black_belt', agent) for agent in os.listdir('black_belt')]

my_agent = '/home/hobbs/bench-dk/.DataPreprocess/files/rps_kaggle_agent/rps_agent.py'
# my_agent = 'blue_belt/transition_matrix.py'
df = eval_agent_against_baselines(my_agent, blue_belt_agents + black_belt_agents)
print(df)
df.to_csv('eval_agent_against_baselines.csv')