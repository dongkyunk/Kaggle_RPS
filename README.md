# Kaggle_RPS

This agent has scored Top 8% (Bronze Medal) in the RPS competition.

## Key Concept

The model uses an ensemble of both public/custom-designed agents and their evil twin (designed to beat them).

The agents used are :
- "IO": Iocane Powder (http://davidbau.com/downloads/rps/rps-iocaine.py)
- "GEO": Geometry Bot (https://www.kaggle.com/superant/rps-geometry-silver-rank-by-minimal-logic)
- "TM": Transition Matrix
- "DT": Decision Tree 
- "MP": Memory Patterns 

It adds random actions as a noise to delude enemy predictions, and the amount of random actions in between agent actions are determined by the current score and step. 

The agent to use is chosen based on a weighted score of their recent predictions.

## TODO

I started this competiton a few weeks before the deadline, and the agent has some room for improvement.

Possible things to add are :
- Better voting scheme among ensemble (currently it just uses the best scored model)
- Enhanced strategy 
- More models 
