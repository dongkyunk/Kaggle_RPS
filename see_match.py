import json
import matplotlib.pyplot as plt

with open('sample.json') as json_file:
    data = json.load(json_file)

player1 = [step[0]["reward"] for step in data["steps"]]
player2 = [step[1]["reward"] for step in data["steps"]]

plt.plot(player1)
plt.ylabel('reward')
plt.show()