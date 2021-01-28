class MemoryPatterns:
    def __init__(self):
        self.current_memory = []
        self.previous_action = {
            "action": None,
            "action_from_pattern": False,
            "pattern_group_index": None,
            "pattern_index": None
        }
        self.steps_to_random = random.randint(3, 5)
        self.current_memory_max_length = 10
        self.reward = 0
        self.group_memory_length = self.current_memory_max_length
        self.groups_of_memory_patterns = []
        for i in range(5, 2, -1):
            self.groups_of_memory_patterns.append({
                "memory_length": self.group_memory_length,
                "memory_patterns": []
            })
            self.group_memory_length -= 2

    def evaluate_pattern_efficiency(self, previous_step_result):
        pattern_group_index = self.previous_action["pattern_group_index"]
        pattern_index = self.previous_action["pattern_index"]
        pattern = self.groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
        pattern["reward"] += previous_step_result
        if pattern["reward"] <= -3:
            del self.groups_of_memory_patterns[pattern_group_index]["memory_patterns"][pattern_index]
    
    def find_action(self, group, group_index):
        if len(self.current_memory) > group["memory_length"]:
            this_step_memory = self.current_memory[-group["memory_length"]:]
            memory_pattern, pattern_index = self.find_pattern(group["memory_patterns"], this_step_memory, group["memory_length"])
            if memory_pattern != None:
                my_action_amount = 0
                for action in memory_pattern["opp_next_actions"]:
                    if (action["amount"] > my_action_amount or
                            (action["amount"] == my_action_amount and random.random() > 0.5)):
                        my_action_amount = action["amount"]
                        my_action = action["response"]
                return my_action, pattern_index
        return None, None

    def find_pattern(self, memory_patterns, memory, memory_length):
        for i in range(len(memory_patterns)):
            actions_matched = 0
            for j in range(memory_length):
                if memory_patterns[i]["actions"][j] == memory[j]:
                    actions_matched += 1
                else:
                    break
            if actions_matched == memory_length:
                return memory_patterns[i], i
        return None, None

    def update_current_memory(self, my_action):
        if len(self.current_memory) > self.current_memory_max_length:
            del self.current_memory[:2]
        self.current_memory.append(my_action)
    
    def update_memory_pattern(self, group, A):
        if len(self.current_memory) > group["memory_length"]:
            previous_step_memory = self.current_memory[-group["memory_length"] - 2 : -2]
            previous_pattern, pattern_index = self.find_pattern(group["memory_patterns"], previous_step_memory, group["memory_length"])
            if previous_pattern == None:
                previous_pattern = {
                    "actions": previous_step_memory.copy(),
                    "reward": 0,
                    "opp_next_actions": [
                        {"action": 0, "amount": 0, "response": 1},
                        {"action": 1, "amount": 0, "response": 2},
                        {"action": 2, "amount": 0, "response": 0}
                    ]
                }
                group["memory_patterns"].append(previous_pattern)
            for action in previous_pattern["opp_next_actions"]:
                if action["action"] == A:
                    action["amount"] += 1
    
    def train(self, my_actions, op_actions, reward, step, configuration):
        self.tactic = None
        T = step
        A = op_actions[-1]
        S = configuration.signs

        self.steps_to_random -= 1
        if self.steps_to_random <= 0:
            self.steps_to_random = random.randint(3, 5)
            self.tactic = secrets.randbelow(S)
            self.previous_action["action"] = self.tactic
            self.previous_action["action_from_pattern"] = False
            self.previous_action["pattern_group_index"] = None
            self.previous_action["pattern_index"] = None

        if T > 0:
            self.current_memory.append(A)
            previous_step_result = get_score(S, self.current_memory[-2], self.current_memory[-1])
            self.reward += previous_step_result
            if self.previous_action["action_from_pattern"]:
                self.evaluate_pattern_efficiency(previous_step_result)

        for i in range(len(self.groups_of_memory_patterns)):
            self.update_memory_pattern(self.groups_of_memory_patterns[i], A)
            if self.tactic == None:
                self.tactic, pattern_index = self.find_action(self.groups_of_memory_patterns[i], i)
                if self.tactic != None:
                    self.previous_action["action"] = self.tactic
                    self.previous_action["action_from_pattern"] = True
                    self.previous_action["pattern_group_index"] = i
                    self.previous_action["pattern_index"] = pattern_index

        if self.tactic == None:
            self.tactic = secrets.randbelow(S)
            self.previous_action["action"] = self.tactic
            self.previous_action["action_from_pattern"] = False
            self.previous_action["pattern_group_index"] = None
            self.previous_action["pattern_index"] = None

        self.update_current_memory(self.tactic)

    def action(self):
        return self.tactic