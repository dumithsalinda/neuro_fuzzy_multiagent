# core/agent.py
# Base agent class
class Agent:
    def __init__(self, name, rules):
        self.name = name
        self.rules = rules
        self.state = {}
    def act(self, observation):
        # Decide and return action
        pass
    def learn(self, experience):
        # Update agent knowledge
        pass
