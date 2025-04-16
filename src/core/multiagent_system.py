"""
multiagent_system.py

Defines MultiAgentSystem for managing and coordinating multiple agents.
"""
from typing import List, Any, Optional

class MultiAgentSystem:
    def __init__(self, agents: List[Any]):
        self.agents = agents
        self.messages = [[] for _ in agents]  # Message inbox per agent
        self.message_history = [[] for _ in agents]  # Optional: store all received messages

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.messages = [[] for _ in self.agents]
        self.message_history = [[] for _ in self.agents]

    def step(self, observations, states=None):
        actions = []
        new_messages = [[] for _ in self.agents]
        for i, agent in enumerate(self.agents):
            action = agent.act(observations[i], state=states[i] if states else None)
            actions.append(action)
        self.messages = new_messages
        return actions

    def send_message(self, sender_idx: int, recipient_idx: int, message: Any, msg_type: str = "INFO"):
        msg = {"from": sender_idx, "type": msg_type, "content": message}
        self.messages[recipient_idx].append(msg)
        self.message_history[recipient_idx].append(msg)

    def broadcast_message(self, sender_idx: int, message: Any, msg_type: str = "BROADCAST"):
        msg = {"from": sender_idx, "type": msg_type, "content": message}
        for i in range(len(self.agents)):
            if i != sender_idx:
                self.messages[i].append(msg)
                self.message_history[i].append(msg)

    def get_messages(self, agent_idx: int, msg_type: Optional[str] = None):
        inbox = self.messages[agent_idx]
        if msg_type is not None:
            return [msg for msg in inbox if msg["type"] == msg_type]
        return inbox

    def get_message_history(self, agent_idx: int, msg_type: Optional[str] = None):
        history = self.message_history[agent_idx]
        if msg_type is not None:
            return [msg for msg in history if msg["type"] == msg_type]
        return history
