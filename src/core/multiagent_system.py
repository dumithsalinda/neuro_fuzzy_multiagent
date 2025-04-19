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
        self.groups = {}  # group_id -> set of agent indices
        self.group_leaders = {}  # group_id -> agent index
        self.group_roles = {}  # group_id -> {agent_idx: role}

    def add_agent(self, agent):
        self.agents.append(agent)
        self.messages.append([])
        self.message_history.append([])

    def remove_agent(self, agent_or_idx):
        if isinstance(agent_or_idx, int):
            idx = agent_or_idx
        else:
            idx = self.agents.index(agent_or_idx)
        self.agents.pop(idx)
        self.messages.pop(idx)
        self.message_history.pop(idx)
        # Remove from any groups and update group membership
        affected_groups = []
        for gid, group in self.groups.items():
            if idx in group:
                group.discard(idx)
                affected_groups.append(gid)
        # Update group indices after removal
        for group in self.groups.values():
            updated = set()
            for i in group:
                updated.add(i if i < idx else i-1)
            group.clear()
            group.update(updated)
        # Update group_roles indices
        for gid, roles in self.group_roles.items():
            updated_roles = {}
            for i, role in roles.items():
                if i == idx:
                    continue
                updated_roles[i if i < idx else i-1] = role
            self.group_roles[gid] = updated_roles
        # Remove as leader if necessary, and re-elect if possible
        for gid in affected_groups:
            if gid in self.groups and self.groups[gid]:
                self.elect_leader(gid)
            else:
                self.group_leaders[gid] = None

    def create_group(self, group_id, agent_indices):
        self.groups[group_id] = set(agent_indices)
        self.group_leaders[group_id] = agent_indices[0] if agent_indices else None
        self.group_roles[group_id] = {idx: None for idx in agent_indices}

    def dissolve_group(self, group_id):
        if group_id in self.groups:
            del self.groups[group_id]
        if group_id in self.group_leaders:
            del self.group_leaders[group_id]
        if group_id in self.group_roles:
            del self.group_roles[group_id]

    def move_agent_to_group(self, agent_idx, group_id):
        # Remove from any group and roles
        for gid, group in self.groups.items():
            group.discard(agent_idx)
            if gid in self.group_roles and agent_idx in self.group_roles[gid]:
                del self.group_roles[gid][agent_idx]
        # Add to new group and roles
        if group_id in self.groups:
            self.groups[group_id].add(agent_idx)
            if group_id in self.group_roles:
                self.group_roles[group_id][agent_idx] = None
            else:
                self.group_roles[group_id] = {agent_idx: None}
        else:
            self.groups[group_id] = {agent_idx}
            self.group_roles[group_id] = {agent_idx: None}
        # Optionally, update leader
        if group_id not in self.group_leaders:
            self.group_leaders[group_id] = agent_idx

    def elect_leader(self, group_id, election_fn=None):
        """
        Elect a leader for the group using a custom function or default (lowest agent index).
        election_fn should take a set of agent indices and return the leader index.
        """
        if group_id not in self.groups or not self.groups[group_id]:
            self.group_leaders[group_id] = None
            return None
        if election_fn is None:
            leader = min(self.groups[group_id])
        else:
            leader = election_fn(self.groups[group_id])
        self.group_leaders[group_id] = leader
        if group_id in self.group_roles:
            self.group_roles[group_id][leader] = "leader"
        else:
            self.group_roles[group_id] = {leader: "leader"}
        return leader

    def assign_role(self, group_id, agent_idx, role):
        if group_id not in self.group_roles:
            self.group_roles[group_id] = {}
        self.group_roles[group_id][agent_idx] = role

    def get_role(self, group_id, agent_idx):
        return self.group_roles.get(group_id, {}).get(agent_idx, None)

    def get_leader(self, group_id):
        return self.group_leaders.get(group_id, None)

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
