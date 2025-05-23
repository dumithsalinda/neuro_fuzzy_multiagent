import numpy as np


class MultiAgentSystem:
    """
    Manages a group of agents and facilitates collaboration and communication.

    Supports dynamic group formation, joining, leaving, and dissolution for self-organization.
    Provides APIs for group leader election, group knowledge sharing, and privacy-aware communication.

    Args:
        agents (list[Agent]): List of agent instances managed by the system.
    """

    def __init__(self, agents):
        self.agents = agents
        self.groups = {}  # group_id -> set of agent indices
        self.group_leaders = {}  # group_id -> agent index
        self.group_modules = {}  # group_id -> group-specific module (rules/subnetwork)

        """
        For each group, aggregate and distribute knowledge among group members.
        mode: 'average' (default), 'copy_leader', or custom.
        """
        for group_id, members in self.groups.items():
            if not members:
                continue
            knowledge_list = [self.agents[idx].share_knowledge() for idx in members]
            # Filter out None
            knowledge_list = [k for k in knowledge_list if k is not None]
            if not knowledge_list:
                continue
            # Simple average if possible, else copy leader's knowledge
            if mode == "average" and isinstance(knowledge_list[0], (np.ndarray, list)):
                avg_knowledge = (
                    np.mean(knowledge_list, axis=0)
                    if len(knowledge_list) > 0
                    else knowledge_list
                )
                for idx in members:
                    self.agents[idx].integrate_online_knowledge(avg_knowledge)
            elif mode == "copy_leader":
                leader_idx = self.group_leaders.get(group_id, list(members)[0])
                leader_knowledge = self.agents[leader_idx].share_knowledge()
                for idx in members:
                    self.agents[idx].integrate_online_knowledge(leader_knowledge)
            # Extend with more aggregation modes as needed

    def collective_action_selection(self, observations, mode="leader"):
        """
        For each group, select actions collectively for all members.
        mode: 'leader' (all follow leader), 'vote' (majority), or custom.
        observations: list of observation vectors, one per agent.
        Returns: list of actions (grouped or individual as per mode)
        """
        actions = [None] * len(self.agents)
        for group_id, members in self.groups.items():
            if not members:
                continue
            if mode == "leader":
                leader_idx = self.group_leaders.get(group_id, list(members)[0])
                leader_obs = observations[leader_idx]
                leader_action = self.agents[leader_idx].act(leader_obs)
                for idx in members:
                    actions[idx] = leader_action
            elif mode == "vote":
                group_obs = [observations[idx] for idx in members]
                group_actions = [
                    self.agents[idx].act(obs) for idx, obs in zip(members, group_obs)
                ]
                # Majority vote
                from collections import Counter

                most_common = Counter(group_actions).most_common(1)[0][0]
                for idx in members:
                    actions[idx] = most_common
            # Extend with more collective modes as needed
        return actions

    def elect_leaders(self):
        """
        Elect a leader for each group (default: agent with lowest index in group).
        Sets is_leader attribute on agents.
        """
        # Reset all
        for agent in self.agents:
            agent.is_leader = False
        for group_id, members in self.groups.items():
            if members:
                leader = min(members)
                self.group_leaders[group_id] = leader
                self.agents[leader].is_leader = True

    def auto_group_by_proximity(self, distance_threshold=1.5):
        """
        Cluster agents into groups based on Euclidean distance between their positions.
        Agents within distance_threshold are grouped together.
        Requires each agent to have a .position attribute (e.g., np.array([x, y])).
        """
        group_id = 0
        assigned = set()
        for i, agent in enumerate(self.agents):
            if i in assigned:
                continue
            # Start a new group
            group_members = [i]
            assigned.add(i)
            for j, other in enumerate(self.agents):
                if j != i and j not in assigned:
                    if hasattr(agent, "position") and hasattr(other, "position"):
                        dist = np.linalg.norm(
                            np.array(agent.position) - np.array(other.position)
                        )
                        if dist <= distance_threshold:
                            group_members.append(j)
                            assigned.add(j)
            self.form_group(f"auto_{group_id}", group_members)
            group_id += 1

    def auto_group_by_som(self, feature_matrix, som_shape=(5, 5), num_iteration=100):
        """
        Cluster agents into groups using a Self-Organizing Map (SOM) based on agent feature vectors.
        Args:
            feature_matrix: np.ndarray or list of shape (n_agents, n_features)
            som_shape: tuple, shape of the SOM grid (default (5,5))
            num_iteration: number of training iterations for SOM
        Each SOM node becomes a group; agents mapped to the same node are grouped together.
        """
        from neuro_fuzzy_multiagent.core.neural_networks.som_cluster import SOMClusterer

        feature_matrix = np.array(feature_matrix)
        som = SOMClusterer(
            input_dim=feature_matrix.shape[1],
            som_shape=som_shape,
            num_iteration=num_iteration,
        )
        som.fit(feature_matrix)
        clusters = som.predict(feature_matrix)
        # Map (x, y) SOM nodes to group IDs
        group_map = {}
        for idx, cluster in enumerate(clusters):
            group_key = f"som_{cluster[0]}_{cluster[1]}"
            if group_key not in group_map:
                group_map[group_key] = []
            group_map[group_key].append(idx)
        # Clear existing groups
        self.groups = {}
        for group_id, agent_indices in group_map.items():
            self.form_group(group_id, agent_indices)

    def form_group(self, group_id, agent_indices):
        """
        Create a new group with the specified agents. Also create a group-specific module (rules/subnetwork).
        Calls add_group_module for dynamic module creation.
        """
        self.groups[group_id] = set(agent_indices)
        for idx in agent_indices:
            self.agents[idx].group = group_id
        # --- Dynamic module/rule/subnetwork creation ---
        self.add_group_module(group_id)

    def add_group_module(self, group_id, module=None):
        """
        Dynamically create and register a group-specific module (rules, subnetworks, etc).
        If module is None, creates a default dummy module.
        """
        if module is None:
            module = {
                "rules": [f"rule_{group_id}_1", f"rule_{group_id}_2"],
                "subnetwork": f"subnet_{group_id}",
            }
        self.group_modules[group_id] = module

    def remove_group_module(self, group_id):
        """
        Remove a group-specific module at runtime.
        """
        if group_id in self.group_modules:
            del self.group_modules[group_id]

    def get_group_module(self, group_id):
        """
        Retrieve the module (rules, subnetworks, etc) for a group.
        Returns None if not present.
        """
        return self.group_modules.get(group_id, None)

    def join_group(self, agent_idx, group_id):
        """
        Add an agent to an existing group.
        """
        if group_id not in self.groups:
            self.groups[group_id] = set()
        self.groups[group_id].add(agent_idx)
        self.agents[agent_idx].group = group_id

    def leave_group(self, agent_idx):
        """
        Remove an agent from its current group.
        """
        group_id = self.agents[agent_idx].group
        if group_id and group_id in self.groups:
            self.groups[group_id].discard(agent_idx)
            if not self.groups[group_id]:
                del self.groups[group_id]
        self.agents[agent_idx].group = None

    def dissolve_group(self, group_id):
        """
        Remove all agents from the specified group and delete the group. Also remove the group-specific module.
        """
        if group_id in self.groups:
            for idx in self.groups[group_id]:
                self.agents[idx].group = None
            del self.groups[group_id]
        # --- Dynamic module/rule/subnetwork removal ---
        self.remove_group_module(group_id)

    """
    Manages a group of agents and facilitates collaboration/communication.
    Supports dynamic group formation, joining, leaving, and dissolution for self-organization.
    """

    def broadcast(self, message, sender=None, group=None):
        """
        Broadcast a message to all agents except sender. If group is specified, only to group members.
        """
        for agent in self.agents:
            if agent is not sender:
                if group is None or (hasattr(agent, "group") and agent.group == group):
                    agent.receive_message(message, sender=sender)

    def step_all(self, observations, states=None):
        """
        Step all agents with their respective observations (and optional states).
        Returns list of actions.
        """
        actions = []
        for i, agent in enumerate(self.agents):
            state = states[i] if states is not None else None
            actions.append(agent.act(observations[i], state=state))
        return actions

    def group_decision(
        self, observations, states=None, method="mean", weights=None, custom_fn=None
    ):
        """
        Make a group decision using the specified method:
        - 'mean': arithmetic mean (default)
        - 'weighted_mean': weighted average (requires weights)
        - 'majority_vote': mode (for discrete actions)
        - 'custom': user-supplied aggregation function (custom_fn)
        Enforces group laws on the result.
        Returns the group action.
        """
        actions = self.step_all(observations, states)
        import numpy as np

        from neuro_fuzzy_multiagent.core.agents.laws import enforce_laws

        result = None
        if method == "mean":
            result = np.mean(actions, axis=0)
        elif method == "weighted_mean":
            if weights is None:
                raise ValueError("Weights required for weighted_mean.")
            weights = np.array(weights)
            actions = np.array(actions)
            result = np.average(actions, axis=0, weights=weights)
        elif method == "majority_vote":
            # Assume actions are 1D discrete values
            from scipy.stats import mode

            mode_result = mode(np.array(actions), axis=0, keepdims=False).mode
            if isinstance(mode_result, np.ndarray):
                result = mode_result[0] if mode_result.size > 0 else None
            else:
                result = mode_result
        elif method == "custom":
            if custom_fn is None:
                raise ValueError("custom_fn must be provided for custom aggregation.")
            result = custom_fn(actions)
        else:
            raise ValueError(f"Unknown group decision method: {method}")
        enforce_laws(result, state={"actions": actions}, category="group")
        return result

    def coordinate_actions(self, observations, states=None):
        """
        Backward-compatible: Let agents agree on a consensus action (mean).
        Enforces group laws on the consensus action.
        Returns the consensus action if legal, else raises LawViolation.
        """
        return self.group_decision(observations, states, method="mean")

    def broadcast_knowledge(self, knowledge, sender=None):
        """
        Broadcast knowledge to all agents except sender, enforcing group and knowledge laws, respecting privacy.
        Supports different privacy levels:
        - 'public': broadcast to all agents
        - 'private': do not share
        - 'group-only': share within sender's group
        - 'recipient-list': share with specified recipients
        """
        from neuro_fuzzy_multiagent.core.agents.laws import LawViolation, enforce_laws

        privacy = (
            knowledge.get("privacy", "public")
            if isinstance(knowledge, dict)
            else "public"
        )
        try:
            enforce_laws(knowledge, state=None, category="group")
            enforce_laws(knowledge, state=None, category="knowledge")
            if privacy == "private":
                return  # Do not share
            elif privacy == "public":
                for agent in self.agents:
                    if agent is not sender:
                        agent.receive_message(
                            {"type": "knowledge", "content": knowledge}, sender=sender
                        )
            elif (
                privacy == "group-only"
                and sender is not None
                and hasattr(sender, "group")
            ):
                for agent in self.agents:
                    if (
                        agent is not sender
                        and hasattr(agent, "group")
                        and agent.group == sender.group
                    ):
                        agent.receive_message(
                            {
                                "type": "knowledge",
                                "content": knowledge,
                                "privacy": "group-only",
                                "group": sender.group,
                            },
                            sender=sender,
                        )
            elif privacy == "recipient-list" and "recipients" in knowledge:
                for agent in knowledge["recipients"]:
                    if agent is not sender:
                        agent.receive_message(
                            {
                                "type": "knowledge",
                                "content": knowledge,
                                "privacy": "recipient-list",
                                "recipients": knowledge["recipients"],
                            },
                            sender=sender,
                        )
        except LawViolation as e:
            print(f"[MultiAgentSystem] Knowledge broadcast blocked by law: {e}")

    def aggregate_knowledge(self, attr="online_knowledge"):
        """
        Aggregate knowledge from all agents (e.g., for consensus or federated update).
        Returns a list of all non-None knowledge attributes.
        """
        return [
            getattr(agent, attr, None)
            for agent in self.agents
            if getattr(agent, attr, None) is not None
        ]
