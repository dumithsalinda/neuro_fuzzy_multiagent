"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""

import numpy as np
from .neuro_fuzzy import NeuroFuzzyHybrid
from .laws import enforce_laws
from .online_learning import OnlineLearnerMixin

from .universal_fuzzy_layer import UniversalFuzzyLayer

class Agent(OnlineLearnerMixin):
    """
    Generic agent that interacts with an environment using a model and policy.
    Now supports online learning from web resources.
    Supports dynamic group membership for self-organization.
    Supports plug-and-play fuzzy logic via UniversalFuzzyLayer.
    """
    def __init__(self, model, policy=None, bus=None, group=None):
        self.model = model
        self.policy = policy if policy is not None else self.random_policy
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0
        self.message_inbox = []
        self.last_message = None
        self.group = group  # Group identifier, None if not in a group
        self.bus = bus
        self._fuzzy_layer = None
        if self.bus is not None:
            self.bus.register(self, group=group)

    # --- Universal Fuzzy Layer Plug-and-Play ---
    def attach_fuzzy_layer(self, fuzzy_layer):
        """Attach a UniversalFuzzyLayer to this agent."""
        assert isinstance(fuzzy_layer, UniversalFuzzyLayer)
        self._fuzzy_layer = fuzzy_layer

    def detach_fuzzy_layer(self):
        """Detach the fuzzy layer from this agent."""
        self._fuzzy_layer = None

    def has_fuzzy_layer(self):
        """Return True if a fuzzy layer is attached."""
        return self._fuzzy_layer is not None

    def fuzzy_evaluate(self, x):
        """Evaluate the fuzzy layer on input x (if attached)."""
        if self._fuzzy_layer:
            return self._fuzzy_layer.evaluate(x)
        raise AttributeError("No fuzzy layer attached.")

    def fuzzy_explain(self, x):
        """Explain the fuzzy inference for input x (if attached)."""
        if self._fuzzy_layer:
            return self._fuzzy_layer.explain(x)
        raise AttributeError("No fuzzy layer attached.")

    def fuzzy_add_rule(self, antecedents, consequent, as_core=False):
        """Add a fuzzy rule to the agent's fuzzy layer (if attached)."""
        if self._fuzzy_layer:
            return self._fuzzy_layer.add_rule(antecedents, consequent, as_core=as_core)
        raise AttributeError("No fuzzy layer attached.")

    def fuzzy_prune_rule(self, rule_idx, from_core=False):
        """Remove a fuzzy rule by index from the agent's fuzzy layer (if attached)."""
        if self._fuzzy_layer:
            return self._fuzzy_layer.prune_rule(rule_idx, from_core=from_core)
        raise AttributeError("No fuzzy layer attached.")

    def fuzzy_list_rules(self):
        """Return a summary of all fuzzy rules in the agent's fuzzy layer (if attached)."""
        if self._fuzzy_layer:
            return self._fuzzy_layer.list_rules()
        raise AttributeError("No fuzzy layer attached.")

    def share_knowledge(self):
        """
        Return a generic representation of the agent's knowledge (Q-table, weights, rules, etc.).
        Override in subclasses for specific agent types.
        """
        if hasattr(self.model, 'get_knowledge'):
            return self.model.get_knowledge()
        return None

    def share_knowledge(self):
        """
        Return a generic representation of the agent's knowledge (Q-table, weights, rules, etc.).
        Override in subclasses for specific agent types.
        """
        if hasattr(self.model, 'get_knowledge'):
            return self.model.get_knowledge()
        return None

        self.model = model
        self.policy = policy if policy is not None else self.random_policy
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0
        self.message_inbox = []
        self.last_message = None
        self.group = group  # Group identifier, None if not in a group
        self.bus = bus
        if self.bus is not None:
            self.bus.register(self, group=group)

    def act(self, observation, state=None):
        """
        Select an action based on the current observation, enforcing unbreakable laws.
        """
        action = self.policy(observation, self.model)
        enforce_laws(action, state if state is not None else {})
        self.last_action = action
        self.last_observation = observation
        return action

    def observe(self, reward, *args, **kwargs):
        """
        Receive reward and update internal state. RL subclasses may override.
        """
        self.total_reward += reward

    def learn(self, *args, **kwargs):
        """
        Placeholder for agent learning (to be overridden for RL, etc.).
        """
        pass

    def self_organize(self, *args, **kwargs):
        """
        Placeholder for agent self-organization (to be overridden).
        """
        pass

    def reset(self):
        """
        Reset agent state for new episodes or reuse.
        """
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0

    def register_to_bus(self, bus, group=None):
        self.bus = bus
        self.group = group
        if self.bus is not None:
            self.bus.register(self, group=group)

    def unregister_from_bus(self):
        if self.bus is not None:
            self.bus.unregister(self)
        self.bus = None

    def send_message(self, message, recipient=None, group=None, broadcast=False):
        """
        Send a message to a recipient, group, or all (broadcast).
        If a MessageBus is attached, use it; else, direct send.
        """
        if self.bus is not None:
            if broadcast:
                self.bus.broadcast(message, sender=self)
            elif group is not None:
                self.bus.groupcast(message, group, sender=self)
            elif recipient is not None:
                self.bus.send(message, recipient)
        elif recipient is not None:
            recipient.receive_message(message, sender=self)

    def receive_message(self, message, sender=None):
        self.message_inbox.append((message, sender))
        self.last_message = message

    def share_knowledge(self, knowledge, system=None, group=None, recipients=None):
        """
        Share knowledge with other agents, respecting privacy policies.
        knowledge: dict with optional 'privacy' key (public, private, group-only, recipient-list)
        system: MultiAgentSystem instance
        group: optional group label (for group-only privacy)
        recipients: optional list of agent instances (for recipient-list privacy)
        Enforces knowledge laws before sharing.
        """
        from .laws import enforce_laws, LawViolation
        privacy = knowledge.get('privacy', 'public') if isinstance(knowledge, dict) else 'public'
        try:
            enforce_laws(knowledge, state=None, category='knowledge')
            if system is None:
                return
            # Determine recipients
            if privacy == 'private':
                return  # Do not share
            elif privacy == 'public':
                system.broadcast({'type': 'knowledge', 'content': knowledge}, sender=self)
            elif privacy == 'group-only' and group is not None:
                system.broadcast({'type': 'knowledge', 'content': knowledge, 'privacy': 'group-only', 'group': group}, sender=self, group=group)
            elif privacy == 'recipient-list' and recipients is not None:
                for agent in recipients:
                    if agent is not self:
                        agent.receive_message({'type': 'knowledge', 'content': knowledge, 'privacy': 'recipient-list', 'recipients': recipients}, sender=self)
        except LawViolation as e:
            print(f"[Agent] Knowledge sharing blocked by law: {e}")

    def receive_message(self, message, sender=None):
        """
        Receive a message from another agent (for collaboration protocols).
        Handles knowledge sharing if message type is 'knowledge'.
        Enforces knowledge laws before integrating shared knowledge.
        """
        if isinstance(message, dict) and message.get('type') == 'knowledge':
            from .laws import enforce_laws, LawViolation
            try:
                enforce_laws(message['content'], state=None, category='knowledge')
                self.integrate_online_knowledge(message['content'])
            except LawViolation as e:
                print(f"[Agent] Received knowledge blocked by law: {e}")
        else:
            self.last_message = (message, sender)

    @staticmethod
    def random_policy(observation, model):
        """
        Example random policy for demonstration/testing.
        """
        import numpy as np
        return np.random.uniform(-1, 1, size=np.shape(observation))

    def integrate_online_knowledge(self, knowledge):
        """
        Default: try to update policy/model if possible, else store as attribute.
        Override for custom knowledge/model/rule updates.
        """
        if hasattr(self.model, 'update_from_knowledge'):
            self.model.update_from_knowledge(knowledge)
        else:
            self.online_knowledge = knowledge

class NeuroFuzzyAgent(Agent):
    """
    Agent that uses a NeuroFuzzyHybrid model to select actions.
    Supports modular self-organization of fuzzy rules, membership functions, and neural network weights.
    """
    def __init__(self, nn_config, fis_config, policy=None):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)

    def self_organize(self, mutation_rate=0.01, tune_fuzzy=True, rule_change=True):
        """
        Trigger self-organization in the underlying neuro-fuzzy hybrid model.
        This adapts neural weights, tunes fuzzy sets, and adds/removes rules.
        """
        if hasattr(self.model, 'self_organize'):
            self.model.self_organize(mutation_rate=mutation_rate, tune_fuzzy=tune_fuzzy, rule_change=rule_change)

    # --- Rule Evolution and Meta-Learning Integration ---
    def add_rule(self, antecedents, consequent, as_core=False):
        """Add a fuzzy rule to this agent's model at runtime."""
        self.model.add_rule(antecedents, consequent, as_core=as_core)

    def prune_rule(self, rule_idx, from_core=False):
        """Remove a fuzzy rule by index from this agent's model."""
        self.model.prune_rule(rule_idx, from_core=from_core)

    def list_rules(self):
        """Return a summary of all fuzzy rules in this agent's model."""
        return self.model.list_rules()

    def set_learning_rate(self, lr):
        """Set learning rate for this agent's model."""
        self.model.set_learning_rate(lr)

    def get_learning_rate(self):
        """Get learning rate for this agent's model."""
        return self.model.get_learning_rate()
