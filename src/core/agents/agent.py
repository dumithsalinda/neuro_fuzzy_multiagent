"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""


from .laws import enforce_laws
from src.core.neuro_fuzzy import NeuroFuzzyHybrid
from src.core.management.online_learning import OnlineLearnerMixin
from src.core.neural_networks.universal_fuzzy_layer import UniversalFuzzyLayer
from typing import Callable, Optional, Dict, Any


class Agent(OnlineLearnerMixin):
    """
    Generic agent that interacts with an environment using a model and policy.

    Supports online learning, dynamic group membership, plug-and-play fuzzy logic (UniversalFuzzyLayer),
    and a standardized communication API (send_message, receive_message).

    Args:
        model: The agent's underlying model (e.g., DQN, NeuroFuzzyHybrid).
        policy: Callable for action selection (optional).
        bus: Optional message bus for inter-agent communication.
        group: Optional group identifier.
    """

    def __init__(self, model: Any, policy: Optional[Callable] = None, bus: Optional[Any] = None, group: Optional[str] = None):
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
        self.knowledge_received = []  # Stores received knowledge
        if self.bus is not None:
            self.bus.register(self, group=group)

    # --- Standardized Agent Communication API ---
    def send_message(self, message, recipient=None, group=None, broadcast=False):
        """
        Send a message to a recipient, group, or broadcast to all via the message bus if present.
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
        """
        Receive a message and append it to the agent's inbox. If the message is knowledge, process it.
        """
        self.message_inbox.append((message, sender))
        self.last_message = (message, sender)
        if isinstance(message, dict) and message.get("type") == "knowledge":
            knowledge = message.get("content")
            self.knowledge_received.append(knowledge)
            self.receive_knowledge(knowledge, sender=sender)

    def receive_knowledge(self, knowledge, sender=None):
        """
        Default handler for received knowledge. Appends (knowledge, sender) to self.knowledge_received.
        Override in subclasses for custom processing.
        """
        self.knowledge_received.append(knowledge)

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

    def get_knowledge(self):
        """
        Return a generic representation of the agent's knowledge (Q-table, weights, rules, etc.).
        Override in subclasses for specific agent types.
        """
        if hasattr(self.model, "get_knowledge"):
            return self.model.get_knowledge()
        return None

    def share_knowledge(self, knowledge, system=None, group=None, recipients=None):
        """
        Share knowledge with other agents, respecting privacy policies.
        knowledge: dict with optional 'privacy' key (public, private, group-only, recipient-list)
        system: MultiAgentSystem instance
        group: optional group label (for group-only privacy)
        recipients: optional list of agent instances (for recipient-list privacy)
        Enforces knowledge laws before sharing.
        """
        from .laws import LawViolation, enforce_laws

        privacy = (
            knowledge.get("privacy", "public")
            if isinstance(knowledge, dict)
            else "public"
        )
        try:
            enforce_laws(knowledge, state=None, category="knowledge")
            if system is None:
                return
            # Determine recipients
            if privacy == "private":
                return  # Do not share
            elif privacy == "public":
                system.broadcast(
                    {"type": "knowledge", "content": knowledge}, sender=self
                )
            elif privacy == "group-only" and group is not None:
                system.broadcast(
                    {
                        "type": "knowledge",
                        "content": knowledge,
                        "privacy": "group-only",
                        "group": group,
                    },
                    sender=self,
                    group=group,
                )
            elif privacy == "recipient-list" and recipients is not None:
                for agent in recipients:
                    if agent is not self:
                        agent.receive_message(
                            {
                                "type": "knowledge",
                                "content": knowledge,
                                "privacy": "recipient-list",
                                "recipients": recipients,
                            },
                            sender=self,
                        )
        except LawViolation as e:
            print(f"[Agent] Knowledge sharing blocked by law: {e}")

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

    def self_organize(self, *args, **kwargs):
        """
        Placeholder for agent self-organization (to be overridden).
        """

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
        if hasattr(self.model, "update_from_knowledge"):
            self.model.update_from_knowledge(knowledge)
        else:
            self.online_knowledge = knowledge


class NeuroFuzzyAgent(Agent):
    """
    Agent that uses a NeuroFuzzyHybrid model to select actions.
    Supports modular self-organization of fuzzy rules, membership functions, and neural network weights.
    Supports runtime mode switching between neural, fuzzy, and hybrid inference.
    """

    def __init__(
        self,
        nn_config,
        fis_config,
        policy=None,
        meta_controller=None,
        universal_fuzzy_layer=None,
    ):
        model = NeuroFuzzyHybrid(nn_config, fis_config)
        if policy is None:
            policy = lambda obs, model: model.forward(obs)
        super().__init__(model, policy)
        # Embedded meta-controller (agent-local)
        if meta_controller is None:
            from src.core.management.meta_controller import MetaController

            self.meta_controller = MetaController()

    def reload_config(self, config_or_path):
        """
        Reload agent configuration from a dict or YAML/JSON file at runtime.
        Updates nn_config, fis_config, meta_controller, universal_fuzzy_layer.
        """
        import json

        import yaml

        if isinstance(config_or_path, str):
            if config_or_path.endswith(".yaml") or config_or_path.endswith(".yml"):
                with open(config_or_path, "r") as f:
                    config = yaml.safe_load(f)
            elif config_or_path.endswith(".json"):
                with open(config_or_path, "r") as f:
                    config = json.load(f)
            else:
                raise ValueError("Config file must be .yaml, .yml, or .json")
        else:
            config = config_or_path
        # Update configs
        nn_config = config.get("nn_config", None)
        fis_config = config.get("fis_config", None)
        meta_cfg = config.get("meta_controller", None)
        fuzzy_cfg = config.get("universal_fuzzy_layer", None)
        if nn_config:
            self.model.update_nn_config(nn_config)
        if fis_config:
            self.model.update_fis_config(fis_config)
        if meta_cfg is not None:
            from src.core.management.meta_controller import MetaController

            self.meta_controller = MetaController(**meta_cfg) if meta_cfg else None
        if fuzzy_cfg is not None:
            from src.core.neural_networks.universal_fuzzy_layer import UniversalFuzzyLayer

            self._fuzzy_layer = UniversalFuzzyLayer(**fuzzy_cfg) if fuzzy_cfg else None

    def meta_adapt(self, data=None, new_lr=None):
        """
        Perform local meta-adaptation: tune fuzzy rules and/or learning rate.
        """
        if data is not None:
            self.meta_controller.tune_fuzzy_rules(self, data)
        if new_lr is not None:
            self.meta_controller.tune_learning_rate(self, new_lr)

    def set_mode(self, mode, hybrid_weight=None):
        """Set the inference mode for this agent's neuro-fuzzy model."""
        self.model.set_mode(mode, hybrid_weight=hybrid_weight)

    def get_mode(self):
        """Get the current inference mode for this agent's neuro-fuzzy model."""
        return self.model.get_mode()

    def evolve_rules(self, recent_inputs=None, min_avg_activation=0.01):
        """Prune dynamic fuzzy rules with low average firing strength."""
        return self.model.evolve_rules(
            recent_inputs=recent_inputs, min_avg_activation=min_avg_activation
        )

    def auto_switch_mode(self, error_history, thresholds=None):
        """Adaptively switch mode based on recent error history."""
        return self.model.auto_switch_mode(error_history, thresholds=thresholds)

    def self_organize(self, mutation_rate=0.01, tune_fuzzy=True, rule_change=True):
        """
        Trigger self-organization in the underlying neuro-fuzzy hybrid model.
        This adapts neural weights, tunes fuzzy sets, and adds/removes rules.
        """
        if hasattr(self.model, "self_organize"):
            self.model.self_organize(
                mutation_rate=mutation_rate,
                tune_fuzzy=tune_fuzzy,
                rule_change=rule_change,
            )

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
        """Get learning rate for this agent's model (neural net)."""
        if hasattr(self.model.nn, "learning_rate"):
            return self.model.nn.learning_rate
        return None
