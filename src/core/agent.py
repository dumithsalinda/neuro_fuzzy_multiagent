"""
agent.py

Defines a generic Agent class for single and multiagent scenarios.
Supports integration with neuro-fuzzy models, transfer learning, and various environments.
"""

import numpy as np
from .neuro_fuzzy import NeuroFuzzyHybrid
from .laws import enforce_laws
from .online_learning import OnlineLearnerMixin

class Agent(OnlineLearnerMixin):
    """
    Generic agent that interacts with an environment using a model and policy.
    Now supports online learning from web resources.
    """
    def __init__(self, model, policy=None):
        self.model = model
        self.policy = policy if policy is not None else self.random_policy
        self.last_action = None
        self.last_observation = None
        self.total_reward = 0

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
