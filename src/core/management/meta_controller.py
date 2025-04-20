

class MetaController:
    """
    Hybrid meta-controller for agent adaptation.
    Can operate as a standalone (global) controller or be embedded in agents.
    Supports fuzzy rule tuning, neural parameter adaptation, and group coordination.
    """

    def __init__(self, agents=None):
        self.agents = agents if agents is not None else []

    def register_agent(self, agent):
        self.agents.append(agent)

    def tune_fuzzy_rules(self, agent, data):
        """
        Tune all fuzzy sets in agent's fuzzy system using provided data.
        data: list/array of input samples (used for updating fuzzy set parameters)
        """
        if hasattr(agent, "model") and hasattr(agent.model, "fis"):
            fis = agent.model.fis
            for rule in fis.rules:
                for i, fs in rule.antecedents:
                    # Gather data for this input index
                    feature_data = [x[i] for x in data]
                    fs.tune(feature_data)

    def tune_learning_rate(self, agent, new_lr):
        """
        Set a new learning rate for the agent's neural network (and FIS if supported).
        """
        if hasattr(agent, "set_learning_rate"):
            agent.set_learning_rate(new_lr)

    def adapt_all_agents(self, data_map=None, lr_map=None):
        """
        Batch adaptation for all registered agents.
        data_map: dict(agent -> data), lr_map: dict(agent -> lr)
        """
        for agent in self.agents:
            if data_map and agent in data_map:
                self.tune_fuzzy_rules(agent, data_map[agent])
            if lr_map and agent in lr_map:
                self.tune_learning_rate(agent, lr_map[agent])

    def meta_step(self, adaptation_signals):
        """
        Perform a meta-adaptation step based on signals (e.g., performance, reward).
        adaptation_signals: dict(agent -> signal)
        """
        # Example: If error is high, decrease learning rate; if low, increase
        for agent, signal in adaptation_signals.items():
            if signal > 0.2:
                self.tune_learning_rate(agent, 0.001)
            else:
                self.tune_learning_rate(agent, 0.01)
