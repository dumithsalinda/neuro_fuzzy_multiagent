"""
online_learning.py

Provides OnlineLearnerMixin for agents to learn from web resources.
"""
import requests

class OnlineLearnerMixin:
    def learn_from_url(self, url, parse_fn=None):
        """
        Fetch data from a URL and update agent knowledge.
        parse_fn: function to convert raw data to agent-compatible format (optional).
        """
        response = requests.get(url)
        response.raise_for_status()
        data = response.text
        if parse_fn:
            parsed = parse_fn(data)
        else:
            parsed = data
        self.integrate_online_knowledge(parsed)

    def integrate_online_knowledge(self, knowledge):
        """
        Integrate fetched knowledge into the agent.
        Override this in the agent class for custom behavior.
        """
        raise NotImplementedError("integrate_online_knowledge must be implemented by the agent.")
