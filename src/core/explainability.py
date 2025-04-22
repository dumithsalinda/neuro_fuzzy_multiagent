from abc import ABC, abstractmethod
from typing import Any, Dict

class Explainable(ABC):
    @abstractmethod
    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        """Return a human-readable explanation of the last action/decision."""
        pass

def explain_agent_action(agent, observation, action):
    if hasattr(agent, 'explain'):
        return agent.explain(observation=observation, action=action)
    return {"explanation": "No explain() method implemented for this agent."}

def explain_env_transition(env, state, action, next_state):
    if hasattr(env, 'explain'):
        return env.explain(state=state, action=action, next_state=next_state)
    return {"explanation": "No explain() method implemented for this environment."}
