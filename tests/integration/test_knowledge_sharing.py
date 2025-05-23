import pytest
from agents.dummy_agent import DummyAgent

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.agents.laws import (
    LawViolation,
    clear_laws,
    register_law,
)
from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem


def test_agent_knowledge_sharing_and_receiving():
    clear_laws("knowledge")
    sender = DummyAgent()
    receiver = DummyAgent()
    system = MultiAgentSystem([sender, receiver])
    knowledge = {"foo": 42}
    sender.share_knowledge(knowledge, system=system)
    assert knowledge in receiver.knowledge_received


def test_knowledge_sharing_blocked_by_law():
    clear_laws("knowledge")

    def block_foo(knowledge, state=None):
        return knowledge.get("foo") != 99

    register_law(block_foo, category="knowledge")
    sender = DummyAgent()
    receiver = DummyAgent()
    system = MultiAgentSystem([sender, receiver])
    knowledge = {"foo": 99}
    sender.share_knowledge(knowledge, system=system)
    assert knowledge not in receiver.knowledge_received


def test_multiagent_broadcast_knowledge_and_aggregation():
    # Only register group law for numeric actions, not for knowledge sharing
    from neuro_fuzzy_multiagent.core.agents.laws import clear_laws

    clear_laws("group")
    a1 = DummyAgent()
    a2 = DummyAgent()
    a3 = DummyAgent()
    system = MultiAgentSystem([a1, a2, a3])
    knowledge = {"bar": 7}
    system.broadcast_knowledge(knowledge, sender=a1)
    assert knowledge in a2.knowledge_received
    assert knowledge in a3.knowledge_received
    # Simulate agents learning different knowledge
    a1.online_knowledge = {"k": 1}
    a2.online_knowledge = {"k": 2}
    a3.online_knowledge = None
    agg = system.aggregate_knowledge()
    assert {"k": 1} in agg and {"k": 2} in agg and None not in agg
