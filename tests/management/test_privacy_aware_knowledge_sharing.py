import pytest
from agents.dummy_agent import DummyAgent

from neuro_fuzzy_multiagent.core.agents.agent import Agent
from neuro_fuzzy_multiagent.core.management.multiagent import MultiAgentSystem


def test_public_knowledge_sharing():
    a1 = DummyAgent()
    a2 = DummyAgent()
    system = MultiAgentSystem([a1, a2])
    knowledge = {"foo": 1, "privacy": "public"}
    a1.share_knowledge(knowledge, system=system)
    assert knowledge in a2.knowledge_received


def test_private_knowledge_not_shared():
    a1 = DummyAgent()
    a2 = DummyAgent()
    system = MultiAgentSystem([a1, a2])
    knowledge = {"foo": 1, "privacy": "private"}
    a1.share_knowledge(knowledge, system=system)
    assert knowledge not in a2.knowledge_received


def test_group_only_knowledge_sharing():
    a1 = DummyAgent(group="A")
    a2 = DummyAgent(group="A")
    a3 = DummyAgent(group="B")
    system = MultiAgentSystem([a1, a2, a3])
    knowledge = {"foo": 2, "privacy": "group-only"}
    a1.share_knowledge(knowledge, system=system, group="A")
    assert knowledge in a2.knowledge_received
    assert knowledge not in a3.knowledge_received


def test_recipient_list_knowledge_sharing():
    a1 = DummyAgent()
    a2 = DummyAgent()
    a3 = DummyAgent()
    system = MultiAgentSystem([a1, a2, a3])
    knowledge = {"foo": 3, "privacy": "recipient-list"}
    a1.share_knowledge(knowledge, system=system, recipients=[a2])
    assert knowledge in a2.knowledge_received
    assert knowledge not in a3.knowledge_received
