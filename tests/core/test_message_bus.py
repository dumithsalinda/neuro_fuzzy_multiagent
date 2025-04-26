import pytest
from src.core.management.message_bus import MessageBus
from src.core.agents.agent import Agent


class DummyAgent(Agent):
    def __init__(self, name):
        super().__init__(model=None)
        self.name = name
        self.received = []

    def receive_message(self, message, sender=None):
        self.received.append((message, sender.name if sender else None))


def test_message_bus_registration():
    bus = MessageBus()
    a1 = DummyAgent("A1")
    a2 = DummyAgent("A2")
    bus.register(a1)
    bus.register(a2, group="G")
    assert a1 in bus.agents
    assert a2 in bus.groups["G"]
    bus.unregister(a1)
    assert a1 not in bus.agents


def test_message_bus_direct():
    bus = MessageBus()
    a1 = DummyAgent("A1")
    a2 = DummyAgent("A2")
    bus.register(a1)
    bus.register(a2)
    bus.send("hello", a2)
    assert ("hello", None) in a2.received


def test_message_bus_broadcast():
    bus = MessageBus()
    a1 = DummyAgent("A1")
    a2 = DummyAgent("A2")
    a3 = DummyAgent("A3")
    bus.register(a1)
    bus.register(a2)
    bus.register(a3)
    bus.broadcast("hi-all", sender=a1)
    assert ("hi-all", "A1") in a2.received
    assert ("hi-all", "A1") in a3.received
    assert all(msg[1] != None for msg in a2.received + a3.received)


def test_message_bus_groupcast():
    bus = MessageBus()
    a1 = DummyAgent("A1")
    a2 = DummyAgent("A2")
    a3 = DummyAgent("A3")
    bus.register(a1, group="G")
    bus.register(a2, group="G")
    bus.register(a3, group="H")
    bus.groupcast("group-msg", "G", sender=a3)
    assert ("group-msg", "A3") in a1.received
    assert ("group-msg", "A3") in a2.received
    assert ("group-msg", "A3") not in a3.received
