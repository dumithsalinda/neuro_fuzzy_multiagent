import pytest
from core.agent import Agent
from core.online_learning import OnlineLearnerMixin
import json

class DummyAgent(Agent, OnlineLearnerMixin):
    def integrate_online_knowledge(self, knowledge):
        self.knowledge = knowledge


def test_learn_from_url_json(monkeypatch):
    # Simulate a JSON response
    class DummyResponse:
        text = json.dumps({'a': 1, 'b': 2})
        headers = {'Content-Type': 'application/json'}
        def raise_for_status(self): pass
    monkeypatch.setattr('requests.get', lambda url: DummyResponse())
    agent = DummyAgent(model=None)
    agent.learn_from_url('http://example.com/data.json')
    assert agent.knowledge == {'a': 1, 'b': 2}

def test_learn_from_url_csv(monkeypatch):
    # Simulate a CSV response
    class DummyResponse:
        text = 'x,y\n1,2\n3,4'
        headers = {'Content-Type': 'text/csv'}
        def raise_for_status(self): pass
    monkeypatch.setattr('requests.get', lambda url: DummyResponse())
    agent = DummyAgent(model=None)
    agent.learn_from_url('http://example.com/data.csv')
    assert agent.knowledge == [['x', 'y'], ['1', '2'], ['3', '4']]

def test_learn_from_url_plain(monkeypatch):
    # Simulate a plain text response
    class DummyResponse:
        text = 'hello world'
        headers = {'Content-Type': 'text/plain'}
        def raise_for_status(self): pass
    monkeypatch.setattr('requests.get', lambda url: DummyResponse())
    agent = DummyAgent(model=None)
    agent.learn_from_url('http://example.com/hello.txt')
    assert agent.knowledge == 'hello world'

def test_learn_from_url_error(monkeypatch):
    class DummyResponse:
        def raise_for_status(self): raise Exception('404')
    monkeypatch.setattr('requests.get', lambda url: DummyResponse())
    agent = DummyAgent(model=None)
    agent.learn_from_url('http://bad.url')
    # Should not raise, just print error
