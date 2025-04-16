import numpy as np
import pytest
from src.core.online_learning import OnlineLearnerMixin

class DummyOnlineAgent(OnlineLearnerMixin):
    def __init__(self):
        self.knowledge = None
    def integrate_online_knowledge(self, knowledge):
        self.knowledge = knowledge

def test_learn_from_url(monkeypatch):
    # Simulate a web resource
    dummy_text = "rule1: IF x > 0 THEN y = 1"
    class DummyResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass
    def dummy_get(url):
        return DummyResponse(dummy_text)
    monkeypatch.setattr("requests.get", dummy_get)
    agent = DummyOnlineAgent()
    agent.learn_from_url("http://dummy.url/rules.txt")
    assert agent.knowledge == dummy_text

def test_learn_from_url_with_parse(monkeypatch):
    dummy_text = "1,2,3\n4,5,6"
    class DummyResponse:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            pass
    def dummy_get(url):
        return DummyResponse(dummy_text)
    monkeypatch.setattr("requests.get", dummy_get)
    agent = DummyOnlineAgent()
    def parse_fn(data):
        return [list(map(int, line.split(","))) for line in data.strip().split("\n")]
    agent.learn_from_url("http://dummy.url/data.csv", parse_fn=parse_fn)
    assert agent.knowledge == [[1,2,3],[4,5,6]]
