import pytest
from src.core.multiagent import MultiAgentSystem

class DummyAgent:
    def __init__(self):
        self.group = None

def test_dynamic_group_module_creation_and_removal():
    agents = [DummyAgent() for _ in range(3)]
    mas = MultiAgentSystem(agents)
    mas.form_group('g1', [0, 1])
    # Module should exist after group creation
    module = mas.get_group_module('g1')
    assert module is not None
    assert 'rules' in module and 'subnetwork' in module
    # Add a custom module
    custom_mod = {'rules': ['rA'], 'subnetwork': 'snA'}
    mas.add_group_module('g2', custom_mod)
    assert mas.get_group_module('g2') == custom_mod
    # Remove module
    mas.remove_group_module('g1')
    assert mas.get_group_module('g1') is None
    # Dissolve group should also remove module
    mas.form_group('g3', [2])
    mas.dissolve_group('g3')
    assert mas.get_group_module('g3') is None
