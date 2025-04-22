"""
marketplace.py

Utilities for fetching and parsing remote plugin index for the plugin marketplace.
"""
import requests
import logging

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/dumithsalinda/neuro_fuzzy_multiagent-plugin-index/main/plugins.json"


def fetch_remote_plugin_index(index_url=DEFAULT_INDEX_URL):
    """
    Fetch remote plugin index (JSON) from the given URL.
    Returns a list of plugin metadata dicts.
    """
    try:
        resp = requests.get(index_url, timeout=5)
        resp.raise_for_status()
        plugins = resp.json()
        assert isinstance(plugins, list)
        return plugins
    except Exception as e:
        logging.error(f"Failed to fetch remote plugin index: {e}")
        return []


def plugin_metadata_by_type(plugins):
    """
    Group remote plugin metadata by type.
    """
    result = {ptype: [] for ptype in ['environment', 'agent', 'sensor', 'actuator']}
    for plugin in plugins:
        ptype = plugin.get('type')
        if ptype in result:
            result[ptype].append(plugin)
    return result
