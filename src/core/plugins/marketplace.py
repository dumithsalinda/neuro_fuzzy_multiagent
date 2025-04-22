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


def download_and_save_plugin(plugin_meta):
    """
    Download plugin file from plugin_meta['source_url'] and save to correct directory.
    Returns (success: bool, message: str, saved_path: str or None)
    """
    ptype = plugin_meta.get('type')
    name = plugin_meta.get('name')
    url = plugin_meta.get('source_url')
    if not (ptype and name and url):
        return False, "Missing required plugin metadata.", None
    # Determine target directory
    if ptype == 'environment':
        target_dir = 'src/env/'
    elif ptype == 'agent':
        target_dir = 'src/core/agents/'
    elif ptype == 'sensor' or ptype == 'actuator':
        target_dir = 'src/plugins/'
    else:
        return False, f"Unknown plugin type: {ptype}", None
    target_path = f"{target_dir}{name}.py"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(resp.text)
        return True, f"Plugin '{name}' installed to {target_path}", target_path
    except Exception as e:
        logging.error(f"Failed to download/install plugin {name}: {e}")
        return False, f"Failed to download/install plugin: {e}", None


def get_local_plugin_version(ptype, name):
    """
    Try to extract __version__ or version comment from local plugin file.
    Returns version string or None.
    """
    if ptype == 'environment':
        path = f'src/env/{name}.py'
    elif ptype == 'agent':
        path = f'src/core/agents/{name}.py'
    elif ptype in ('sensor', 'actuator'):
        path = f'src/plugins/{name}.py'
    else:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for _ in range(10):  # Only scan first 10 lines
                line = f.readline()
                if not line:
                    break
                # __version__ = "1.2.3"
                if '__version__' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        return parts[1].strip().strip('"\'')
                # # version: 1.2.3
                if 'version:' in line.lower():
                    return line.split(':', 1)[1].strip().strip('"\'')
        return None
    except Exception:
        return None


def uninstall_plugin(ptype, name):
    """
    Delete the plugin file for the given type and name.
    Returns (success: bool, message: str, deleted_path: str or None)
    """
    import os
    if ptype == 'environment':
        path = f'src/env/{name}.py'
    elif ptype == 'agent':
        path = f'src/core/agents/{name}.py'
    elif ptype in ('sensor', 'actuator'):
        path = f'src/plugins/{name}.py'
    else:
        return False, f"Unknown plugin type: {ptype}", None
    try:
        if os.path.exists(path):
            os.remove(path)
            return True, f"Plugin '{name}' uninstalled from {path}", path
        else:
            return False, f"Plugin file not found: {path}", None
    except Exception as e:
        return False, f"Failed to uninstall plugin: {e}", None
