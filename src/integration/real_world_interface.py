"""
Abstraction layer for real-world integration: robots, APIs, and IoT sensors.
Supports both observation (read) and action (write) operations.
"""

import requests


class RealWorldInterface:
    def __init__(self):
        pass

    def get_observation(self, source_type, config):
        """
        Fetch observation from a real-world source.
        source_type: 'robot', 'api', 'iot_sensor'
        config: dict with connection info (url, topic, etc.)
        """
        if source_type == "api":
            response = requests.get(config["url"], timeout=2)
            return response.json()
        elif source_type == "iot_sensor":
            # Example: fetch from MQTT broker or HTTP endpoint
            # Placeholder: HTTP GET
            response = requests.get(config["url"], timeout=2)
            return response.json()
        elif source_type == "robot":
            # Example: send command or fetch state from robot API
            response = requests.get(config["url"], timeout=2)
            return response.json()
        else:
            raise ValueError(f"Unsupported source_type: {source_type}")

    def send_action(self, target_type, action, config):
        """
        Send an action/command to a real-world target.
        target_type: 'robot', 'api', 'iot_sensor'
        action: dict or primitive
        config: dict with connection info (url, topic, etc.)
        """
        if target_type == "api":
            response = requests.post(config["url"], json=action, timeout=2)
            return response.json()
        elif target_type == "iot_sensor":
            # Example: HTTP POST or MQTT publish
            response = requests.post(config["url"], json=action, timeout=2)
            return response.json()
        elif target_type == "robot":
            # Example: send command to robot API
            response = requests.post(config["url"], json=action, timeout=2)
            return response.json()
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")
