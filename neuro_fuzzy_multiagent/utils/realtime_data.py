"""
Utilities for real-time data integration: MQTT, REST API, and mock sources.
"""

import random
import requests

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


def get_mock_sensor_value():
    return random.uniform(-1, 1)


def get_rest_api_value(url):
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        return float(resp.text.strip())
    except Exception as e:
        return None


class MQTTClientWrapper:
    def __init__(self, broker, topic, on_message=None):
        if mqtt is None:
            raise ImportError("paho-mqtt not installed")
        self.client = mqtt.Client()
        self.broker = broker
        self.topic = topic
        self.value = None
        if on_message is None:
            on_message = self._on_message
        self.client.on_message = on_message

    def _on_message(self, client, userdata, msg):
        try:
            self.value = float(msg.payload.decode())
        except Exception:
            self.value = None

    def connect_and_subscribe(self):
        self.client.connect(self.broker)
        self.client.subscribe(self.topic)
        self.client.loop_start()

    def get_value(self):
        return self.value

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
