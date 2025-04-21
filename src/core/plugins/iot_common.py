import threading
import time
import random
import requests
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

class IoTDevice:
    """
    General IoT device abstraction supporting MQTT, HTTP, and simulated sensors/actuators.
    Usage:
        device = IoTDevice(
            name='livingroom_temp',
            mode='mqtt',
            mqtt_config={...},
            callback=cb
        )
        device.start()
    """
    def __init__(self, name, mode='sim', interval=1.0, value_fn=None, callback=None,
                 mqtt_config=None, http_config=None):
        self.name = name
        self.mode = mode  # 'sim', 'mqtt', 'http', 'gpio', etc
        self.interval = interval
        self.value_fn = value_fn if value_fn else self.default_value_fn
        self.callback = callback
        self.mqtt_config = mqtt_config or {}
        self.http_config = http_config or {}
        self._running = False
        self._thread = None
        self._mqtt_client = None

    def default_value_fn(self):
        # Simulate a temperature sensor
        return 20 + 5 * random.random()

    def start(self):
        self._running = True
        if self.mode == 'sim':
            self._thread = threading.Thread(target=self._run_sim)
            self._thread.daemon = True
            self._thread.start()
        elif self.mode == 'mqtt' and MQTT_AVAILABLE:
            self._setup_mqtt()
        elif self.mode == 'http':
            self._thread = threading.Thread(target=self._run_http)
            self._thread.daemon = True
            self._thread.start()
        else:
            raise NotImplementedError(f"Mode {self.mode} not supported or missing dependencies.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()

    def _run_sim(self):
        while self._running:
            value = self.value_fn()
            if self.callback:
                self.callback(self.name, value)
            time.sleep(self.interval)

    def _setup_mqtt(self):
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt is required for MQTT mode.")
        self._mqtt_client = mqtt.Client()
        if 'on_message' in self.mqtt_config:
            self._mqtt_client.on_message = self.mqtt_config['on_message']
        else:
            self._mqtt_client.on_message = self._on_mqtt_message
        self._mqtt_client.connect(self.mqtt_config.get('broker', 'localhost'),
                                 self.mqtt_config.get('port', 1883))
        topic = self.mqtt_config.get('topic', f'{self.name}/value')
        self._mqtt_client.subscribe(topic)
        self._mqtt_client.loop_start()

    def _on_mqtt_message(self, client, userdata, msg):
        value = msg.payload.decode()
        if self.callback:
            self.callback(self.name, value)

    def _run_http(self):
        url = self.http_config.get('url', '')
        method = self.http_config.get('method', 'GET')
        while self._running:
            if method == 'GET':
                resp = requests.get(url)
                value = resp.json().get('value')
            elif method == 'POST':
                resp = requests.post(url, data=self.http_config.get('data', {}))
                value = resp.json().get('value')
            else:
                value = None
            if self.callback:
                self.callback(self.name, value)
            time.sleep(self.interval)

    def send_command(self, value):
        if self.mode == 'mqtt' and self._mqtt_client:
            topic = self.mqtt_config.get('command_topic', f'{self.name}/set')
            self._mqtt_client.publish(topic, str(value))
        elif self.mode == 'http':
            url = self.http_config.get('command_url', self.http_config.get('url', ''))
            requests.post(url, json={'value': value})
        else:
            raise NotImplementedError(f"Command not supported for mode {self.mode}")
