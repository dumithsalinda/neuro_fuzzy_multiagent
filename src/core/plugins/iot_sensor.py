import threading
import time
import random


class IoTSensor:
    """
    Simulated IoT Sensor for smart environment demos.
    Can be extended to support real hardware (e.g., MQTT, HTTP, serial, etc).
    """

    def __init__(self, name, interval=1.0, value_fn=None, callback=None):
        self.name = name
        self.interval = interval
        self.value_fn = value_fn if value_fn else self.default_value_fn
        self.callback = callback
        self._running = False
        self._thread = None

    def default_value_fn(self):
        # Simulate a temperature sensor (Celsius)
        return 20 + 5 * random.random()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        while self._running:
            value = self.value_fn()
            if self.callback:
                self.callback(self.name, value)
            time.sleep(self.interval)
