import time
import pytest
from src.core.plugins.iot_common import IoTDevice


def test_sim_mode():
    results = []

    def cb(name, value):
        results.append((name, value))

    device = IoTDevice("sim_sensor", mode="sim", interval=0.1, callback=cb)
    device.start()
    time.sleep(0.35)
    device.stop()
    assert len(results) >= 3
    assert results[0][0] == "sim_sensor"
    assert isinstance(float(results[0][1]), float)


@pytest.mark.skipif(
    "paho.mqtt.client" not in globals(), reason="paho-mqtt not installed"
)
def test_mqtt_mode():
    # This test is a placeholder; requires local MQTT broker
    def cb(name, value):
        pass

    device = IoTDevice(
        "mqtt_sensor",
        mode="mqtt",
        mqtt_config={"broker": "localhost", "topic": "test/topic"},
        callback=cb,
    )
    # device.start()  # Uncomment if MQTT broker is available
    # device.stop()
    pass


def test_http_mode(monkeypatch):
    # Simulate HTTP GET
    class DummyResp:
        def json(self):
            return {"value": 42}

    def fake_get(url):
        return DummyResp()

    results = []
    monkeypatch.setattr("requests.get", fake_get)
    device = IoTDevice(
        "http_sensor",
        mode="http",
        http_config={"url": "http://dummy"},
        interval=0.1,
        callback=lambda n, v: results.append((n, v)),
    )
    device.start()
    time.sleep(0.25)
    device.stop()
    assert results[0][1] == 42
