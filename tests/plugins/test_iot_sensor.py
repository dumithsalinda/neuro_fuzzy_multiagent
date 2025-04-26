import time
from src.core.plugins.iot_sensor import IoTSensor


def test_iot_sensor_runs_and_callback():
    results = []

    def cb(name, value):
        results.append((name, value))

    sensor = IoTSensor("test_sensor", interval=0.1, callback=cb)
    sensor.start()
    time.sleep(0.35)
    sensor.stop()
    assert len(results) >= 3
    assert results[0][0] == "test_sensor"
    assert isinstance(results[0][1], float)
