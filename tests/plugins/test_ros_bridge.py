import sys

import pytest

ros_installed = False
try:
    import rospy

    from neuro_fuzzy_multiagent.core.plugins.ros_bridge import ROSBridge

    ros_installed = True
except ImportError:
    ros_installed = False


def test_ros_import():
    if not ros_installed:
        pytest.skip("rospy not installed")
    bridge = ROSBridge(node_name="test_nfma_ros_bridge")
    assert bridge is not None
