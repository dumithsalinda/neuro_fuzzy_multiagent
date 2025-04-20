try:
    import rospy
    from std_msgs.msg import String
except ImportError:
    rospy = None

class ROSBridge:
    """
    Minimal ROS integration for agent-environment communication.
    Allows publishing/subscribing to ROS topics from agents or environments.
    """
    def __init__(self, node_name='nfma_ros_bridge'):
        if rospy is None:
            raise ImportError('rospy is not installed. ROS integration requires rospy.')
        rospy.init_node(node_name, anonymous=True)
        self.publishers = {}
        self.subscribers = {}

    def create_publisher(self, topic, msg_type=String, queue_size=10):
        pub = rospy.Publisher(topic, msg_type, queue_size=queue_size)
        self.publishers[topic] = pub
        return pub

    def create_subscriber(self, topic, callback, msg_type=String):
        sub = rospy.Subscriber(topic, msg_type, callback)
        self.subscribers[topic] = sub
        return sub

    def publish(self, topic, msg):
        if topic in self.publishers:
            self.publishers[topic].publish(msg)
        else:
            raise ValueError(f'No publisher for topic {topic}')

    def spin(self):
        rospy.spin()
