#!/usr/bin/env python3
"""

Features:
- Load YOLO model
- Process camera feed
- Publish detections
- Filter by confidence
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64
from cv_bridge import CvBridge
import cv2
import numpy as np


class YOLODetector(Node):


    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('classes', ['landing_pad', 'obstacle', 'gate'])

        self.conf_threshold = self.get_parameter('confidence_threshold').value

        self.bridge = CvBridge()
        self.model = None

        # Try to load model
        model_path = self.get_parameter('model_path').value
        if model_path:
            try:
                self.model = cv2.dnn.readNet(model_path)
                self.get_logger().info(f'YOLO model loaded: {model_path}')
            except Exception as e:
                self.get_logger().warn(f'Could not load YOLO model: {e}')

        self.image_pub = self.create_publisher(Image, '/detection/yolo/image', 10)
        self.detection_pub = self.create_publisher(String, '/detection/yolo/detections', 10)

        self.create_subscription(Image, '/drone/camera/image_raw', self._image_callback, 10)

        self.get_logger().info('YOLO Detector initialized')

    def _image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        detections = []

        if self.model is not None:
            detections = self._run_inference(cv_image)
        else:
            # Placeholder: simulate detection
            h, w = cv_image.shape[:2]
            cv2.rectangle(cv_image, (w//2-50, h//2-50), (w//2+50, h//2+50), (0, 255, 0), 2)
            cv2.putText(cv_image, 'YOLO Placeholder', (w//2-60, h//2-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().warn(f'Publish error: {e}')

        # Publish detections
        import json
        self.detection_pub.publish(String(data=json.dumps(detections)))

    def _run_inference(self, image):
        """Run YOLO inference. Placeholder implementation."""
        return []


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
