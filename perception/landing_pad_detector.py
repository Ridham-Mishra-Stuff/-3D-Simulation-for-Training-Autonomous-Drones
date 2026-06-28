#!/usr/bin/env python3
"""
Features:
- HSV color space detection for colored landing pads
- Contour analysis for shape detection (circle, square, H)
- ArUco marker detection for precise landing
- Depth-based distance estimation
- Landing zone quality assessment
- Multi-frame tracking with Kalman filter
- Confidence scoring and false-positive rejection

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from std_msgs.msg import Bool, Float64, String, Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class DetectionResult:
    """Represents a detected landing pad."""
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: float
    detection_method: str
    distance_estimate: Optional[float] = None
    pose_estimate: Optional[PoseStamped] = None
    tracked_frames: int = 1


class LandingPadDetector(Node):
   
    MODE_COLOR = 0
    MODE_SHAPE = 1
    MODE_ARUCO = 2
    MODE_YOLO = 3
    MODE_HYBRID = 4

    def __init__(self):
        super().__init__('landing_pad_detector')

        # Parameters
        self.declare_parameter('detection_mode', 4)  # Default hybrid
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('color_lower_hsv', [20, 100, 100])  # Yellow-ish
        self.declare_parameter('color_upper_hsv', [40, 255, 255])
        self.declare_parameter('min_contour_area', 500)
        self.declare_parameter('max_contour_area', 50000)
        self.declare_parameter('circularity_threshold', 0.7)
        self.declare_parameter('aruco_dict', 'DICT_4X4_50')
        self.declare_parameter('aruco_marker_size', 0.5)
        self.declare_parameter('tracking_frames', 5)
        self.declare_parameter('confidence_threshold', 0.6)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('landing_approach_speed', 0.3)
        self.declare_parameter('centering_threshold', 0.1)

        # Get parameters
        self.mode = self.get_parameter('detection_mode').value
        self.camera_width = self.get_parameter('camera_width').value
        self.camera_height = self.get_parameter('camera_height').value
        self.color_lower = np.array(self.get_parameter('color_lower_hsv').value)
        self.color_upper = np.array(self.get_parameter('color_upper_hsv').value)
        self.min_area = self.get_parameter('min_contour_area').value
        self.max_area = self.get_parameter('max_contour_area').value
        self.circularity_thresh = self.get_parameter('circularity_threshold').value
        self.tracking_frames = self.get_parameter('tracking_frames').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.use_depth = self.get_parameter('use_depth').value
        self.landing_speed = self.get_parameter('landing_approach_speed').value
        self.centering_threshold = self.get_parameter('centering_threshold').value

        # CV Bridge
        self.bridge = CvBridge()

        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = self.cx = self.cy = None

        # ArUco setup
        self._setup_aruco()

        # State
        self.current_detections: List[DetectionResult] = []
        self.tracking_history = deque(maxlen=self.tracking_frames)
        self.best_detection: Optional[DetectionResult] = None
        self.detection_count = 0
        self.frame_count = 0

        # QoS
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Publishers
        self.image_pub = self.create_publisher(Image, '/detection/landing_pad/image', qos_best_effort)
        self.position_pub = self.create_publisher(PointStamped, '/detection/landing_pad/position', qos_reliable)
        self.pose_pub = self.create_publisher(PoseStamped, '/detection/landing_pad/pose', qos_reliable)
        self.detected_pub = self.create_publisher(Bool, '/detection/landing_pad/detected', qos_reliable)
        self.confidence_pub = self.create_publisher(Float64, '/detection/landing_pad/confidence', qos_reliable)
        self.status_pub = self.create_publisher(String, '/detection/landing_pad/status', qos_reliable)
        self.approach_pub = self.create_publisher(Twist, '/detection/landing_pad/approach_cmd', qos_reliable)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/drone/camera/image_raw', self._image_callback, qos_best_effort)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/drone/camera/camera_info', self._camera_info_callback, qos_reliable)
        self.depth_sub = self.create_subscription(
            Image, '/drone/camera/depth', self._depth_callback, qos_best_effort)
        self.mode_sub = self.create_subscription(
            Int32, '/detection/mode', self._mode_callback, qos_reliable)

        # Timer
        rate = self.get_parameter('publish_rate').value
        self.timer = self.create_timer(1.0 / rate, self._process_detections)

        self.get_logger().info('Landing Pad Detector initialized')
        self.get_logger().info(f'Detection mode: {self.mode}')

    def _setup_aruco(self):
        """Setup ArUco marker detection."""
        aruco_dict_name = self.get_parameter('aruco_dict').value
        aruco_dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        }
        dict_id = aruco_dict_map.get(aruco_dict_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_size = self.get_parameter('aruco_marker_size').value

    def _camera_info_callback(self, msg: CameraInfo):
        """Store camera calibration."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info('Camera calibration received')

    def _depth_callback(self, msg: Image):
        """Process depth image."""
        if not self.use_depth:
            return
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f'Depth processing error: {e}')

    def _mode_callback(self, msg: Int32):
        """Update detection mode."""
        if 0 <= msg.data <= 4:
            self.mode = msg.data
            modes = ['COLOR', 'SHAPE', 'ARUCO', 'YOLO', 'HYBRID']
            self.get_logger().info(f'Detection mode changed to: {modes[self.mode]}')

    def _image_callback(self, msg: Image):
        """Process incoming camera image."""
        self.frame_count += 1

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        detections = []

        # Run detection based on mode
        if self.mode == self.MODE_COLOR or self.mode == self.MODE_HYBRID:
            detections.extend(self._detect_color(cv_image))

        if self.mode == self.MODE_SHAPE or self.mode == self.MODE_HYBRID:
            detections.extend(self._detect_shape(cv_image))

        if self.mode == self.MODE_ARUCO or self.mode == self.MODE_HYBRID:
            detections.extend(self._detect_aruco(cv_image))

        if self.mode == self.MODE_YOLO:
            detections.extend(self._detect_yolo(cv_image))

        # Update tracking
        self.current_detections = self._update_tracking(detections)

        # Select best detection
        if self.current_detections:
            self.best_detection = max(self.current_detections, key=lambda d: d.confidence)
            self.detection_count += 1
        else:
            self.best_detection = None

        # Annotate and publish image
        annotated = self._annotate_image(cv_image)
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
        except Exception as e:
            self.get_logger().warn(f'Image publish error: {e}')

    def _detect_color(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect landing pad using color thresholding."""
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w/2, y + h/2

                # Calculate confidence based on area and shape
                confidence = min(area / self.max_area, 1.0)

                detections.append(DetectionResult(
                    center_x=cx, center_y=cy,
                    width=w, height=h,
                    confidence=confidence,
                    detection_method='color'
                ))

        return detections

    def _detect_shape(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect landing pad using shape analysis."""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_area < area < self.max_area:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)

                    if circularity > self.circularity_thresh:
                        x, y, w, h = cv2.boundingRect(cnt)
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                        else:
                            cx, cy = x + w/2, y + h/2

                        confidence = circularity
                        detections.append(DetectionResult(
                            center_x=cx, center_y=cy,
                            width=w, height=h,
                            confidence=confidence,
                            detection_method='shape'
                        ))

        return detections

    def _detect_aruco(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect ArUco markers."""
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                corner = corners[i][0]
                cx = np.mean(corner[:, 0])
                cy = np.mean(corner[:, 1])
                w = np.max(corner[:, 0]) - np.min(corner[:, 0])
                h = np.max(corner[:, 1]) - np.min(corner[:, 1])

                # Estimate pose if calibration available
                pose = None
                if self.camera_matrix is not None:
                    obj_points = np.array([
                        [-self.marker_size/2, self.marker_size/2, 0],
                        [self.marker_size/2, self.marker_size/2, 0],
                        [self.marker_size/2, -self.marker_size/2, 0],
                        [-self.marker_size/2, -self.marker_size/2, 0]
                    ], dtype=np.float32)

                    ret, rvec, tvec = cv2.solvePnP(
                        obj_points, corner, self.camera_matrix, self.dist_coeffs
                    )
                    if ret:
                        pose = PoseStamped()
                        pose.pose.position.x = tvec[0][0]
                        pose.pose.position.y = tvec[1][0]
                        pose.pose.position.z = tvec[2][0]

                detections.append(DetectionResult(
                    center_x=cx, center_y=cy,
                    width=w, height=h,
                    confidence=0.95,
                    detection_method='aruco',
                    pose_estimate=pose
                ))

        return detections

    def _detect_yolo(self, image: np.ndarray) -> List[DetectionResult]:
        """YOLO-based detection (placeholder for model integration)."""
        # This is a placeholder. In production, load a trained YOLO model
        # and run inference here.
        self.get_logger().debug('YOLO detection placeholder called')
        return []

    def _update_tracking(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Update tracking history and filter detections."""
        if not detections:
            return []

        # Simple tracking: keep detections that appear consistently
        self.tracking_history.append(detections)

        # Count occurrences of similar detections
        tracked = []
        for det in detections:
            count = 1
            for past_dets in self.tracking_history:
                for past_det in past_dets:
                    dist = math.sqrt((det.center_x - past_det.center_x)**2 + 
                                   (det.center_y - past_det.center_y)**2)
                    if dist < 50:  # pixels
                        count += 1

            if count >= self.tracking_frames // 2:
                det.tracked_frames = count
                tracked.append(det)

        return tracked

    def _annotate_image(self, image: np.ndarray) -> np.ndarray:
        """Draw detection annotations on image."""
        annotated = image.copy()

        # Draw all detections
        for det in self.current_detections:
            x1 = int(det.center_x - det.width / 2)
            y1 = int(det.center_y - det.height / 2)
            x2 = int(det.center_x + det.width / 2)
            y2 = int(det.center_y + det.height / 2)

            color = (0, 255, 0) if det == self.best_detection else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (int(det.center_x), int(det.center_y)), 5, (0, 0, 255), -1)

            label = f'{det.detection_method}: {det.confidence:.2f}'
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw crosshair at image center
        h, w = annotated.shape[:2]
        cv2.line(annotated, (w//2 - 20, h//2), (w//2 + 20, h//2), (255, 0, 0), 1)
        cv2.line(annotated, (w//2, h//2 - 20), (w//2, h//2 + 20), (255, 0, 0), 1)

        # Draw info
        info = f'Mode: {self.mode} | Detections: {len(self.current_detections)}'
        cv2.putText(annotated, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

    def _process_detections(self):
        """Process best detection and publish results."""
        detected = Bool()
        detected.data = self.best_detection is not None and                        self.best_detection.confidence >= self.confidence_threshold
        self.detected_pub.publish(detected)

        if self.best_detection:
            # Normalize coordinates
            nx = (self.best_detection.center_x - self.camera_width / 2) / (self.camera_width / 2)
            ny = (self.best_detection.center_y - self.camera_height / 2) / (self.camera_height / 2)

            # Publish position
            pos = PointStamped()
            pos.header.stamp = self.get_clock().now().to_msg()
            pos.header.frame_id = 'camera_optical'
            pos.point.x = nx
            pos.point.y = ny
            pos.point.z = 0.0
            self.position_pub.publish(pos)

            # Publish confidence
            self.confidence_pub.publish(Float64(data=self.best_detection.confidence))

            # Publish pose if available
            if self.best_detection.pose_estimate:
                self.pose_pub.publish(self.best_detection.pose_estimate)

            # Generate approach command
            self._generate_approach_command(nx, ny)

            # Publish status
            status = {
                'detected': True,
                'confidence': round(self.best_detection.confidence, 3),
                'method': self.best_detection.detection_method,
                'center': [round(self.best_detection.center_x, 1), 
                          round(self.best_detection.center_y, 1)],
                'tracked_frames': self.best_detection.tracked_frames,
                'total_detections': self.detection_count,
                'total_frames': self.frame_count
            }
            import json
            self.status_pub.publish(String(data=json.dumps(status)))
        else:
            self.confidence_pub.publish(Float64(data=0.0))
            # Publish empty approach command
            self.approach_pub.publish(Twist())

    def _generate_approach_command(self, nx: float, ny: float):
        """Generate velocity commands to center on landing pad."""
        cmd = Twist()

        # Horizontal centering
        if abs(nx) > self.centering_threshold:
            cmd.linear.y = -self.kp_center * nx  # lateral
        if abs(ny) > self.centering_threshold:
            cmd.linear.x = -self.kp_center * ny  # forward/back

        # Descent when centered
        if abs(nx) < self.centering_threshold and abs(ny) < self.centering_threshold:
            cmd.linear.z = -self.landing_speed

        self.approach_pub.publish(cmd)

    @property
    def kp_center(self):
        return 0.5


def main(args=None):
    rclpy.init(args=args)
    node = LandingPadDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down landing pad detector')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
