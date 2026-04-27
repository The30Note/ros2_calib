# MIT License
#
# Copyright (c) 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import datetime
import os
from typing import Dict, List, Optional

import yaml

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import ros_utils
from . import tf_transformations as tf
from .bag_handler import (
    RosbagProcessingWorker,
    convert_to_mock,
    get_topic_info,
    get_total_message_count,
)
from .calibration_widget import CalibrationWidget
from .common import UIStyles
from .frame_selection_widget import FrameSelectionWidget
from .lidar2lidar_o3d_widget import launch_lidar2lidar_calibration
from .tf_graph_widget import TFGraphWidget


class _IntrinsicsPreviewView(QGraphicsView):
    """Scrollable/zoomable view for the intrinsics preview image."""

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMinimumHeight(360)

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class MainWindow(QMainWindow):
    # Signal for thread-safe calibration completion
    calibration_completed = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ros2_calib - Multi-Sensor Calibration Tool")
        self.setGeometry(100, 100, 1800, 800)
        self.setAcceptDrops(True)

        # Connect signals for thread-safe communication
        self.calibration_completed.connect(self.show_calibration_results)

        # Initialize data containers
        self.topics = {}
        self.bag_file = None
        self.selected_topics = {}
        self.tf_tree = {}
        self.current_transform = np.eye(4, dtype=np.float64)
        self.calibration_type = "LiDAR2Cam_Context"  # Default type
        self.calibrated_transform = np.eye(4, dtype=np.float64)
        self.original_source_frame = ""
        self.original_target_frame = ""
        self.tf_graph_window = None  # To hold a reference to the pop-up window
        self.active_camerainfo_msg = None

        # Set up stacked widget for multiple views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create and add the different views (order determines stacked widget indices)
        # 0: type selection, 1: load, 2: intrinsics, 3: transform, 4: frames, 5: results
        self.setup_calibration_type_view()
        self.setup_load_view()
        self.setup_intrinsics_view()
        self.setup_transform_view()
        self.setup_frame_selection_view()
        self.setup_results_view()

        # Start with the calibration type selection view
        self.stacked_widget.setCurrentIndex(0)

    def setup_calibration_type_view(self):
        """Setup the calibration type selection view."""
        self.calib_type_widget = QWidget()
        self.calib_type_layout = QVBoxLayout(self.calib_type_widget)

        self.calib_type_layout.addStretch()
        title_label = QLabel("Select Calibration Type")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        self.calib_type_layout.addWidget(title_label)
        desc_label = QLabel("Choose the type of calibration you want to perform:")
        desc_label.setStyleSheet("font-size: 14px; margin: 10px; color: #666;")
        desc_label.setAlignment(Qt.AlignCenter)
        self.calib_type_layout.addWidget(desc_label)

        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.addStretch()

        large_button_style = (
            UIStyles.HIGHLIGHT_BUTTON
            + """
        QPushButton {
            min-width: 220px;
            min-height: 110px;
            font-size: 18px;
            font-weight: 600;
            padding: 18px 30px;
            margin: 18px;
            border-radius: 14px;
            border: 2px solid transparent;
            background-color: #e45c28;
        }
        QPushButton:hover {
            background-color: #f37b3e;
        }
        QPushButton:pressed {
            background-color: #c64c1f;
            padding-top: 20px;
            padding-bottom: 16px;
        }
        """
        )

        self.lidar2cam_context_button = QPushButton("LiDAR ↔ Context Camera\nCalibration")
        self.lidar2cam_context_button.setStyleSheet(large_button_style)
        self.lidar2cam_context_button.clicked.connect(
            lambda: self.select_calibration_type("LiDAR2Cam_Context")
        )
        self._apply_button_shadow(self.lidar2cam_context_button)
        button_layout.addWidget(self.lidar2cam_context_button)

        button_layout.addSpacing(40)

        self.lidar2cam_zoom_button = QPushButton("LiDAR ↔ Zoom Camera\nCalibration")
        self.lidar2cam_zoom_button.setStyleSheet(large_button_style)
        self.lidar2cam_zoom_button.clicked.connect(
            lambda: self.select_calibration_type("LiDAR2Cam_Zoom")
        )
        self._apply_button_shadow(self.lidar2cam_zoom_button)
        button_layout.addWidget(self.lidar2cam_zoom_button)

        button_layout.addSpacing(40)

        self.lidar2lidar_button = QPushButton("LiDAR ↔ LiDAR\nCalibration")
        self.lidar2lidar_button.setStyleSheet(large_button_style)
        self.lidar2lidar_button.clicked.connect(lambda: self.select_calibration_type("LiDAR2LiDAR"))
        self._apply_button_shadow(self.lidar2lidar_button)
        button_layout.addWidget(self.lidar2lidar_button)

        button_layout.addStretch()
        self.calib_type_layout.addWidget(button_container)
        self.calib_type_layout.addStretch()
        self.stacked_widget.addWidget(self.calib_type_widget)

    def select_calibration_type(self, calib_type):
        """Handle calibration type selection."""
        self.calibration_type = calib_type
        print(f"[DEBUG] Selected calibration type: {calib_type}")
        self.update_load_view_for_calibration_type()
        self.stacked_widget.setCurrentIndex(1)

    def _apply_button_shadow(self, button: QPushButton) -> None:
        shadow = QGraphicsDropShadowEffect(button)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(0, 0, 0, 90))
        button.setGraphicsEffect(shadow)

    @property
    def _is_lidar_cam(self) -> bool:
        return self.calibration_type in ("LiDAR2Cam_Context", "LiDAR2Cam_Zoom")

    @property
    def _camera_type_label(self) -> str:
        """Short label for the selected camera type: 'context' or 'zoom'."""
        if self.calibration_type == "LiDAR2Cam_Context":
            return "context"
        if self.calibration_type == "LiDAR2Cam_Zoom":
            return "zoom"
        return ""

    @property
    def _child_frame(self) -> str:
        """Known child frame for the selected LiDAR-to-Camera calibration type."""
        if self.calibration_type == "LiDAR2Cam_Context":
            return "context_camera_frame"
        if self.calibration_type == "LiDAR2Cam_Zoom":
            return "zoom_camera_frame"
        return getattr(self, "camera_frame", "")

    @property
    def _parent_frame(self) -> str:
        """Known parent (LiDAR) frame for LiDAR-to-Camera calibration."""
        return "livox_frame"

    def update_load_view_for_calibration_type(self):
        """Update load view UI based on selected calibration type."""
        self.image_label.setVisible(self._is_lidar_cam)
        self.image_topic_combo.setVisible(self._is_lidar_cam)
        self.camerainfo_label.setVisible(self._is_lidar_cam)
        self.camerainfo_topic_combo.setVisible(self._is_lidar_cam)
        self.pointcloud2_label.setVisible(not self._is_lidar_cam)
        self.pointcloud2_topic_combo.setVisible(not self._is_lidar_cam)
        self.pointcloud_label.setText(
            "PointCloud2 Topic:" if self._is_lidar_cam else "PointCloud2 Topic (Source):"
        )
        type_label = {
            "LiDAR2Cam_Context": "LiDAR ↔ Context Camera",
            "LiDAR2Cam_Zoom": "LiDAR ↔ Zoom Camera",
            "LiDAR2LiDAR": "LiDAR ↔ LiDAR",
        }.get(self.calibration_type, self.calibration_type)
        self.selection_group.setTitle(f"Topic Selection — {type_label}")
        self.proceed_button.setText(
            "Proceed to Intrinsics" if self._is_lidar_cam else "Proceed to Transform Selection"
        )

    def setup_load_view(self):
        self.load_widget = QWidget()
        self.load_layout = QVBoxLayout(self.load_widget)
        load_section_layout = QHBoxLayout()
        self.load_bag_button = QPushButton("Load Rosbag")
        self.load_bag_button.clicked.connect(self.load_bag)
        load_section_layout.addWidget(self.load_bag_button)
        load_section_layout.addWidget(QLabel("ROS Version:"))
        self.ros_version_combo = QComboBox()
        self.ros_version_combo.addItems(["HUMBLE", "JAZZY"])
        load_section_layout.addWidget(self.ros_version_combo)
        drag_drop_label = QLabel("or Drag & Drop")
        drag_drop_label.setStyleSheet("color: #666; font-style: italic;")
        load_section_layout.addWidget(drag_drop_label)
        self.bag_path_label = QLabel("No rosbag loaded.")
        self.bag_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
        load_section_layout.addWidget(self.bag_path_label, 1)
        self.load_layout.addLayout(load_section_layout)
        topic_list_group = QGroupBox("Available Topics")
        topic_list_layout = QVBoxLayout(topic_list_group)
        self.topic_list_widget = QListWidget()
        topic_list_layout.addWidget(self.topic_list_widget)
        self.load_layout.addWidget(topic_list_group, 1)

        self.selection_group = QGroupBox()
        self.calib_topic_layout = QFormLayout(self.selection_group)
        self.image_topic_combo = QComboBox()
        self.image_topic_combo.currentIndexChanged.connect(self.auto_select_camera_info)
        self.camerainfo_topic_combo = QComboBox()
        self.pointcloud_topic_combo = QComboBox()
        self.pointcloud_topic_combo.currentTextChanged.connect(self.validate_lidar_topic_selection)
        self.pointcloud2_topic_combo = QComboBox()
        self.pointcloud2_topic_combo.currentTextChanged.connect(self.validate_lidar_topic_selection)
        self.frame_count_spinbox = QSpinBox()
        self.frame_count_spinbox.setRange(3, 20)
        self.frame_count_spinbox.setValue(6)
        self.frame_count_spinbox.setSuffix(" frames")

        self.image_label = QLabel("Image Topic:")
        self.camerainfo_label = QLabel("CameraInfo Topic:")
        self.pointcloud_label = QLabel("PointCloud2 Topic:")
        self.pointcloud2_label = QLabel("PointCloud2 Topic (Target):")

        self.calib_topic_layout.addRow(self.image_label, self.image_topic_combo)
        self.calib_topic_layout.addRow(self.camerainfo_label, self.camerainfo_topic_combo)
        self.calib_topic_layout.addRow(self.pointcloud_label, self.pointcloud_topic_combo)
        self.calib_topic_layout.addRow(self.pointcloud2_label, self.pointcloud2_topic_combo)
        self.calib_topic_layout.addRow("Frame Samples:", self.frame_count_spinbox)

        self.proceed_button = QPushButton()
        self.proceed_button.setEnabled(False)
        self.proceed_button.clicked.connect(self.process_rosbag_data)
        self.proceed_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        self.progress_bar = QProgressBar(visible=False, textVisible=True)
        self.calib_topic_layout.addRow(self.proceed_button)
        self.calib_topic_layout.addRow(self.progress_bar)

        self.load_layout.addWidget(self.selection_group)
        self.stacked_widget.addWidget(self.load_widget)
        self.update_load_view_for_calibration_type()

    def setup_intrinsics_view(self):
        self.intrinsics_widget = QWidget()
        outer = QVBoxLayout(self.intrinsics_widget)

        self.intrinsics_title_label = QLabel()
        self.intrinsics_title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 8px;")
        outer.addWidget(self.intrinsics_title_label)

        back_row = QHBoxLayout()
        back_btn = QPushButton("← Back to Topic Selection")
        back_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        back_row.addWidget(back_btn)
        back_row.addStretch()
        outer.addLayout(back_row)

        # --- left / right split ---
        content_row = QHBoxLayout()
        outer.addLayout(content_row, stretch=1)

        # Left panel — source selection + intrinsics text
        left = QVBoxLayout()
        content_row.addLayout(left, stretch=1)

        source_group = QGroupBox("Intrinsics Source")
        source_row = QHBoxLayout(source_group)
        self.intrinsics_default_btn = QPushButton("Use Default File")
        self.intrinsics_default_btn.clicked.connect(self._load_default_intrinsics)
        self.intrinsics_file_btn = QPushButton("Load from File…")
        self.intrinsics_file_btn.clicked.connect(self._browse_intrinsics_file)
        self.intrinsics_rosbag_btn = QPushButton("Use Rosbag Topic")
        self.intrinsics_rosbag_btn.clicked.connect(self._use_rosbag_intrinsics)
        source_row.addWidget(self.intrinsics_default_btn)
        source_row.addWidget(self.intrinsics_file_btn)
        source_row.addWidget(self.intrinsics_rosbag_btn)
        source_row.addStretch()
        left.addWidget(source_group)

        self.intrinsics_source_label = QLabel()
        self.intrinsics_source_label.setStyleSheet("color: #666; font-style: italic; margin: 4px;")
        left.addWidget(self.intrinsics_source_label)

        display_group = QGroupBox("Loaded Intrinsics")
        display_layout = QVBoxLayout(display_group)
        self.intrinsics_display = QTextEdit(readOnly=True, fontFamily="monospace")
        display_layout.addWidget(self.intrinsics_display)
        left.addWidget(display_group, stretch=1)

        # Right panel — image preview with grid overlay
        right = QVBoxLayout()
        content_row.addLayout(right, stretch=2)

        preview_group = QGroupBox("Intrinsics Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._intrinsics_scene = QGraphicsScene()
        self._intrinsics_pixmap_item = None
        self._intrinsics_view = _IntrinsicsPreviewView(self._intrinsics_scene)
        preview_layout.addWidget(self._intrinsics_view)

        self.intrinsics_rectify_check = QCheckBox("Rectify Image (undistort)")
        self.intrinsics_rectify_check.toggled.connect(self._render_intrinsics_preview)
        preview_layout.addWidget(self.intrinsics_rectify_check)

        right.addWidget(preview_group, stretch=1)

        # Continue button
        continue_row = QHBoxLayout()
        continue_row.addStretch()
        self.intrinsics_continue_btn = QPushButton("Continue to Frame Selection →")
        self.intrinsics_continue_btn.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        self.intrinsics_continue_btn.setEnabled(False)
        self.intrinsics_continue_btn.clicked.connect(self._intrinsics_continue)
        continue_row.addWidget(self.intrinsics_continue_btn)
        outer.addLayout(continue_row)

        self.stacked_widget.addWidget(self.intrinsics_widget)

    def _open_intrinsics_view(self):
        label = self._camera_type_label.capitalize()
        self.intrinsics_title_label.setText(f"Camera Intrinsics — {label} Camera")
        has_rosbag_topic = bool(
            getattr(self, "selected_topics_data", {}).get("camerainfo_topic", "")
        )
        self.intrinsics_rosbag_btn.setEnabled(has_rosbag_topic)
        self.intrinsics_rectify_check.blockSignals(True)
        self.intrinsics_rectify_check.setChecked(False)
        self.intrinsics_rectify_check.blockSignals(False)
        self._load_default_intrinsics()
        self.stacked_widget.setCurrentIndex(2)

    def _load_default_intrinsics(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path = os.path.join(data_dir, f"{self._camera_type_label}_intrinsics.yaml")
        if os.path.exists(path):
            self.active_camerainfo_msg = self._parse_camera_info_yaml(path)
            self.intrinsics_source_label.setText(
                f"Source: Default file ({os.path.basename(path)})"
            )
            self._refresh_intrinsics_display()
        else:
            self.intrinsics_source_label.setText("Default file not found.")
            self.intrinsics_display.setPlainText("")
            self.active_camerainfo_msg = None
            self.intrinsics_continue_btn.setEnabled(False)

    def _browse_intrinsics_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Intrinsics", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                self.active_camerainfo_msg = self._parse_camera_info_yaml(path)
                self.intrinsics_source_label.setText(f"Source: {path}")
                self._refresh_intrinsics_display()
            except Exception as e:
                self.intrinsics_display.setPlainText(f"Error loading file:\n{e}")
                self.intrinsics_continue_btn.setEnabled(False)

    def _use_rosbag_intrinsics(self):
        from .bag_handler import convert_to_mock

        camerainfo_topic = getattr(self, "selected_topics_data", {}).get("camerainfo_topic", "")
        raw = None
        if camerainfo_topic and hasattr(self, "frame_samples"):
            samples = self.frame_samples.get(camerainfo_topic)
            if samples:
                raw = samples[0]["data"]
        elif camerainfo_topic and hasattr(self, "selected_topics"):
            raw = self.selected_topics.get("raw_messages", {}).get(camerainfo_topic)

        if raw is None:
            self.intrinsics_source_label.setText("No camera_info data found in rosbag.")
            return

        self.active_camerainfo_msg = convert_to_mock(
            raw, self.topic_types.get(camerainfo_topic, "sensor_msgs/msg/CameraInfo")
        )
        self.intrinsics_source_label.setText(f"Source: Rosbag topic ({camerainfo_topic})")
        self._refresh_intrinsics_display()

    def _refresh_intrinsics_display(self):
        msg = self.active_camerainfo_msg
        if msg is None:
            self.intrinsics_display.setPlainText("No intrinsics loaded.")
            self.intrinsics_continue_btn.setEnabled(False)
            return
        k = list(msg.k)
        fx, fy = k[0], k[4]
        cx, cy = k[2], k[5]
        coeffs = ", ".join(f"{v:.6g}" for v in msg.d)
        text = (
            f"Image:        {msg.width} × {msg.height}\n"
            f"Camera name:  {getattr(msg, 'camera_name', '—')}\n"
            f"fx:           {fx:.5g}\n"
            f"fy:           {fy:.5g}\n"
            f"cx:           {cx:.5g}\n"
            f"cy:           {cy:.5g}\n"
            f"Distortion:   {msg.distortion_model}\n"
            f"Coefficients: [{coeffs}]"
        )
        self.intrinsics_display.setPlainText(text)
        self.intrinsics_continue_btn.setEnabled(True)
        self._render_intrinsics_preview()

    def _render_intrinsics_preview(self):
        """Render the first frame image with a projected 3D grid overlay."""
        msg = self.active_camerainfo_msg
        if msg is None:
            return

        # Get raw image from first available frame sample
        image_topic = getattr(self, "selected_topics_data", {}).get("image_topic", "")
        raw_img = None
        if image_topic and hasattr(self, "frame_samples"):
            samples = self.frame_samples.get(image_topic)
            if samples:
                raw_img = samples[0]["data"]
        if raw_img is None and hasattr(self, "selected_topics"):
            raw_img = self.selected_topics.get("raw_messages", {}).get(image_topic)
        if raw_img is None:
            return

        try:
            encoding = getattr(raw_img, "encoding", "")
            if not encoding or encoding not in ros_utils.name_to_dtypes:
                np_arr = np.frombuffer(raw_img.data, np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = ros_utils.image_to_numpy(raw_img)
                if "bgr" in encoding:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[WARNING] Could not decode preview image: {e}")
            return

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        is_fisheye = msg.distortion_model.lower() in ("fisheye", "equidistant")
        rectified = self.intrinsics_rectify_check.isChecked()

        if rectified:
            try:
                if is_fisheye:
                    bgr_in = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgr_out = cv2.fisheye.undistortImage(
                        bgr_in, K, D[:4].reshape(4, 1), None, K
                    )
                    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
                else:
                    bgr_in = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgr_out = cv2.undistort(bgr_in, K, D, None, K)
                    rgb = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"[WARNING] Undistort failed: {e}")

        rgb = self._draw_grid_on_image(rgb, K, D, is_fisheye, rectified)

        h, w = rgb.shape[:2]
        q_img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        if self._intrinsics_pixmap_item is None:
            self._intrinsics_pixmap_item = self._intrinsics_scene.addPixmap(pixmap)
        else:
            self._intrinsics_pixmap_item.setPixmap(pixmap)
        self._intrinsics_scene.setSceneRect(0, 0, w, h)
        self._intrinsics_view.fitInView(
            self._intrinsics_scene.sceneRect(), Qt.KeepAspectRatio
        )

    def _draw_grid_on_image(
        self,
        image_rgb: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        is_fisheye: bool,
        rectified: bool,
    ) -> np.ndarray:
        """Draw a projected 3D grid onto image_rgb and return the annotated copy."""
        h, w = image_rgb.shape[:2]
        n_lines = 9
        n_pts = 40  # polyline resolution per line
        depth = 1.0

        fx, fy = K[0, 0], K[1, 1]
        # Span slightly wider than the image FOV
        x_half = (w / 2) / fx * 1.05
        y_half = (h / 2) / fy * 1.05

        xs = np.linspace(-x_half, x_half, n_lines)
        ys = np.linspace(-y_half, y_half, n_lines)
        along_x = np.linspace(-x_half, x_half, n_pts)
        along_y = np.linspace(-y_half, y_half, n_pts)

        # Use zero distortion when drawing on the undistorted image
        D_use = np.zeros_like(D) if rectified else D
        rvec = tvec = np.zeros(3)
        out = image_rgb.copy()

        def project(pts_3d):
            pts = pts_3d.reshape(-1, 1, 3).astype(np.float64)
            try:
                if is_fisheye and not rectified:
                    d4 = D_use[:4].reshape(4, 1)
                    proj, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K, d4)
                else:
                    proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D_use)
                return proj.reshape(-1, 2)
            except Exception:
                return np.empty((0, 2))

        def draw_poly(pts2d):
            for i in range(len(pts2d) - 1):
                x1, y1 = int(round(pts2d[i][0])), int(round(pts2d[i][1]))
                x2, y2 = int(round(pts2d[i + 1][0])), int(round(pts2d[i + 1][1]))
                if max(abs(x1), abs(x2)) < w * 3 and max(abs(y1), abs(y2)) < h * 3:
                    cv2.line(out, (x1, y1), (x2, y2), (0, 220, 0), 1, cv2.LINE_AA)

        # Horizontal grid lines (fixed y, varying x)
        for y_val in ys:
            pts = np.column_stack([along_x, np.full(n_pts, y_val), np.full(n_pts, depth)])
            draw_poly(project(pts))

        # Vertical grid lines (fixed x, varying y)
        for x_val in xs:
            pts = np.column_stack([np.full(n_pts, x_val), along_y, np.full(n_pts, depth)])
            draw_poly(project(pts))

        return out

    def _intrinsics_continue(self):
        self.stacked_widget.setCurrentIndex(4)  # Frame selection view

    def _go_back_from_transform(self):
        if self._is_lidar_cam:
            self.stacked_widget.setCurrentIndex(4)  # Back to frame selection
        else:
            self.stacked_widget.setCurrentIndex(1)  # Back to load view

    def _parse_camera_info_yaml(self, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        msg = ros_utils.CameraInfo(
            height=data["image_height"],
            width=data["image_width"],
            distortion_model=data["distortion_model"],
            k=data["camera_matrix"]["data"],
            d=data["distortion_coefficients"]["data"],
            r=data["rectification_matrix"]["data"],
            p=data["projection_matrix"]["data"],
        )
        msg.camera_name = data.get("camera_name", "")
        return msg

    def setup_transform_view(self):
        self.transform_widget = QWidget()
        self.transform_layout = QVBoxLayout(self.transform_widget)
        self.tf_title_label = QLabel()
        self.tf_title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        self.transform_layout.addWidget(self.tf_title_label)

        back_button_layout = QHBoxLayout()
        self.back_button = QPushButton("← Back")
        self.back_button.clicked.connect(self._go_back_from_transform)
        back_button_layout.addWidget(self.back_button)
        back_button_layout.addStretch()
        self.transform_layout.addLayout(back_button_layout)

        tf_group = QGroupBox("Transform Source")
        tf_layout = QVBoxLayout(tf_group)
        tf_topic_layout = QHBoxLayout()
        tf_topic_layout.addWidget(QLabel("TF Topic:"))
        self.tf_topic_combo = QComboBox()
        self.tf_topic_combo.currentTextChanged.connect(self.on_tf_topic_changed)
        tf_topic_layout.addWidget(self.tf_topic_combo, 1)
        self.load_tf_button = QPushButton("Load TF Tree")
        self.load_tf_button.clicked.connect(self.load_tf_tree)
        tf_topic_layout.addWidget(self.load_tf_button)
        tf_layout.addLayout(tf_topic_layout)
        self.show_graph_button = QPushButton("Show TF Tree Graph")
        self.show_graph_button.clicked.connect(self.show_tf_graph)
        self.show_graph_button.setEnabled(False)
        tf_layout.addWidget(self.show_graph_button)
        self.tf_info_text = QTextEdit(plainText="No TF data loaded.", maximumHeight=150)
        tf_layout.addWidget(QLabel("TF Tree Information:"))
        tf_layout.addWidget(self.tf_info_text)
        self.transform_layout.addWidget(tf_group)

        manual_group = QGroupBox("Manual Transform Input")
        manual_layout = QGridLayout(manual_group)
        self.tx_input, self.ty_input, self.tz_input = (
            QLineEdit("0.0"),
            QLineEdit("0.0"),
            QLineEdit("0.0"),
        )
        self.rx_input, self.ry_input, self.rz_input = (
            QLineEdit("0.0"),
            QLineEdit("0.0"),
            QLineEdit("0.0"),
        )
        manual_layout.addWidget(QLabel("Translation (x, y, z):"), 0, 0)
        manual_layout.addWidget(self.tx_input, 0, 1)
        manual_layout.addWidget(self.ty_input, 0, 2)
        manual_layout.addWidget(self.tz_input, 0, 3)
        manual_layout.addWidget(QLabel("Rotation (roll, pitch, yaw) [deg]:"), 1, 0)
        manual_layout.addWidget(self.rx_input, 1, 1)
        manual_layout.addWidget(self.ry_input, 1, 2)
        manual_layout.addWidget(self.rz_input, 1, 3)
        self.update_manual_button = QPushButton("Update Transform from Manual Input")
        self.update_manual_button.clicked.connect(self.update_manual_transform)
        manual_layout.addWidget(self.update_manual_button, 2, 0, 1, 4)
        self.transform_layout.addWidget(manual_group)

        transform_group = QGroupBox("Current Transformation Matrix")
        transform_layout_inner = QVBoxLayout(transform_group)
        self.transform_display = QTextEdit(maximumHeight=180, readOnly=True, fontFamily="monospace")
        transform_layout_inner.addWidget(self.transform_display)
        self.transform_layout.addWidget(transform_group)

        button_layout = QHBoxLayout()
        self.use_identity_button = QPushButton("Use Identity Transform")
        self.use_identity_button.clicked.connect(self.use_identity_transform)
        button_layout.addWidget(self.use_identity_button)
        button_layout.addStretch()
        self.confirm_button = QPushButton("Start Calibration")
        self.confirm_button.clicked.connect(self.confirm_transformation)
        self.confirm_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        button_layout.addWidget(self.confirm_button)
        self.transform_layout.addLayout(button_layout)
        self.update_transform_display()
        self.stacked_widget.addWidget(self.transform_widget)

    def setup_frame_selection_view(self):
        self.frame_selection_widget = FrameSelectionWidget(self)
        self.frame_selection_widget.frame_selected.connect(self.on_frame_selected)
        self.frame_selection_widget.frames_accumulated.connect(self.on_frames_accumulated)
        self.stacked_widget.addWidget(self.frame_selection_widget)

    def setup_results_view(self):
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        title = QLabel("Calibration Export")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.results_layout.addWidget(title)
        back_layout = QHBoxLayout()
        self.results_back_button = QPushButton("← Back to Calibration")
        self.results_back_button.clicked.connect(self.go_back_to_calibration)
        back_layout.addWidget(self.results_back_button)

        # Add frame selection dropdowns right next to the back button
        back_layout.addWidget(QLabel("Source Frame:"))
        self.source_frame_combo = QComboBox()
        self.source_frame_combo.currentTextChanged.connect(self.update_target_transform)
        back_layout.addWidget(self.source_frame_combo)

        back_layout.addWidget(QLabel("Target Frame:"))
        self.target_frame_combo = QComboBox()
        self.target_frame_combo.currentTextChanged.connect(self.update_target_transform)
        back_layout.addWidget(self.target_frame_combo)

        back_layout.addStretch()
        self.results_layout.addLayout(back_layout)

        chain_graph_group = QGroupBox("Transformation Path")
        chain_graph_layout = QVBoxLayout(chain_graph_group)
        self.chain_display = QTextEdit(maximumHeight=60, readOnly=True)
        chain_graph_layout.addWidget(self.chain_display)
        self.graph_container = QWidget(minimumHeight=400, maximumHeight=500)
        self.init_graph_placeholder()
        chain_graph_layout.addWidget(self.graph_container)
        self.results_layout.addWidget(chain_graph_group)

        results_content_layout = QHBoxLayout()
        left_group = QGroupBox("Overall Calibrated Transform")
        left_layout = QVBoxLayout(left_group)
        self.calibration_result_display = QTextEdit(readOnly=True, fontFamily="monospace")
        left_layout.addWidget(self.calibration_result_display)
        results_content_layout.addWidget(left_group)
        right_group = QGroupBox("Export Target Transform (Selected Frames)")
        right_layout = QVBoxLayout(right_group)
        self.final_transform_display = QTextEdit(readOnly=True, fontFamily="monospace")
        right_layout.addWidget(self.final_transform_display)
        results_content_layout.addWidget(right_group)
        self.results_layout.addLayout(results_content_layout)

        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_calibration_button = QPushButton("Export Target Transform")
        self.export_calibration_button.clicked.connect(self.export_calibration_result)
        self.export_calibration_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        export_layout.addWidget(self.export_calibration_button)
        export_layout.addStretch()
        self.results_layout.addLayout(export_layout)
        self.stacked_widget.addWidget(self.results_widget)

    def init_graph_placeholder(self):
        """Initialize or reset the graph container with a placeholder."""
        layout = self.graph_container.layout()
        if layout is None:
            layout = QVBoxLayout(self.graph_container)

        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def load_bag(self):
        bags_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bags")
        start_dir = bags_dir if os.path.isdir(bags_dir) else os.path.dirname(os.path.dirname(__file__))
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Rosbag", start_dir, "MCAP Rosbag (*.mcap)")
        if file_path:
            self.load_bag_from_path(file_path)

    def find_yaml_file(self, mcap_path):
        directory = os.path.dirname(mcap_path)
        mcap_basename = os.path.splitext(os.path.basename(mcap_path))[0]
        metadata_yaml = os.path.join(directory, "metadata.yaml")
        if os.path.exists(metadata_yaml):
            return metadata_yaml
        matching_yaml = os.path.join(directory, f"{mcap_basename}.yaml")
        if os.path.exists(matching_yaml):
            return matching_yaml
        return None

    def process_dropped_path(self, path):
        """Process a dropped file or folder path."""
        self.bag_path_label.setStyleSheet("padding: 5px; border: 1px solid red;")
        if os.path.isfile(path) and path.endswith(".mcap"):
            if self.find_yaml_file(path):
                self.load_bag_from_path(path)
            else:
                self.bag_path_label.setText(
                    "Error: Corresponding metadata.yaml or matching .yaml not found."
                )
        elif os.path.isdir(path):
            mcap_files = [f for f in os.listdir(path) if f.endswith(".mcap")]
            if len(mcap_files) == 1:
                self.process_dropped_path(os.path.join(path, mcap_files[0]))
            elif not mcap_files:
                self.bag_path_label.setText("Error: No .mcap file found in the dropped folder.")
            else:
                self.bag_path_label.setText("Error: Multiple .mcap files found. Please drop one.")
        else:
            self.bag_path_label.setText("Error: Please drop a valid .mcap file or folder.")

    def load_bag_from_path(self, file_path):
        """Load bag from a specific file path."""
        try:
            self.bag_file = file_path
            self.bag_path_label.setText(file_path)
            self.bag_path_label.setStyleSheet("padding: 5px; border: 1px solid gray;")
            ros_version = self.ros_version_combo.currentText()
            self.topics = get_topic_info(file_path, ros_version)
            self.update_topic_widgets()
        except Exception as e:
            self.bag_path_label.setText(f"Error loading bag: {str(e)}")
            self.bag_path_label.setStyleSheet("padding: 5px; border: 1px solid red;")

    def update_topic_widgets(self):
        self.topic_list_widget.clear()
        self.image_topic_combo.clear()
        self.pointcloud_topic_combo.clear()
        self.pointcloud2_topic_combo.clear()
        self.camerainfo_topic_combo.clear()

        topic_types = {
            "image": [
                t
                for t, m, _ in self.topics
                if m in ["sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"]
            ],
            "pointcloud": [t for t, m, _ in self.topics if m == "sensor_msgs/msg/PointCloud2"],
            "camerainfo": [t for t, m, _ in self.topics if m == "sensor_msgs/msg/CameraInfo"],
        }

        for topic, msgtype, msgcount in self.topics:
            self.topic_list_widget.addItem(f"{topic} ({msgtype}) - {msgcount} messages")

        self.image_topic_combo.addItems(topic_types["image"])
        self.pointcloud_topic_combo.addItems(topic_types["pointcloud"])
        self.pointcloud2_topic_combo.addItems(topic_types["pointcloud"])
        self.camerainfo_topic_combo.addItems(topic_types["camerainfo"])
        self._auto_select_topics()
        self.update_proceed_button_state()

    def _auto_select_topics(self):
        """Auto-select topics based on calibration type using known naming conventions."""
        preferred = {
            "LiDAR2Cam_Context": {
                "image": "/context_camera/main",
                "camerainfo": "/context_camera/camera_info",
                "pointcloud": "/livox/lidar",
            },
            "LiDAR2Cam_Zoom": {
                "image": "/zoom_camera/main",
                "camerainfo": "/zoom_camera/camera_info",
                "pointcloud": "/livox/lidar",
            },
            "LiDAR2LiDAR": {
                "pointcloud": "/livox/lidar",
            },
        }.get(self.calibration_type, {})

        def _select(combo, topic):
            idx = combo.findText(topic, Qt.MatchExactly)
            if idx != -1:
                combo.setCurrentIndex(idx)

        _select(self.pointcloud_topic_combo, preferred.get("pointcloud", ""))
        _select(self.pointcloud2_topic_combo, preferred.get("pointcloud", ""))
        if self._is_lidar_cam:
            _select(self.image_topic_combo, preferred.get("image", ""))
            _select(self.camerainfo_topic_combo, preferred.get("camerainfo", ""))

    def auto_select_camera_info(self, index):
        if not self._is_lidar_cam or index == -1:
            self.update_proceed_button_state()
            return

        camera_info_count = self.camerainfo_topic_combo.count()
        if camera_info_count == 0:
            self.update_proceed_button_state()
            return

        image_topic = self.image_topic_combo.currentText()
        if not image_topic:
            self.update_proceed_button_state()
            return

        transport_suffixes = {
            "compressed",
            "compressedDepth",
            "compressed_depth",
            "Theora",
            "theora",
        }

        def candidate_paths(topic: str) -> List[str]:
            tokens = [part for part in topic.strip("/").split("/") if part]
            while tokens and tokens[-1] in transport_suffixes:
                tokens.pop()

            variants: List[str] = []
            if tokens:
                variants.append("/" + "/".join(tokens[:-1] + ["camera_info"]))

            for idx in range(len(tokens) - 1, -1, -1):
                if "image" in tokens[idx]:
                    replaced = tokens[:]
                    replaced[idx] = "camera_info"
                    variants.append("/" + "/".join(replaced))
                    variants.append("/" + "/".join(replaced[: idx + 1]))
                    break

            base_path = "/" + "/".join(tokens[:-1]) if len(tokens) > 1 else ""
            if base_path:
                variants.append(f"{base_path}/camera_info")

            seen = set()
            unique_variants: List[str] = []
            for item in variants:
                if item and item not in seen:
                    seen.add(item)
                    unique_variants.append(item)
            return unique_variants

        for candidate in candidate_paths(image_topic):
            found_index = self.camerainfo_topic_combo.findText(candidate, Qt.MatchExactly)
            if found_index != -1 and found_index != self.camerainfo_topic_combo.currentIndex():
                self.camerainfo_topic_combo.blockSignals(True)
                self.camerainfo_topic_combo.setCurrentIndex(found_index)
                self.camerainfo_topic_combo.blockSignals(False)
                self.update_proceed_button_state()
                return

        def prefix_score(candidate: str) -> int:
            image_parts = image_topic.strip("/").split("/")
            camera_parts = candidate.strip("/").split("/")
            score = 0
            for image_part, camera_part in zip(image_parts, camera_parts):
                if image_part == camera_part:
                    score += 2
                    continue
                if "image" in image_part and "camera_info" in camera_part:
                    score += 1
                    continue
                break
            return score

        best_index = None
        best_score = 0
        for candidate_index in range(camera_info_count):
            candidate_topic = self.camerainfo_topic_combo.itemText(candidate_index)
            score = prefix_score(candidate_topic)
            if score > best_score:
                best_score = score
                best_index = candidate_index

        if best_index is not None and best_index != self.camerainfo_topic_combo.currentIndex():
            self.camerainfo_topic_combo.blockSignals(True)
            self.camerainfo_topic_combo.setCurrentIndex(best_index)
            self.camerainfo_topic_combo.blockSignals(False)

        self.update_proceed_button_state()

    def validate_lidar_topic_selection(self):
        if self._is_lidar_cam:
            return
        source_topic = self.pointcloud_topic_combo.currentText()
        target_topic = self.pointcloud2_topic_combo.currentText()
        if source_topic and source_topic == target_topic:
            self.pointcloud2_topic_combo.blockSignals(True)
            for i in range(self.pointcloud2_topic_combo.count()):
                if self.pointcloud2_topic_combo.itemText(i) != source_topic:
                    self.pointcloud2_topic_combo.setCurrentIndex(i)
                    break
            self.pointcloud2_topic_combo.blockSignals(False)
        self.update_proceed_button_state()

    def update_proceed_button_state(self):
        if self._is_lidar_cam:
            is_valid = all(
                [
                    self.image_topic_combo.currentText(),
                    self.pointcloud_topic_combo.currentText(),
                ]
            )
        else:
            is_valid = all(
                [
                    self.pointcloud_topic_combo.currentText(),
                    self.pointcloud2_topic_combo.currentText(),
                    self.pointcloud_topic_combo.currentText()
                    != self.pointcloud2_topic_combo.currentText(),
                ]
            )
        self.proceed_button.setEnabled(is_valid)

    def process_rosbag_data(self):
        """Read and process required data from the rosbag in a worker thread."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.proceed_button.setEnabled(False)
        self.repaint()

        # This dictionary maps all available topic names to their types
        topic_types = {topic: msgtype for topic, msgtype, _ in self.topics}
        tf_topics = [
            topic
            for topic in topic_types
            if "tf" in topic.lower() and "TFMessage" in topic_types[topic]
        ]

        if self._is_lidar_cam:
            selected_topics_data = {
                "calibration_type": self.calibration_type,
                "image_topic": self.image_topic_combo.currentText(),
                "pointcloud_topic": self.pointcloud_topic_combo.currentText(),
                "camerainfo_topic": self.camerainfo_topic_combo.currentText(),
                "tf_topics": tf_topics,
            }
        else:  # LiDAR2LiDAR
            selected_topics_data = {
                "calibration_type": self.calibration_type,
                "pointcloud_topic": self.pointcloud_topic_combo.currentText(),
                "pointcloud2_topic": self.pointcloud2_topic_combo.currentText(),
                "tf_topics": tf_topics,
            }

        # --- THIS IS THE FIX ---
        # Correctly build the topics_to_read dictionary.
        # The KEYS should be the actual topic names (e.g., '/drivers/lidar/points')
        # and the VALUES should be their message types.
        topics_to_read = {}
        for key, topic_name in selected_topics_data.items():
            # Find all keys that represent a user-selected topic
            if key.endswith("_topic") and topic_name:
                # Use the topic_name (the value) as the key for our new dict
                if topic_name in topic_types:
                    topics_to_read[topic_name] = topic_types[topic_name]

        # Add all found TF topics to the list of topics to be read
        for tf_topic in tf_topics:
            if tf_topic in topic_types:
                topics_to_read[tf_topic] = topic_types[tf_topic]

        # The worker receives the correctly structured dictionary
        self.processing_worker = RosbagProcessingWorker(
            bag_file=self.bag_file,
            topics_to_read=topics_to_read,
            selected_topics_data=selected_topics_data,
            total_messages=get_total_message_count(
                self.bag_file, self.ros_version_combo.currentText()
            ),
            frame_samples=self.frame_count_spinbox.value(),
            topic_message_counts={name: count for name, _, count in self.topics},
            ros_version=self.ros_version_combo.currentText(),
            sync_tolerance=0.05,
        )
        self.processing_worker.progress_updated.connect(self.update_processing_progress)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        self.processing_worker.processing_failed.connect(self.on_processing_failed)
        self.processing_worker.start()

    def update_processing_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(message)

    def on_processing_finished(
        self, raw_messages: Dict, topic_types: Dict, selected_topics_data: Dict
    ):
        """
        Handles completion of rosbag processing. Correctly routes to frame selection
        for LiDAR2Cam or directly to transform selection for LiDAR2LiDAR.
        This method is now robust against different return formats from the worker.
        """
        print("[DEBUG] Rosbag processing completed successfully.")
        self.progress_bar.setVisible(False)
        self.proceed_button.setEnabled(True)

        self.topic_types = topic_types
        self.selected_topics_data = selected_topics_data

        self.tf_messages = {
            topic: raw_messages.get(topic)
            for topic in selected_topics_data.get("tf_topics", [])
            if topic in raw_messages
        }

        # Check if the worker returned multiple frame samples or a single synchronized message set.
        if "frame_samples" in raw_messages and raw_messages["frame_samples"]:
            print("[DEBUG] Processing multi-frame samples.")
            self.frame_samples = raw_messages["frame_samples"]

            if self._is_lidar_cam:
                pointcloud_topic = selected_topics_data["pointcloud_topic"]
                camerainfo_topic = selected_topics_data.get("camerainfo_topic", "")

                self.lidar_frame = self.extract_frame_id(
                    self.frame_samples[pointcloud_topic][0]["data"]
                )
                if camerainfo_topic and camerainfo_topic in self.frame_samples:
                    self.camera_frame = self.extract_frame_id(
                        self.frame_samples[camerainfo_topic][0]["data"]
                    )
                else:
                    self.camera_frame = self._child_frame

                self.frame_selection_widget.set_frame_samples(
                    self.frame_samples, selected_topics_data["image_topic"]
                )
                self._open_intrinsics_view()

            else:  # LiDAR2LiDAR multi-frame (take the first)
                pointcloud_topic = selected_topics_data["pointcloud_topic"]
                pointcloud2_topic = selected_topics_data["pointcloud2_topic"]

                if not self.frame_samples.get(pointcloud_topic) or not self.frame_samples.get(
                    pointcloud2_topic
                ):
                    self.on_processing_failed(
                        "No synchronized message pairs found for LiDAR topics."
                    )
                    return

                first_pc1 = self.frame_samples[pointcloud_topic][0]["data"]
                first_pc2 = self.frame_samples[pointcloud2_topic][0]["data"]
                self.prepare_for_transform_view(
                    topic_types, selected_topics_data, first_pc1, first_pc2
                )

        else:
            # Fallback for workers that return a single message set (flat dictionary).
            print("[DEBUG] Processing single message set (no 'frame_samples' key found).")
            if not self._is_lidar_cam:
                pointcloud_topic = selected_topics_data["pointcloud_topic"]
                pointcloud2_topic = selected_topics_data["pointcloud2_topic"]

                pc1_msg = raw_messages.get(pointcloud_topic)
                pc2_msg = raw_messages.get(pointcloud2_topic)

                if not pc1_msg or not pc2_msg:
                    self.on_processing_failed("Synchronized LiDAR messages not found in bag.")
                    return
                self.prepare_for_transform_view(topic_types, selected_topics_data, pc1_msg, pc2_msg)
            else:  # Fallback for LiDAR2Cam single frame
                image_topic = selected_topics_data["image_topic"]
                pointcloud_topic = selected_topics_data["pointcloud_topic"]
                camerainfo_topic = selected_topics_data.get("camerainfo_topic", "")

                self.lidar_frame = self.extract_frame_id(raw_messages[pointcloud_topic])
                if camerainfo_topic and camerainfo_topic in raw_messages:
                    self.camera_frame = self.extract_frame_id(raw_messages[camerainfo_topic])
                else:
                    self.camera_frame = self._child_frame

                self.selected_topics = {
                    "image_topic": image_topic,
                    "pointcloud_topic": pointcloud_topic,
                    "camerainfo_topic": camerainfo_topic,
                    "topic_types": topic_types,
                    "raw_messages": {
                        topic: raw_messages.get(topic)
                        for topic in [image_topic, pointcloud_topic, camerainfo_topic]
                        if topic
                    },
                    "tf_messages": self.tf_messages,
                }
                self._open_intrinsics_view()

        if hasattr(self, "processing_worker") and self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

    def prepare_for_transform_view(self, topic_types, selected_topics_data, pc1_msg, pc2_msg):
        """Helper function to set up data structures and switch to the transform view for LiDAR2LiDAR."""
        pointcloud_topic = selected_topics_data["pointcloud_topic"]
        pointcloud2_topic = selected_topics_data["pointcloud2_topic"]

        self.lidar_frame = self.extract_frame_id(pc1_msg)
        self.lidar2_frame = self.extract_frame_id(pc2_msg)

        self.selected_topics = {
            "calibration_type": "LiDAR2LiDAR",
            "pointcloud_topic": pointcloud_topic,
            "pointcloud2_topic": pointcloud2_topic,
            "topic_types": topic_types,
            "raw_messages": {
                pointcloud_topic: pc1_msg,
                pointcloud2_topic: pc2_msg,
            },
            "tf_messages": self.tf_messages,
        }
        self.tf_title_label.setText(
            f"Select Initial Transformation: {self.lidar_frame} → {self.lidar2_frame}"
        )
        self.load_tf_topics_in_transform_view()
        self.stacked_widget.setCurrentIndex(3)  # Switch to Transform View

    def on_frame_selected(self, frame_index: int):
        print(f"[DEBUG] Frame {frame_index + 1} selected for LiDAR2Cam calibration.")
        image_topic = self.selected_topics_data["image_topic"]
        pointcloud_topic = self.selected_topics_data["pointcloud_topic"]
        camerainfo_topic = self.selected_topics_data["camerainfo_topic"]
        self.selected_topics = {
            "image_topic": image_topic,
            "pointcloud_topic": pointcloud_topic,
            "camerainfo_topic": camerainfo_topic,
            "topic_types": self.topic_types,
            "raw_messages": {
                image_topic: self.frame_samples[image_topic][frame_index]["data"],
                pointcloud_topic: self.frame_samples[pointcloud_topic][frame_index]["data"],
                camerainfo_topic: self.frame_samples[camerainfo_topic][frame_index]["data"],
            },
            "tf_messages": self.tf_messages,
        }
        self.tf_title_label.setText(
            f"Initial Transform: {self._parent_frame} → {self._child_frame} (deg)"
        )
        self.load_tf_topics_in_transform_view()
        self.stacked_widget.setCurrentIndex(3)

    def on_frames_accumulated(self, image_frame_index: int):
        """Merge all LiDAR frames into one dense cloud and proceed with the selected image frame."""
        print(f"[DEBUG] Accumulating all LiDAR frames, using image frame {image_frame_index + 1}.")
        image_topic = self.selected_topics_data["image_topic"]
        pointcloud_topic = self.selected_topics_data["pointcloud_topic"]
        camerainfo_topic = self.selected_topics_data["camerainfo_topic"]

        pc_frames = self.frame_samples[pointcloud_topic]
        first_raw = pc_frames[0]["data"]

        merged_data = b"".join(bytes(f["data"].data) for f in pc_frames)
        merged_width = sum(f["data"].width * f["data"].height for f in pc_frames)

        merged_pc = ros_utils.PointCloud2(
            header=first_raw.header,
            height=1,
            width=merged_width,
            fields=[
                ros_utils.PointField(
                    name=f.name, offset=f.offset, datatype=f.datatype, count=f.count
                )
                for f in first_raw.fields
            ],
            is_bigendian=first_raw.is_bigendian,
            point_step=first_raw.point_step,
            row_step=first_raw.point_step * merged_width,
            data=merged_data,
            is_dense=first_raw.is_dense,
        )

        print(f"[DEBUG] Merged {len(pc_frames)} frames → {merged_width} points total.")

        self.selected_topics = {
            "image_topic": image_topic,
            "pointcloud_topic": pointcloud_topic,
            "camerainfo_topic": camerainfo_topic,
            "topic_types": self.topic_types,
            "raw_messages": {
                image_topic: self.frame_samples[image_topic][image_frame_index]["data"],
                pointcloud_topic: merged_pc,
                camerainfo_topic: self.frame_samples[camerainfo_topic][image_frame_index]["data"],
            },
            "tf_messages": self.tf_messages,
        }
        self.tf_title_label.setText(
            f"Initial Transform: {self._parent_frame} → {self._child_frame} (deg)"
        )
        self.load_tf_topics_in_transform_view()
        self.stacked_widget.setCurrentIndex(3)

    def on_processing_failed(self, error_message):
        print(f"[ERROR] Rosbag processing failed: {error_message}")
        self.progress_bar.setFormat(f"Error: {error_message}")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        self.proceed_button.setEnabled(True)
        if hasattr(self, "processing_worker") and self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

    def extract_frame_id(self, msg):
        return getattr(getattr(msg, "header", None), "frame_id", "unknown_frame")

    def proceed_to_calibration(self, initial_transform):
        for i in range(self.stacked_widget.count() - 1, 5, -1):
            widget = self.stacked_widget.widget(i)
            self.stacked_widget.removeWidget(widget)
            widget.deleteLater()

        if not self._is_lidar_cam:
            self.proceed_to_lidar_calibration_with_transform(initial_transform)
            return

        image_msg = convert_to_mock(
            self.selected_topics["raw_messages"][self.selected_topics["image_topic"]],
            self.selected_topics["topic_types"][self.selected_topics["image_topic"]],
        )
        pointcloud_msg = convert_to_mock(
            self.selected_topics["raw_messages"][self.selected_topics["pointcloud_topic"]],
            self.selected_topics["topic_types"][self.selected_topics["pointcloud_topic"]],
        )
        camerainfo_msg = self.active_camerainfo_msg or convert_to_mock(
            self.selected_topics["raw_messages"][self.selected_topics["camerainfo_topic"]],
            self.selected_topics["topic_types"][self.selected_topics["camerainfo_topic"]],
        )
        self.calibration_widget = CalibrationWidget(
            image_msg, pointcloud_msg, camerainfo_msg, ros_utils, initial_transform
        )
        self.calibration_widget.calibration_completed.connect(self.show_calibration_results)
        self.stacked_widget.setCurrentIndex(self.stacked_widget.addWidget(self.calibration_widget))

    def proceed_to_lidar_calibration_with_transform(self, initial_transform):
        raw_msgs = self.selected_topics["raw_messages"]
        topic_types = self.selected_topics["topic_types"]
        pc_topic1 = self.selected_topics["pointcloud_topic"]
        pc_topic2 = self.selected_topics["pointcloud2_topic"]

        pointcloud_msg1 = convert_to_mock(raw_msgs[pc_topic1], topic_types[pc_topic1])
        pointcloud_msg2 = convert_to_mock(raw_msgs[pc_topic2], topic_types[pc_topic2])

        import threading

        threading.Thread(
            target=self._run_lidar_calibration_thread,
            args=(pointcloud_msg1, pointcloud_msg2, initial_transform),
            daemon=True,
        ).start()

    def _run_lidar_calibration_thread(self, pc1, pc2, initial_transform):
        try:
            launch_lidar2lidar_calibration(
                pc1, pc2, initial_transform, self._on_lidar_calibration_completed
            )
        except Exception as e:
            print(f"[ERROR] LiDAR calibration thread failed: {e}")

    def _on_lidar_calibration_completed(self, final_transform: np.ndarray):
        print("[DEBUG] LiDAR calibration completed via callback.")
        self.calibration_completed.emit(final_transform)

    def go_back_to_calibration(self):
        if not self._is_lidar_cam:
            self.restart_lidar_calibration()
            return
        for i in range(self.stacked_widget.count()):
            if isinstance(self.stacked_widget.widget(i), CalibrationWidget):
                self.stacked_widget.setCurrentIndex(i)
                return

    def get_results_view_index(self):
        return 5

    def restart_lidar_calibration(self):
        raw_msgs = self.selected_topics.get("raw_messages")
        topic_types = self.selected_topics.get("topic_types")
        pc_topic1 = self.selected_topics.get("pointcloud_topic")
        pc_topic2 = self.selected_topics.get("pointcloud2_topic")

        if not (raw_msgs and topic_types and pc_topic1 and pc_topic2):
            print("[WARN] Missing LiDAR topics or raw messages; cannot restart calibration.")
            return

        if isinstance(self.calibrated_transform, np.ndarray):
            initial_transform = np.array(self.calibrated_transform, copy=True)
        else:
            initial_transform = np.eye(4, dtype=np.float64)

        self.proceed_to_lidar_calibration_with_transform(initial_transform)

    def load_tf_topics_in_transform_view(self):
        self.tf_topic_combo.clear()
        tf_messages = self.selected_topics.get("tf_messages", self.tf_messages)
        if tf_messages:
            tf_topics = list(tf_messages.keys())
            self.tf_topic_combo.addItems(tf_topics)
            tf_static = next(
                (t for t in tf_topics if "tf_static" in t), tf_topics[0] if tf_topics else None
            )
            if tf_static:
                self.tf_topic_combo.setCurrentText(tf_static)
                self.load_tf_tree_from_preloaded()
        else:
            self.tf_topic_combo.addItem("No TF topics found")
            self.load_tf_button.setEnabled(False)

    def on_tf_topic_changed(self):
        self.tf_tree = {}
        self.tf_info_text.setPlainText("Select 'Load TF Tree' to load transformations.")
        self.show_graph_button.setEnabled(False)

    def load_tf_tree(self):
        self.load_tf_tree_from_preloaded()

    def load_tf_tree_from_preloaded(self):
        topic_name = self.tf_topic_combo.currentText()
        tf_messages = self.selected_topics.get("tf_messages", self.tf_messages)
        if not topic_name or topic_name not in tf_messages:
            return

        self.tf_tree = self.parse_preloaded_tf_message(tf_messages[topic_name])
        self.update_tf_info_display()
        self.try_find_transform()
        self.show_graph_button.setEnabled(bool(self.tf_tree))

    def parse_preloaded_tf_message(self, msg_data) -> Dict[str, Dict]:
        tf_tree = {}
        for transform_stamped in self.deserialize_tf_message(msg_data).transforms:
            parent = transform_stamped.header.frame_id
            child = transform_stamped.child_frame_id
            tf_tree.setdefault(parent, {})[child] = {
                "transform": ros_utils.transform_to_numpy(transform_stamped.transform)
            }
        return tf_tree

    def deserialize_tf_message(self, msg_data) -> ros_utils.TFMessage:
        """
        Convert raw rosbag TF message data to our internal mock TFMessage format.
        This version is corrected to handle metadata fields from the rosbags library.
        """
        if not hasattr(msg_data, "transforms"):
            return ros_utils.TFMessage(transforms=[])

        transforms = []
        for transform_msg in msg_data.transforms:
            # Get the translation and rotation objects
            translation_obj = transform_msg.transform.translation
            rotation_obj = transform_msg.transform.rotation

            # Instead of using **vars(), we explicitly access the x, y, z, w attributes.
            # This is safer and ignores any extra metadata like '__msgtype__'.
            new_transform = ros_utils.Transform(
                translation=ros_utils.Vector3(
                    x=translation_obj.x, y=translation_obj.y, z=translation_obj.z
                ),
                rotation=ros_utils.Quaternion(
                    x=rotation_obj.x, y=rotation_obj.y, z=rotation_obj.z, w=rotation_obj.w
                ),
            )

            new_stamped = ros_utils.TransformStamped(
                header=ros_utils.Header(frame_id=transform_msg.header.frame_id),
                child_frame_id=transform_msg.child_frame_id,
                transform=new_transform,
            )
            transforms.append(new_stamped)

        return ros_utils.TFMessage(transforms=transforms)

    def try_find_transform(self):
        if not self.tf_tree:
            return
        source = self.lidar_frame
        target = self.camera_frame if self._is_lidar_cam else self.lidar2_frame
        if (transform_matrix := self.find_transform_path(source, target)) is not None:
            self.current_transform = transform_matrix
            self.update_transform_display()
            self.update_manual_inputs_from_matrix()

    def find_transform_path(self, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        if from_frame == to_frame:
            return np.eye(4)
        if not self.tf_tree:
            return None

        from collections import deque

        q = deque([(from_frame, np.eye(4))])
        visited = {from_frame}

        adj = {frame: [] for frame in self._get_all_tf_frames()}
        for p, children in self.tf_tree.items():
            for c, data in children.items():
                adj[p].append((c, data["transform"]))
                adj[c].append((p, np.linalg.inv(data["transform"])))

        while q:
            curr_frame, T = q.popleft()
            if curr_frame == to_frame:
                return T
            for neighbor, t in adj.get(curr_frame, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, T @ t))
        return None

    def find_transformation_path_frames(
        self, from_frame: str, to_frame: str
    ) -> Optional[List[str]]:
        if from_frame == to_frame:
            return [from_frame]
        if not self.tf_tree:
            return None

        from collections import deque

        q = deque([(from_frame, [from_frame])])
        visited = {from_frame}

        adj = {frame: [] for frame in self._get_all_tf_frames()}
        for p, children in self.tf_tree.items():
            for c in children:
                adj[p].append(c)
                adj[c].append(p)

        while q:
            curr_frame, path = q.popleft()
            if curr_frame == to_frame:
                return path
            for neighbor in adj.get(curr_frame, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append((neighbor, path + [neighbor]))
        return None

    def update_tf_info_display(self):
        if not self.tf_tree:
            self.tf_info_text.setPlainText("No TF data available.")
            return
        source = self.lidar_frame
        target = self.camera_frame if self._is_lidar_cam else self.lidar2_frame
        path_frames = self.find_transformation_path_frames(source, target)
        path_info = (
            f"\n✓ Found transformation path: {' → '.join(path_frames)}"
            if path_frames
            else f"\n✗ No transformation found: {source} → {target}"
        )
        self.tf_info_text.setPlainText(
            f"Available transformations: {sum(len(c) for c in self.tf_tree.values())}{path_info}"
        )

    def show_tf_graph(self):
        if not self.tf_tree:
            return
        source = self.lidar_frame
        target = self.camera_frame if self._is_lidar_cam else self.lidar2_frame
        path_frames = self.find_transformation_path_frames(source, target)
        self.tf_graph_window = TFGraphWidget(self.tf_tree, source, target, path_frames, parent=self)
        self.tf_graph_window.show()

    def update_manual_transform(self):
        try:
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_euler(
                "xyz",
                [
                    float(self.rx_input.text()),
                    float(self.ry_input.text()),
                    float(self.rz_input.text()),
                ],
                degrees=True,
            )
            self.current_transform = np.eye(4, dtype=np.float64)
            self.current_transform[:3, :3] = rot.as_matrix()
            self.current_transform[:3, 3] = [
                float(self.tx_input.text()),
                float(self.ty_input.text()),
                float(self.tz_input.text()),
            ]
            self.update_transform_display()
        except ValueError as e:
            print(f"[ERROR] Invalid manual transform input: {e}")

    def update_manual_inputs_from_matrix(self):
        trans = tf.translation_from_matrix(self.current_transform)
        euler = tf.euler_from_matrix(self.current_transform)
        self.tx_input.setText(f"{trans[0]:.6f}")
        self.ty_input.setText(f"{trans[1]:.6f}")
        self.tz_input.setText(f"{trans[2]:.6f}")
        self.rx_input.setText(f"{np.degrees(euler[0]):.4f}")
        self.ry_input.setText(f"{np.degrees(euler[1]):.4f}")
        self.rz_input.setText(f"{np.degrees(euler[2]):.4f}")

    def use_identity_transform(self):
        self.current_transform = np.eye(4, dtype=np.float64)
        self.update_transform_display()
        self.update_manual_inputs_from_matrix()

    def update_transform_display(self):
        matrix_str = "\n".join(
            ["  ".join([f"{val:8.4f}" for val in row]) for row in self.current_transform]
        )
        trans = tf.translation_from_matrix(self.current_transform)
        rpy = tf.euler_from_matrix(self.current_transform)

        combined_display = f"Transformation Matrix:\n{matrix_str}\n\nTranslation and Rotation:\nXYZ: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]\nRPY: [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}]"
        self.transform_display.setPlainText(combined_display)

    def confirm_transformation(self):
        self.proceed_to_calibration(self.current_transform)

    def show_calibration_results(self, calibration_results):
        print("[DEBUG] show_calibration_results called on main thread.")
        print(f"[DEBUG] Received calibration_results: {calibration_results}")
        print(f"[DEBUG] Results type: {type(calibration_results)}")

        # Handle both old format (direct numpy array) and new format (dictionary)
        if isinstance(calibration_results, dict):
            calibrated_transform = calibration_results["master_to_camera"]
        elif isinstance(calibration_results, np.ndarray):
            calibrated_transform = calibration_results
        else:
            print(f"[ERROR] Unexpected calibration result type: {type(calibration_results)}")
            return

        self.calibrated_transform = calibrated_transform
        if self._is_lidar_cam:
            self.calibrated_transform = np.linalg.inv(calibrated_transform)
            self.original_source_frame = self._parent_frame
            self.original_target_frame = self._child_frame
        else:
            self.original_source_frame = self.lidar_frame
            self.original_target_frame = self.lidar2_frame
        self.populate_results_view()
        self.stacked_widget.setCurrentIndex(self.get_results_view_index())

    def _get_all_tf_frames(self) -> List[str]:
        frames = set(self.tf_tree.keys())
        for children in self.tf_tree.values():
            frames.update(children.keys())
        return sorted(list(frames))

    def populate_results_view(self):
        self.display_transform_urdf(
            self.calibration_result_display,
            self.calibrated_transform,
            self.original_source_frame,
            self.original_target_frame,
        )
        all_frames = self._get_all_tf_frames()
        self.source_frame_combo.blockSignals(True)
        self.target_frame_combo.blockSignals(True)
        self.source_frame_combo.clear()
        self.source_frame_combo.addItems(all_frames)
        self.target_frame_combo.clear()
        self.target_frame_combo.addItems(all_frames)
        self.source_frame_combo.setCurrentText(self.original_source_frame)
        self.target_frame_combo.setCurrentText(self.original_target_frame)
        self.source_frame_combo.blockSignals(False)
        self.target_frame_combo.blockSignals(False)
        self.update_target_transform()

    def update_target_transform(self):
        new_source = self.source_frame_combo.currentText()
        new_target = self.target_frame_combo.currentText()
        if not new_source or not new_target:
            return
        path_frames = self.find_transformation_path_frames(new_source, new_target)
        self.chain_display.setPlainText(
            " → ".join(path_frames)
            if path_frames
            else f"No static path found between {new_source} and {new_target}."
        )
        self.update_embedded_graph(new_source, new_target)

        T_orig_src_to_new_src = self.find_transform_path(self.original_source_frame, new_source)
        T_orig_tgt_to_new_tgt = self.find_transform_path(self.original_target_frame, new_target)

        if T_orig_src_to_new_src is None or T_orig_tgt_to_new_tgt is None:
            self.final_transform_display.setPlainText(
                "Error: Cannot find path from original frames in TF tree."
            )
            return

        T_new_src_to_orig_src = np.linalg.inv(T_orig_src_to_new_src)
        final_transform = T_new_src_to_orig_src @ self.calibrated_transform @ T_orig_tgt_to_new_tgt
        self.display_transform_urdf(
            self.final_transform_display, final_transform, new_source, new_target
        )
        self.current_final_transform = final_transform

    def display_transform_urdf(
        self, text_widget: QTextEdit, transform: np.ndarray, parent: str, child: str
    ):
        from . import tf_transformations as tf

        if parent == child:
            text_widget.setPlainText("Source and target frames cannot be the same.")
            return
        matrix_str = "\n".join(["  ".join([f"{val:8.4f}" for val in row]) for row in transform])
        xyz = tf.translation_from_matrix(transform)
        rpy = tf.euler_from_matrix(transform)
        joint_name = f"joint_{parent.replace('/', '_')}_to_{child.replace('/', '_')}"
        urdf = f'<joint name="{joint_name}" type="fixed">\n  <parent link="{parent}" />\n  <child link="{child}" />\n  <origin xyz="{" ".join(map(str, xyz))}" rpy="{" ".join(map(str, rpy))}" />\n</joint>'
        text_widget.setPlainText(
            f"Transform: {parent} → {child}\nMatrix:\n{matrix_str}\n\nURDF Snippet:\n{urdf}"
        )

    def update_embedded_graph(self, source_frame: str, target_frame: str):
        self.init_graph_placeholder()
        if not self.tf_tree:
            return
        try:
            path_frames = self.find_transformation_path_frames(source_frame, target_frame)
            graph_widget = TFGraphWidget(self.tf_tree, source_frame, target_frame, path_frames)
            self.graph_container.layout().addWidget(graph_widget)
        except Exception as e:
            print(f"[ERROR] Failed to create or embed TF graph: {e}")

    def export_calibration_result(self):
        if self._is_lidar_cam:
            default_name = f"livox_to_{self._camera_type_label}.yaml"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Calibration", default_name, "YAML Files (*.yaml)"
            )
            if file_path:
                self._write_calib_yaml(file_path)
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Target Transform", "", "Text Files (*.txt);;All Files (*)"
            )
            if file_path:
                with open(file_path, "w") as f:
                    f.write(self.final_transform_display.toPlainText())

    def _write_calib_yaml(self, file_path: str) -> None:
        from scipy.spatial.transform import Rotation

        T = self.current_final_transform
        t = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]

        parent = self._parent_frame
        child = self._child_frame
        key = f"livox_to_{self._camera_type_label}"
        timestamp = datetime.date.today().isoformat()
        description = f"LiDAR to {self._camera_type_label.capitalize()} Camera calibration"

        content = (
            f"{key}:\n"
            f"  parent_frame: {parent}\n"
            f"  child_frame: {child}\n"
            f"  translation:\n"
            f"    x: {t[0]:.4f}\n"
            f"    y: {t[1]:.4f}\n"
            f"    z: {t[2]:.4f}\n"
            f"  rotation:\n"
            f"    x: {q[0]:.4f}\n"
            f"    y: {q[1]:.4f}\n"
            f"    z: {q[2]:.4f}\n"
            f"    w: {q[3]:.4f}\n"
            f"  metadata:\n"
            f"    source_file: ros2_calib\n"
            f"    conversion_timestamp: '{timestamp}'\n"
            f"    description: {description}\n"
        )
        with open(file_path, "w") as f:
            f.write(content)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            self.process_dropped_path(event.mimeData().urls()[0].toLocalFile())
            event.acceptProposedAction()
