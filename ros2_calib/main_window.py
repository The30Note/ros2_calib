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
from scipy.spatial.transform import Rotation
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QDoubleSpinBox,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
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
# NOTE: lidar2lidar_o3d_widget imports open3d, whose PyPI wheels are built with
# AVX2/FMA. On CPUs without AVX (e.g. Apollo Lake Celeron J3455), importing
# open3d raises SIGILL ("Illegal instruction"). Import it lazily inside the
# LiDAR-to-LiDAR launch path so the rest of the app (LiDAR-to-Camera) still runs
# on such machines.
from .tf_graph_widget import TFGraphWidget

MAX_RENDER_POINTS = 500_000
VOXEL_SIZE_INIT   = 0.01    # metres — default starting voxel size for density equalisation


class MainWindow(QMainWindow):
    calibration_completed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ros2_calib — Multi-Sensor Calibration")
        self.setGeometry(100, 100, 1920, 1080)
        self.setAcceptDrops(True)

        # ── State ──────────────────────────────────────────────────────
        self.bag_file: Optional[str] = None
        self.topics: list = []
        self.topic_types: Dict[str, str] = {}
        self.calibration_type = "LiDAR2Cam_Context"
        self.current_transform = np.eye(4, dtype=np.float64)
        self.active_camerainfo_msg = None
        self.frame_samples: Dict = {}
        self.tf_messages: Dict = {}
        self.tf_tree: Dict = {}
        self.lidar_frame = "livox_frame"
        self.camera_frame = "context_camera_frame"
        self.calib_widget: Optional[CalibrationWidget] = None
        self.tf_graph_window = None
        self.selected_topics_data: Dict = {}

        self.calibration_completed.connect(self._on_calibration_result)

        # ── Layout ─────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        root.addWidget(self._build_topbar())

        # ── Vertical splitter: middle config row | bottom view ─────────
        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.setChildrenCollapsible(False)
        self.v_splitter.setHandleWidth(6)

        self.middle_row = self._build_middle_row()
        self.middle_row.setEnabled(False)
        self.middle_row.setMinimumHeight(140)
        self.v_splitter.addWidget(self.middle_row)

        # ── Horizontal splitter: view | side panel ─────────────────────
        self.h_splitter = QSplitter(Qt.Horizontal)
        self.h_splitter.setChildrenCollapsible(False)
        self.h_splitter.setHandleWidth(6)

        self.view_container = QWidget()
        self.view_container.setStyleSheet("background: #141414; border-radius: 4px;")
        self._vcl = QVBoxLayout(self.view_container)
        self._vcl.setContentsMargins(0, 0, 0, 0)
        self._view_placeholder = QLabel("← Load a rosbag to begin calibration")
        self._view_placeholder.setAlignment(Qt.AlignCenter)
        self._view_placeholder.setStyleSheet("color: #444; font-size: 16px;")
        self._vcl.addWidget(self._view_placeholder)
        self.h_splitter.addWidget(self.view_container)

        sp_outer = QWidget()
        sp_outer.setMinimumWidth(200)
        spl = QVBoxLayout(sp_outer)
        spl.setContentsMargins(0, 0, 0, 0)
        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _sph = QLabel("Controls appear after processing")
        _sph.setAlignment(Qt.AlignCenter)
        _sph.setWordWrap(True)
        _sph.setStyleSheet("color: #444; padding: 20px;")
        self.side_scroll.setWidget(_sph)
        spl.addWidget(self.side_scroll)
        self.h_splitter.addWidget(sp_outer)

        # Initial sizes: view 75%, side panel 25%
        self.h_splitter.setStretchFactor(0, 3)
        self.h_splitter.setStretchFactor(1, 1)

        self.v_splitter.addWidget(self.h_splitter)

        # Initial sizes: middle ~190px, rest goes to bottom
        self.v_splitter.setStretchFactor(0, 0)
        self.v_splitter.setStretchFactor(1, 1)
        self.v_splitter.setSizes([190, 700])

        root.addWidget(self.v_splitter, stretch=1)

    # ================================================================== #
    #  Properties                                                          #
    # ================================================================== #

    @property
    def _is_lidar_cam(self) -> bool:
        return self.calibration_type in ("LiDAR2Cam_Context", "LiDAR2Cam_Zoom")

    @property
    def _camera_type_label(self) -> str:
        if self.calibration_type == "LiDAR2Cam_Context":
            return "context"
        if self.calibration_type == "LiDAR2Cam_Zoom":
            return "zoom"
        return ""

    @property
    def _child_frame(self) -> str:
        if self.calibration_type == "LiDAR2Cam_Context":
            return "context_camera_frame"
        if self.calibration_type == "LiDAR2Cam_Zoom":
            return "zoom_camera_frame"
        return getattr(self, "camera_frame", "")

    @property
    def _parent_frame(self) -> str:
        return "livox_frame"

    # ================================================================== #
    #  Layout builders                                                     #
    # ================================================================== #

    def _build_topbar(self) -> QFrame:
        bar = QFrame()
        bar.setFrameShape(QFrame.StyledPanel)
        bar.setMaximumHeight(130)
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # ── Row 1: bag load + sub-mode + main mode ─────────────────────
        row1 = QHBoxLayout()

        self.load_bag_button = QPushButton("📂  Load Rosbag")
        self.load_bag_button.setFixedHeight(36)
        self.load_bag_button.setMinimumWidth(150)
        self.load_bag_button.clicked.connect(self.load_bag)
        row1.addWidget(self.load_bag_button)

        self.bag_path_label = QLabel("No rosbag loaded  (drag & drop supported)")
        self.bag_path_label.setStyleSheet("color: #666; font-style: italic; padding: 0 8px;")
        self.bag_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row1.addWidget(self.bag_path_label, 1)

        row1.addSpacing(16)

        # Context / Zoom sub-toggle (cam2lidar only)
        self._sub_mode_frame = QFrame()
        sub_row = QHBoxLayout(self._sub_mode_frame)
        sub_row.setContentsMargins(0, 0, 0, 0)
        sub_row.setSpacing(0)

        _sub_style_active = (
            "QPushButton { background: #d64814; color: white; border-radius: 0; "
            "padding: 6px 18px; font-weight: bold; }"
        )
        _sub_style_inactive = (
            "QPushButton { background: #3a3a3a; color: #888; border-radius: 0; "
            "padding: 6px 18px; }"
            "QPushButton:hover { background: #484848; color: #bbb; }"
        )

        self.btn_context = QPushButton("Context")
        self.btn_context.setFixedHeight(36)
        self.btn_context.setStyleSheet(_sub_style_active)
        self.btn_context.clicked.connect(lambda: self._on_sub_mode_clicked("context"))

        self.btn_zoom = QPushButton("Zoom")
        self.btn_zoom.setFixedHeight(36)
        self.btn_zoom.setStyleSheet(_sub_style_inactive)
        self.btn_zoom.clicked.connect(lambda: self._on_sub_mode_clicked("zoom"))

        sub_row.addWidget(self.btn_context)
        sub_row.addWidget(self.btn_zoom)
        row1.addWidget(self._sub_mode_frame)

        row1.addStretch()

        layout.addLayout(row1)

        # ── Row 2: topic selection ─────────────────────────────────────
        row2 = QHBoxLayout()

        self.ros_version_combo = QComboBox()
        self.ros_version_combo.addItems(["HUMBLE", "JAZZY"])
        self.ros_version_combo.setFixedWidth(90)
        row2.addWidget(QLabel("ROS:"))
        row2.addWidget(self.ros_version_combo)
        row2.addSpacing(8)

        self._image_label = QLabel("Image:")
        self.image_topic_combo = QComboBox()
        self.image_topic_combo.setEditable(True)
        self.image_topic_combo.setMinimumWidth(180)
        self.image_topic_combo.currentTextChanged.connect(self._on_topic_changed)
        row2.addWidget(self._image_label)
        row2.addWidget(self.image_topic_combo, 1)

        self._pc_label = QLabel("PointCloud:")
        self.pointcloud_topic_combo = QComboBox()
        self.pointcloud_topic_combo.setEditable(True)
        self.pointcloud_topic_combo.setMinimumWidth(180)
        self.pointcloud_topic_combo.currentTextChanged.connect(self._on_topic_changed)
        row2.addWidget(self._pc_label)
        row2.addWidget(self.pointcloud_topic_combo, 1)

        self._info_label = QLabel("CamInfo:")
        self.camerainfo_topic_combo = QComboBox()
        self.camerainfo_topic_combo.setEditable(True)
        self.camerainfo_topic_combo.setMinimumWidth(180)
        self.camerainfo_topic_combo.currentTextChanged.connect(self._on_topic_changed)
        row2.addWidget(self._info_label)
        row2.addWidget(self.camerainfo_topic_combo, 1)

        self._frames_label = QLabel("Frames:")
        self.frame_samples_spinbox = QSpinBox()
        self.frame_samples_spinbox.setRange(1, 9999)
        self.frame_samples_spinbox.setValue(12)
        self.frame_samples_spinbox.setToolTip(
            "Max number of synchronised LiDAR frames to read from the bag.\n"
            "Frames are uniformly sampled; use a higher count for a denser\n"
            "accumulated cloud, 1 to load just a single frame."
        )
        row2.addWidget(self._frames_label)
        row2.addWidget(self.frame_samples_spinbox)

        self._max_points_label = QLabel("Max Points:")
        self.max_points_spinbox = QSpinBox()
        self.max_points_spinbox.setRange(1_000, 50_000_000)
        self.max_points_spinbox.setSingleStep(50_000)
        self.max_points_spinbox.setGroupSeparatorShown(True)
        self.max_points_spinbox.setValue(MAX_RENDER_POINTS)
        self.max_points_spinbox.setToolTip(
            "Voxel-downsampling cap on the accumulated cloud.\n"
            "Voxel size grows until the merged cloud fits within this many points.\n"
            "Lower for faster rendering, higher to keep more detail."
        )
        row2.addWidget(self._max_points_label)
        row2.addWidget(self.max_points_spinbox)

        self._voxel_label = QLabel("Voxel (m):")
        self.voxel_size_spinbox = QDoubleSpinBox()
        self.voxel_size_spinbox.setRange(0.001, 1.0)
        self.voxel_size_spinbox.setSingleStep(0.005)
        self.voxel_size_spinbox.setDecimals(3)
        self.voxel_size_spinbox.setValue(VOXEL_SIZE_INIT)
        self.voxel_size_spinbox.setToolTip(
            "Starting voxel cell size for the accumulated cloud.\n"
            "Smaller keeps more detail (denser at distance); the voxel only grows\n"
            "from here if needed to stay under Max Points."
        )
        row2.addWidget(self._voxel_label)
        row2.addWidget(self.voxel_size_spinbox)

        self.process_button = QPushButton("▶  Process Bag")
        self.process_button.setFixedHeight(30)
        self.process_button.setEnabled(False)
        self.process_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        self.process_button.clicked.connect(self.process_rosbag_data)
        row2.addWidget(self.process_button)

        self.process_progress = QProgressBar()
        self.process_progress.setVisible(False)
        self.process_progress.setFixedWidth(140)
        row2.addWidget(self.process_progress)

        layout.addLayout(row2)

        # Initial visibility for L2L vs cam2lidar
        self._update_topic_visibility()
        return bar

    def _build_middle_row(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setMinimumHeight(140)
        row = QHBoxLayout(frame)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self._build_intrinsics_panel(), 1)
        row.addWidget(self._build_extrinsics_panel(), 1)
        return frame

    def _build_intrinsics_panel(self) -> QFrame:
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        title = QLabel("Camera Intrinsics")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer.addWidget(title)

        split_row = QHBoxLayout()
        split_row.setSpacing(8)
        outer.addLayout(split_row, stretch=1)

        # Left: source buttons + label
        left = QVBoxLayout()
        left.setSpacing(4)

        src_row = QHBoxLayout()
        self.intrinsics_default_btn = QPushButton("Default File")
        self.intrinsics_default_btn.clicked.connect(self._load_default_intrinsics)
        self.intrinsics_file_btn = QPushButton("Load File…")
        self.intrinsics_file_btn.clicked.connect(self._browse_intrinsics_file)
        self.intrinsics_rosbag_btn = QPushButton("ROS Topic")
        self.intrinsics_rosbag_btn.clicked.connect(self._use_rosbag_intrinsics)
        for b in (self.intrinsics_default_btn, self.intrinsics_file_btn,
                  self.intrinsics_rosbag_btn):
            src_row.addWidget(b)
        src_row.addStretch()
        left.addLayout(src_row)

        self.intrinsics_source_label = QLabel()
        self.intrinsics_source_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        left.addWidget(self.intrinsics_source_label)
        left.addStretch()
        split_row.addLayout(left, 1)

        # Right: intrinsics display + export
        right = QVBoxLayout()
        right.setSpacing(4)

        self.intrinsics_display = QTextEdit()
        self.intrinsics_display.setReadOnly(True)
        self.intrinsics_display.setFontFamily("monospace")
        self.intrinsics_display.setFontPointSize(9)
        right.addWidget(self.intrinsics_display, stretch=1)

        self.intrinsics_export_btn = QPushButton("Export…")
        self.intrinsics_export_btn.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        self.intrinsics_export_btn.clicked.connect(self._export_intrinsics)
        right.addWidget(self.intrinsics_export_btn)

        split_row.addLayout(right, 1)
        return panel

    def _build_extrinsics_panel(self) -> QFrame:
        from PySide6.QtWidgets import QGridLayout as _QGL

        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        title = QLabel("Extrinsics (LiDAR → Camera)")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        outer.addWidget(title)

        # ── Left/right split ──────────────────────────────────────────
        split_row = QHBoxLayout()
        split_row.setSpacing(8)
        outer.addLayout(split_row, stretch=1)

        # Left: inputs
        left = QVBoxLayout()
        left.setSpacing(4)

        src_row = QHBoxLayout()
        self.extrinsics_default_btn = QPushButton("Default File")
        self.extrinsics_default_btn.clicked.connect(self._load_default_extrinsics)
        self.extrinsics_file_btn = QPushButton("Load File…")
        self.extrinsics_file_btn.clicked.connect(self._browse_extrinsics_file)
        self.extrinsics_rosbag_btn = QPushButton("ROS Topic")
        self.extrinsics_rosbag_btn.clicked.connect(self._use_rosbag_extrinsics)
        for b in (self.extrinsics_default_btn, self.extrinsics_file_btn,
                  self.extrinsics_rosbag_btn):
            src_row.addWidget(b)
        src_row.addStretch()
        left.addLayout(src_row)

        self.extrinsics_source_label = QLabel()
        self.extrinsics_source_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 11px;"
        )
        left.addWidget(self.extrinsics_source_label)

        # Input mode + angle unit
        mode_row = QHBoxLayout()
        self.transform_mode_combo = QComboBox()
        self.transform_mode_combo.addItems(["XYZ + RPY", "Quaternion"])
        self.transform_mode_combo.currentTextChanged.connect(self._on_transform_mode_changed)
        self.angle_unit_combo = QComboBox()
        self.angle_unit_combo.addItems(["Degrees", "Radians"])
        self.angle_unit_combo.currentTextChanged.connect(self._on_angle_unit_changed)
        self.use_identity_button = QPushButton("Identity")
        self.use_identity_button.clicked.connect(self.use_identity_transform)
        mode_row.addWidget(self.transform_mode_combo)
        mode_row.addWidget(self.angle_unit_combo)
        mode_row.addStretch()
        mode_row.addWidget(self.use_identity_button)
        left.addLayout(mode_row)

        # Stacked input forms
        self.transform_input_stack = QStackedWidget()

        # Index 0 — XYZ + RPY  (2-row grid: tx ty tz / roll pitch yaw)
        xyzrpy_w = QWidget()
        grid = _QGL(xyzrpy_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)

        self.tx_input = QLineEdit("0.0")
        self.ty_input = QLineEdit("0.0")
        self.tz_input = QLineEdit("0.0")
        self.rx_input = QLineEdit("0.0")
        self.ry_input = QLineEdit("0.0")
        self.rz_input = QLineEdit("0.0")

        _lbl_style = "color: #aaa; font-size: 11px;"
        for col, (lbl, le) in enumerate(
            [("tx", self.tx_input), ("ty", self.ty_input), ("tz", self.tz_input)]
        ):
            lw = QLabel(f"{lbl}:")
            lw.setStyleSheet(_lbl_style)
            grid.addWidget(lw, 0, col * 2)
            grid.addWidget(le, 0, col * 2 + 1)
        for col, (lbl, le) in enumerate(
            [("roll", self.rx_input), ("pitch", self.ry_input), ("yaw", self.rz_input)]
        ):
            lw = QLabel(f"{lbl}:")
            lw.setStyleSheet(_lbl_style)
            grid.addWidget(lw, 1, col * 2)
            grid.addWidget(le, 1, col * 2 + 1)

        for le in (self.tx_input, self.ty_input, self.tz_input,
                   self.rx_input, self.ry_input, self.rz_input):
            le.textEdited.connect(self._on_transform_input_edited)
        self.transform_input_stack.addWidget(xyzrpy_w)

        # Index 1 — Quaternion (2-row grid: tx ty tz / qx qy qz qw)
        quat_w = QWidget()
        qgrid = _QGL(quat_w)
        qgrid.setContentsMargins(0, 0, 0, 0)
        qgrid.setSpacing(4)
        qgrid.setColumnStretch(1, 1)
        qgrid.setColumnStretch(3, 1)
        qgrid.setColumnStretch(5, 1)
        qgrid.setColumnStretch(7, 1)

        self.pos_x_input = QLineEdit("0.0")
        self.pos_y_input = QLineEdit("0.0")
        self.pos_z_input = QLineEdit("0.0")
        self.quat_x_input = QLineEdit("0.0")
        self.quat_y_input = QLineEdit("0.0")
        self.quat_z_input = QLineEdit("0.0")
        self.quat_w_input = QLineEdit("1.0")

        for col, (lbl, le) in enumerate(
            [("tx", self.pos_x_input), ("ty", self.pos_y_input), ("tz", self.pos_z_input)]
        ):
            lw = QLabel(f"{lbl}:")
            lw.setStyleSheet(_lbl_style)
            qgrid.addWidget(lw, 0, col * 2)
            qgrid.addWidget(le, 0, col * 2 + 1)
        for col, (lbl, le) in enumerate(
            [("qx", self.quat_x_input), ("qy", self.quat_y_input),
             ("qz", self.quat_z_input), ("qw", self.quat_w_input)]
        ):
            lw = QLabel(f"{lbl}:")
            lw.setStyleSheet(_lbl_style)
            qgrid.addWidget(lw, 1, col * 2)
            qgrid.addWidget(le, 1, col * 2 + 1)

        for le in (self.pos_x_input, self.pos_y_input, self.pos_z_input,
                   self.quat_x_input, self.quat_y_input, self.quat_z_input, self.quat_w_input):
            le.textEdited.connect(self._on_transform_input_edited)
        self.transform_input_stack.addWidget(quat_w)

        left.addWidget(self.transform_input_stack)
        left.addStretch()
        split_row.addLayout(left, 1)

        # Right: YAML display + export
        right = QVBoxLayout()
        right.setSpacing(4)
        yaml_title = QLabel("Static Transform YAML")
        yaml_title.setStyleSheet("color: #aaa; font-size: 11px;")
        right.addWidget(yaml_title)
        self.transform_yaml_display = QTextEdit()
        self.transform_yaml_display.setReadOnly(True)
        self.transform_yaml_display.setFontFamily("monospace")
        self.transform_yaml_display.setFontPointSize(9)
        right.addWidget(self.transform_yaml_display, stretch=1)
        self.extrinsics_export_btn = QPushButton("Export…")
        self.extrinsics_export_btn.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
        self.extrinsics_export_btn.clicked.connect(self._export_extrinsics)
        right.addWidget(self.extrinsics_export_btn)
        split_row.addLayout(right, 1)

        self._update_transform_yaml_display()
        return panel

    # ================================================================== #
    #  Mode toggles                                                        #
    # ================================================================== #

    def _on_mode_btn_clicked(self, mode: str):
        pass  # only cam2lidar mode remains; sub-mode is handled by _on_sub_mode_clicked

    def _on_sub_mode_clicked(self, sub: str):
        _sub_active = (
            "QPushButton { background: #d64814; color: white; border-radius: 0; "
            "padding: 6px 18px; font-weight: bold; }"
        )
        _sub_inactive = (
            "QPushButton { background: #3a3a3a; color: #888; border-radius: 0; "
            "padding: 6px 18px; }"
            "QPushButton:hover { background: #484848; color: #bbb; }"
        )
        if sub == "context":
            self.calibration_type = "LiDAR2Cam_Context"
            self.btn_context.setStyleSheet(_sub_active)
            self.btn_zoom.setStyleSheet(_sub_inactive)
        else:
            self.calibration_type = "LiDAR2Cam_Zoom"
            self.btn_zoom.setStyleSheet(_sub_active)
            self.btn_context.setStyleSheet(_sub_inactive)
        if self.bag_file:
            self._auto_select_topics()
            self._validate_all_topics()

    def _update_topic_visibility(self):
        pass  # always cam2lidar — all topic fields are always visible

    # ================================================================== #
    #  Bag loading                                                         #
    # ================================================================== #

    def load_bag(self):
        bags_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "bags")
        start_dir = bags_dir if os.path.isdir(bags_dir) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Open Rosbag", start_dir, "MCAP Rosbag (*.mcap)")
        if path:
            self.load_bag_from_path(path)

    def find_yaml_file(self, mcap_path: str) -> Optional[str]:
        directory = os.path.dirname(mcap_path)
        base = os.path.splitext(os.path.basename(mcap_path))[0]
        for candidate in [
            os.path.join(directory, "metadata.yaml"),
            os.path.join(directory, f"{base}.yaml"),
        ]:
            if os.path.exists(candidate):
                return candidate
        return None

    def process_dropped_path(self, path: str):
        if os.path.isfile(path) and path.endswith(".mcap"):
            if self.find_yaml_file(path):
                self.load_bag_from_path(path)
            else:
                self.bag_path_label.setText("Error: metadata.yaml not found next to .mcap")
                self.bag_path_label.setStyleSheet("color: #f44; padding: 0 8px;")
        elif os.path.isdir(path):
            mcaps = [f for f in os.listdir(path) if f.endswith(".mcap")]
            if len(mcaps) == 1:
                self.process_dropped_path(os.path.join(path, mcaps[0]))
            elif not mcaps:
                self.bag_path_label.setText("Error: No .mcap file in folder")
            else:
                self.bag_path_label.setText("Error: Multiple .mcap files — drop one")
        else:
            self.bag_path_label.setText("Error: Drop a valid .mcap file or folder")

    def load_bag_from_path(self, file_path: str):
        try:
            self.bag_file = file_path
            self.bag_path_label.setText(os.path.basename(file_path))
            self.bag_path_label.setStyleSheet("color: #ccc; padding: 0 8px;")
            ros_version = self.ros_version_combo.currentText()
            self.topics = get_topic_info(file_path, ros_version)
            self.topic_types = {t: m for t, m, _ in self.topics}
            self.update_topic_widgets()
        except Exception as e:
            self.bag_path_label.setText(f"Error: {e}")
            self.bag_path_label.setStyleSheet("color: #f44; padding: 0 8px;")

    def update_topic_widgets(self):
        image_topics = [
            t for t, m, _ in self.topics
            if m in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
        ]
        pc_topics = [t for t, m, _ in self.topics if m == "sensor_msgs/msg/PointCloud2"]
        info_topics = [t for t, m, _ in self.topics if m == "sensor_msgs/msg/CameraInfo"]

        for combo, items in [
            (self.image_topic_combo, image_topics),
            (self.pointcloud_topic_combo, pc_topics),
            (self.camerainfo_topic_combo, info_topics),
        ]:
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(items)
            combo.blockSignals(False)

        self._auto_select_topics()
        self._validate_all_topics()

    def _auto_select_topics(self):
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
        }.get(self.calibration_type, {})

        available = {t for t, _, _ in self.topics}

        def _try_set(combo, topic):
            if topic in available:
                combo.blockSignals(True)
                combo.setCurrentText(topic)
                combo.blockSignals(False)

        _try_set(self.pointcloud_topic_combo, preferred.get("pointcloud", ""))
        _try_set(self.image_topic_combo, preferred.get("image", ""))
        _try_set(self.camerainfo_topic_combo, preferred.get("camerainfo", ""))

    def _validate_all_topics(self):
        available = {t for t, _, _ in self.topics}
        for combo in (self.image_topic_combo, self.pointcloud_topic_combo,
                      self.camerainfo_topic_combo):
            self._set_topic_state(combo, combo.currentText(), available)
        self._update_process_button_state()

    def _set_topic_state(self, combo: "QComboBox", topic: str, available: set):
        if topic and topic not in available:
            combo.lineEdit().setPlaceholderText("Topic does not exist")
            combo.setStyleSheet(
                "QComboBox { background: #5a1010; color: #ff9999; }"
                "QComboBox QAbstractItemView { background: #2a2a2a; color: #ccc; }"
            )
            combo.blockSignals(True)
            combo.setCurrentText("")
            combo.blockSignals(False)
        else:
            combo.lineEdit().setPlaceholderText("")
            combo.setStyleSheet("")

    def _on_topic_changed(self):
        if self.bag_file:
            self._validate_all_topics()

    def _update_process_button_state(self):
        available = {t for t, _, _ in self.topics}
        ok = (
            self.image_topic_combo.currentText() in available
            and self.pointcloud_topic_combo.currentText() in available
        )
        self.process_button.setEnabled(ok)

    # ================================================================== #
    #  Bag processing                                                      #
    # ================================================================== #

    def process_rosbag_data(self):
        self.process_progress.setVisible(True)
        self.process_progress.setValue(0)
        self.process_button.setEnabled(False)

        tf_topics = [
            t for t, m in self.topic_types.items()
            if "tf" in t.lower() and "TFMessage" in m
        ]

        selected_topics_data = {
            "calibration_type": self.calibration_type,
            "image_topic": self.image_topic_combo.currentText(),
            "pointcloud_topic": self.pointcloud_topic_combo.currentText(),
            "camerainfo_topic": self.camerainfo_topic_combo.currentText(),
            "tf_topics": tf_topics,
        }

        topics_to_read = {}
        for key, val in selected_topics_data.items():
            if key.endswith("_topic") and val and val in self.topic_types:
                topics_to_read[val] = self.topic_types[val]
        for t in tf_topics:
            if t in self.topic_types:
                topics_to_read[t] = self.topic_types[t]

        self.processing_worker = RosbagProcessingWorker(
            bag_file=self.bag_file,
            topics_to_read=topics_to_read,
            selected_topics_data=selected_topics_data,
            total_messages=get_total_message_count(
                self.bag_file, self.ros_version_combo.currentText()
            ),
            frame_samples=self.frame_samples_spinbox.value(),   # max synchronised pairs to read
            topic_message_counts={n: c for n, _, c in self.topics},
            ros_version=self.ros_version_combo.currentText(),
            sync_tolerance=0.05,
        )
        self.processing_worker.progress_updated.connect(self.update_processing_progress)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        self.processing_worker.processing_failed.connect(self.on_processing_failed)
        self.processing_worker.start()

    def update_processing_progress(self, value: int, message: str):
        self.process_progress.setValue(value)
        self.process_progress.setFormat(message)

    def on_processing_finished(
        self, raw_messages: Dict, topic_types: Dict, selected_topics_data: Dict
    ):
        self.process_progress.setVisible(False)
        self.process_button.setEnabled(True)

        self.topic_types.update(topic_types)
        self.selected_topics_data = selected_topics_data

        self.tf_messages = {
            t: raw_messages[t]
            for t in selected_topics_data.get("tf_topics", [])
            if t in raw_messages
        }

        if not self._is_lidar_cam:
            # L2L: just set up extrinsics and launch Open3D
            pc1_topic = selected_topics_data["pointcloud_topic"]
            pc2_topic = selected_topics_data["pointcloud2_topic"]
            frame_samples = raw_messages.get("frame_samples", {})
            if frame_samples:
                pc1_msg = frame_samples[pc1_topic][0]["data"]
                pc2_msg = frame_samples[pc2_topic][0]["data"]
            else:
                pc1_msg = raw_messages.get(pc1_topic)
                pc2_msg = raw_messages.get(pc2_topic)
            if not pc1_msg or not pc2_msg:
                self.on_processing_failed("Synchronized LiDAR messages not found.")
                return
            self.lidar_frame = self.extract_frame_id(pc1_msg)
            self.lidar2_frame = self.extract_frame_id(pc2_msg)
            self._auto_load_extrinsics()
            self.proceed_to_lidar_calibration_with_transform(
                self.current_transform,
                convert_to_mock(pc1_msg, self.topic_types.get(pc1_topic, "")),
                convert_to_mock(pc2_msg, self.topic_types.get(pc2_topic, "")),
            )
            return

        # ── LiDAR-to-Camera path ──────────────────────────────────────
        frame_samples = raw_messages.get("frame_samples", {})
        if not frame_samples:
            self.on_processing_failed("No frame samples returned from bag processor.")
            return

        self.frame_samples = frame_samples
        pc_topic = selected_topics_data["pointcloud_topic"]
        img_topic = selected_topics_data["image_topic"]
        info_topic = selected_topics_data.get("camerainfo_topic", "")

        self.lidar_frame = self.extract_frame_id(frame_samples[pc_topic][0]["data"])
        self.camera_frame = (
            self.extract_frame_id(frame_samples[info_topic][0]["data"])
            if info_topic and info_topic in frame_samples
            else self._child_frame
        )

        # Auto-load intrinsics (rosbag topic → default file)
        self._auto_load_intrinsics()

        # Auto-load extrinsics (TF static → default file)
        self._auto_load_extrinsics()

        # Accumulate all LiDAR frames
        pc_frames = frame_samples[pc_topic]
        merged_pc = self._accumulate_all_frames(pc_frames)

        # Get image from first frame
        image_raw = frame_samples[img_topic][0]["data"]
        image_msg = convert_to_mock(image_raw, self.topic_types.get(img_topic, ""))

        # Enable middle row
        self.middle_row.setEnabled(True)
        self._refresh_intrinsics_display()

        # Create and embed CalibrationWidget
        self._create_calibration_widget(image_msg, merged_pc)

        if hasattr(self, "processing_worker") and self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

    def _accumulate_all_frames(self, pc_frames: list):
        """Decode all LiDAR frames, voxel-downsample for uniform spatial density."""
        pc_topic = self.selected_topics_data.get("pointcloud_topic", "")
        topic_type = self.topic_types.get(pc_topic, "sensor_msgs/msg/PointCloud2")

        # Decode first frame to obtain field layout
        first_msg = convert_to_mock(pc_frames[0]["data"], topic_type)
        dtype_list = ros_utils.fields_to_dtype(first_msg.fields, first_msg.point_step)
        full_dtype = np.dtype(dtype_list)

        # Decode every frame into one concatenated structured array
        parts = []
        for frame in pc_frames:
            msg = convert_to_mock(frame["data"], topic_type)
            arr = np.frombuffer(bytes(msg.data), full_dtype).copy()
            valid = np.isfinite(arr["x"]) & np.isfinite(arr["y"]) & np.isfinite(arr["z"])
            parts.append(arr[valid])

        combined = np.concatenate(parts)

        # Density-equalise with voxel grid downsampling
        downsampled = self._voxel_downsample(
            combined,
            target_max=self.max_points_spinbox.value(),
            voxel_size=self.voxel_size_spinbox.value(),
        )

        n = len(downsampled)
        return ros_utils.PointCloud2(
            header=first_msg.header,
            height=1,
            width=n,
            fields=[
                ros_utils.PointField(
                    name=f.name, offset=f.offset, datatype=f.datatype, count=f.count
                )
                for f in first_msg.fields
            ],
            is_bigendian=first_msg.is_bigendian,
            point_step=first_msg.point_step,
            row_step=first_msg.point_step * n,
            data=downsampled.tobytes(),
            is_dense=True,
        )

    def _voxel_downsample(
        self, arr: np.ndarray, target_max: int, voxel_size: float = VOXEL_SIZE_INIT
    ) -> np.ndarray:
        """Keep one point per voxel cell for uniform spatial density.

        Starts at voxel_size and increases by 50 % each iteration until the
        result fits within target_max points.  np.unique returns the *first* index
        for each occupied voxel, so earlier frames naturally win (they accumulate
        cleanly without duplication).
        """
        xyz = np.column_stack([
            arr["x"].astype(np.float64),
            arr["y"].astype(np.float64),
            arr["z"].astype(np.float64),
        ])

        while True:
            mins = xyz.min(axis=0)
            vi = ((xyz - mins) / voxel_size).astype(np.int64)
            dims = vi.max(axis=0) + 1
            # Encode 3-D voxel index as a single int64 — safe up to ~9e18
            keys = vi[:, 0] * (dims[1] * dims[2]) + vi[:, 1] * dims[2] + vi[:, 2]
            _, keep = np.unique(keys, return_index=True)
            if len(keep) <= target_max:
                break
            voxel_size *= 1.5

        keep.sort()   # restore spatial ordering
        return arr[keep]

    def on_processing_failed(self, error: str):
        self.process_progress.setFormat(f"Error: {error}")
        self.process_button.setEnabled(True)
        if hasattr(self, "processing_worker") and self.processing_worker:
            self.processing_worker.deleteLater()
            self.processing_worker = None

    def extract_frame_id(self, msg) -> str:
        return getattr(getattr(msg, "header", None), "frame_id", "unknown_frame")

    # ================================================================== #
    #  Intrinsics                                                          #
    # ================================================================== #

    def _auto_load_intrinsics(self):
        """Try rosbag CameraInfo topic first, then fall back to default file."""
        info_topic = self.selected_topics_data.get("camerainfo_topic", "")
        if info_topic and info_topic in self.frame_samples:
            try:
                raw = self.frame_samples[info_topic][0]["data"]
                self.active_camerainfo_msg = convert_to_mock(
                    raw, self.topic_types.get(info_topic, "sensor_msgs/msg/CameraInfo")
                )
                self.intrinsics_source_label.setText(f"Source: {info_topic}")
                self._refresh_intrinsics_display()
                self._on_intrinsics_loaded()
                return
            except Exception:
                pass
        self._load_default_intrinsics()

    def _load_default_intrinsics(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path = os.path.join(data_dir, f"{self._camera_type_label}_intrinsics.yaml")
        if os.path.exists(path):
            try:
                self.active_camerainfo_msg = self._parse_camera_info_yaml(path)
                self.intrinsics_source_label.setText(f"Source: {os.path.basename(path)}")
                self._refresh_intrinsics_display()
                self._on_intrinsics_loaded()
            except Exception as e:
                self.intrinsics_source_label.setText(f"Error: {e}")
        else:
            self.intrinsics_source_label.setText("Default file not found.")

    def _browse_intrinsics_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera Intrinsics", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                self.active_camerainfo_msg = self._parse_camera_info_yaml(path)
                self.intrinsics_source_label.setText(f"Source: {path}")
                self._refresh_intrinsics_display()
                self._on_intrinsics_loaded()
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))

    def _use_rosbag_intrinsics(self):
        info_topic = self.selected_topics_data.get("camerainfo_topic", "")
        if not info_topic or info_topic not in self.frame_samples:
            QMessageBox.information(self, "Not available", "No CameraInfo topic in loaded bag.")
            return
        try:
            raw = self.frame_samples[info_topic][0]["data"]
            self.active_camerainfo_msg = convert_to_mock(
                raw, self.topic_types.get(info_topic, "sensor_msgs/msg/CameraInfo")
            )
            self.intrinsics_source_label.setText(f"Source: {info_topic}")
            self._refresh_intrinsics_display()
            self._on_intrinsics_loaded()
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _refresh_intrinsics_display(self):
        msg = self.active_camerainfo_msg
        if msg is None:
            self.intrinsics_display.setPlainText("No intrinsics loaded.")
            return
        k = list(msg.k)
        fx, fy, cx, cy = k[0], k[4], k[2], k[5]
        coeffs = ", ".join(f"{v:.6g}" for v in msg.d)
        self.intrinsics_display.setPlainText(
            f"Image: {msg.width} × {msg.height}\n"
            f"fx={fx:.4g}  fy={fy:.4g}  cx={cx:.4g}  cy={cy:.4g}\n"
            f"Model: {msg.distortion_model}\n"
            f"D: [{coeffs}]"
        )

    def _on_intrinsics_loaded(self):
        if self.calib_widget and self.active_camerainfo_msg:
            self.calib_widget.update_intrinsics(self.active_camerainfo_msg)

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

    def _export_intrinsics(self):
        if not self.active_camerainfo_msg:
            QMessageBox.information(self, "Nothing to export", "Load intrinsics first.")
            return
        msg = self.active_camerainfo_msg
        default = f"{self._camera_type_label}_intrinsics.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save Intrinsics", default, "YAML (*.yaml)")
        if not path:
            return
        k = list(msg.k)
        d = list(msg.d)
        r = list(msg.r)
        p = list(msg.p)
        content = {
            "image_width": msg.width,
            "image_height": msg.height,
            "camera_name": getattr(msg, "camera_name", ""),
            "camera_matrix": {"rows": 3, "cols": 3, "data": k},
            "distortion_model": msg.distortion_model,
            "distortion_coefficients": {"rows": 1, "cols": len(d), "data": d},
            "rectification_matrix": {"rows": 3, "cols": 3, "data": r},
            "projection_matrix": {"rows": 3, "cols": 4, "data": p},
        }
        with open(path, "w") as f:
            yaml.dump(content, f, default_flow_style=False)

    # ================================================================== #
    #  Extrinsics                                                          #
    # ================================================================== #

    def _auto_load_extrinsics(self):
        """Try TF static from bag first, then fall back to default file."""
        T = self._find_transform_in_bag()
        if T is not None:
            self.current_transform = T
            self.extrinsics_source_label.setText(
                f"Source: {self._parent_frame} → {self._child_frame} (TF static)"
            )
            self._sync_transform_inputs_from_matrix()
            return
        self._load_default_extrinsics()

    def _load_default_extrinsics(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        path = os.path.join(data_dir, f"default_{self._camera_type_label}_extrinsics.yaml")
        if os.path.exists(path):
            try:
                self.current_transform = self._parse_extrinsics_yaml(path)
                self.extrinsics_source_label.setText(f"Source: {os.path.basename(path)}")
                self._sync_transform_inputs_from_matrix()
                self._on_extrinsics_changed()
                return
            except Exception as e:
                self.extrinsics_source_label.setText(f"Error loading file: {e}")
        # File not found or parse error — fall back to identity
        self.current_transform = np.eye(4, dtype=np.float64)
        self.extrinsics_source_label.setText("Default file not found — using identity")
        self._sync_transform_inputs_from_matrix()
        self._on_extrinsics_changed()

    def _browse_extrinsics_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Extrinsics", "", "YAML Files (*.yaml *.yml)"
        )
        if path:
            try:
                self.current_transform = self._parse_extrinsics_yaml(path)
                self.extrinsics_source_label.setText(f"Source: {path}")
                self._sync_transform_inputs_from_matrix()
                self._on_extrinsics_changed()
            except Exception as e:
                QMessageBox.warning(self, "Load Error", str(e))

    def _use_rosbag_extrinsics(self):
        T = self._find_transform_in_bag()
        if T is not None:
            self.current_transform = T
            self.extrinsics_source_label.setText(
                f"Source: {self._parent_frame} → {self._child_frame} (TF static)"
            )
        else:
            self.current_transform = np.eye(4, dtype=np.float64)
            self.extrinsics_source_label.setText(
                f"Transform not found in bag — using identity"
            )
        self._sync_transform_inputs_from_matrix()
        self._on_extrinsics_changed()

    def _parse_extrinsics_yaml(self, path: str) -> np.ndarray:
        with open(path) as f:
            data = yaml.safe_load(f)
        entry = list(data.values())[0]
        t = entry["translation"]
        r = entry["rotation"]
        tx, ty, tz = t["x"], t["y"], t["z"]
        qx, qy, qz, qw = r["x"], r["y"], r["z"], r["w"]
        T = tf.quaternion_matrix([qx, qy, qz, qw])
        T[:3, 3] = [tx, ty, tz]
        return T

    def _sync_transform_inputs_from_matrix(self):
        """Populate both input forms from self.current_transform, blocking signals."""
        T = self.current_transform
        trans = tf.translation_from_matrix(T)
        q = tf.quaternion_from_matrix(T)   # [x, y, z, w]
        euler = tf.euler_from_matrix(T)    # radians

        use_degrees = self.angle_unit_combo.currentText() == "Degrees"
        rpy = np.degrees(euler) if use_degrees else euler

        pairs_rpy = [
            (self.tx_input, trans[0]), (self.ty_input, trans[1]), (self.tz_input, trans[2]),
            (self.rx_input, rpy[0]), (self.ry_input, rpy[1]), (self.rz_input, rpy[2]),
        ]
        pairs_quat = [
            (self.pos_x_input, trans[0]), (self.pos_y_input, trans[1]),
            (self.pos_z_input, trans[2]),
            (self.quat_x_input, q[0]), (self.quat_y_input, q[1]),
            (self.quat_z_input, q[2]), (self.quat_w_input, q[3]),
        ]
        for le, val in pairs_rpy + pairs_quat:
            le.blockSignals(True)
            le.setText(f"{val:.7f}")
            le.blockSignals(False)

        self._update_transform_yaml_display()

    def _on_transform_input_edited(self):
        try:
            if self.transform_input_stack.currentIndex() == 0:
                tx = float(self.tx_input.text())
                ty = float(self.ty_input.text())
                tz = float(self.tz_input.text())
                rx = float(self.rx_input.text())
                ry = float(self.ry_input.text())
                rz = float(self.rz_input.text())
                use_degrees = self.angle_unit_combo.currentText() == "Degrees"
                R = Rotation.from_euler("xyz", [rx, ry, rz], degrees=use_degrees).as_matrix()
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
            else:
                tx = float(self.pos_x_input.text())
                ty = float(self.pos_y_input.text())
                tz = float(self.pos_z_input.text())
                qx = float(self.quat_x_input.text())
                qy = float(self.quat_y_input.text())
                qz = float(self.quat_z_input.text())
                qw = float(self.quat_w_input.text())
                norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
                if norm < 1e-9:
                    return
                qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
                T = tf.quaternion_matrix([qx, qy, qz, qw])
                T[:3, 3] = [tx, ty, tz]
        except ValueError:
            return  # still typing

        self.current_transform = T
        self._update_other_input_form()
        self._update_transform_yaml_display()
        self._on_extrinsics_changed()

    def _update_other_input_form(self):
        T = self.current_transform
        trans = tf.translation_from_matrix(T)
        q = tf.quaternion_from_matrix(T)
        euler = tf.euler_from_matrix(T)
        use_degrees = self.angle_unit_combo.currentText() == "Degrees"
        rpy = np.degrees(euler) if use_degrees else euler

        if self.transform_input_stack.currentIndex() == 0:
            for le, val in [
                (self.pos_x_input, trans[0]), (self.pos_y_input, trans[1]),
                (self.pos_z_input, trans[2]),
                (self.quat_x_input, q[0]), (self.quat_y_input, q[1]),
                (self.quat_z_input, q[2]), (self.quat_w_input, q[3]),
            ]:
                le.blockSignals(True)
                le.setText(f"{val:.7f}")
                le.blockSignals(False)
        else:
            for le, val in [
                (self.tx_input, trans[0]), (self.ty_input, trans[1]),
                (self.tz_input, trans[2]),
                (self.rx_input, rpy[0]), (self.ry_input, rpy[1]), (self.rz_input, rpy[2]),
            ]:
                le.blockSignals(True)
                le.setText(f"{val:.7f}")
                le.blockSignals(False)

    def _on_transform_mode_changed(self, mode: str):
        self.transform_input_stack.setCurrentIndex(0 if mode == "XYZ + RPY" else 1)
        self.angle_unit_combo.setVisible(mode == "XYZ + RPY")

    def _on_angle_unit_changed(self):
        T = self.current_transform
        euler = tf.euler_from_matrix(T)
        use_degrees = self.angle_unit_combo.currentText() == "Degrees"
        rpy = np.degrees(euler) if use_degrees else euler
        for le, val in [
            (self.rx_input, rpy[0]), (self.ry_input, rpy[1]), (self.rz_input, rpy[2])
        ]:
            le.blockSignals(True)
            le.setText(f"{val:.7f}")
            le.blockSignals(False)

    def _update_transform_yaml_display(self):
        T = self.current_transform
        trans = tf.translation_from_matrix(T)
        q = tf.quaternion_from_matrix(T)  # [x, y, z, w]
        key = f"livox_to_{self._camera_type_label}" if self._is_lidar_cam else "lidar_to_lidar"
        parent = self._parent_frame
        child = self._child_frame
        text = (
            f"{key}:\n"
            f"  parent_frame: {parent}\n"
            f"  child_frame: {child}\n"
            f"  translation:\n"
            f"    x: {trans[0]:.7f}\n"
            f"    y: {trans[1]:.7f}\n"
            f"    z: {trans[2]:.7f}\n"
            f"  rotation:\n"
            f"    x: {q[0]:.7f}\n"
            f"    y: {q[1]:.7f}\n"
            f"    z: {q[2]:.7f}\n"
            f"    w: {q[3]:.7f}\n"
        )
        self.transform_yaml_display.setPlainText(text)

    def use_identity_transform(self):
        self.current_transform = np.eye(4, dtype=np.float64)
        self._sync_transform_inputs_from_matrix()
        self._on_extrinsics_changed()

    def _on_extrinsics_changed(self):
        if self.calib_widget:
            self.calib_widget.update_extrinsics(self.current_transform)

    def _export_extrinsics(self):
        default = f"livox_to_{self._camera_type_label}.yaml"
        path, _ = QFileDialog.getSaveFileName(self, "Save Extrinsics", default, "YAML (*.yaml)")
        if path:
            self._write_calib_yaml(path)

    # ================================================================== #
    #  TF tree                                                            #
    # ================================================================== #

    def _find_transform_in_bag(self) -> Optional[np.ndarray]:
        """Parse /tf_static from bag and return parent→child transform, or None."""
        if not self.tf_messages:
            return None
        tf_topic = next(
            (t for t in self.tf_messages if "tf_static" in t),
            next(iter(self.tf_messages), None),
        )
        if not tf_topic:
            return None
        self.tf_tree = self.parse_preloaded_tf_message(self.tf_messages[tf_topic])
        return self.find_transform_path(self._parent_frame, self._child_frame)

    def parse_preloaded_tf_message(self, msg_data) -> Dict:
        tf_tree: Dict = {}
        for ts in self.deserialize_tf_message(msg_data).transforms:
            parent = ts.header.frame_id
            child = ts.child_frame_id
            tf_tree.setdefault(parent, {})[child] = {
                "transform": ros_utils.transform_to_numpy(ts.transform)
            }
        return tf_tree

    def deserialize_tf_message(self, msg_data) -> "ros_utils.TFMessage":
        if not hasattr(msg_data, "transforms"):
            return ros_utils.TFMessage(transforms=[])
        transforms = []
        for tm in msg_data.transforms:
            t_obj = tm.transform.translation
            r_obj = tm.transform.rotation
            transforms.append(
                ros_utils.TransformStamped(
                    header=ros_utils.Header(frame_id=tm.header.frame_id),
                    child_frame_id=tm.child_frame_id,
                    transform=ros_utils.Transform(
                        translation=ros_utils.Vector3(x=t_obj.x, y=t_obj.y, z=t_obj.z),
                        rotation=ros_utils.Quaternion(x=r_obj.x, y=r_obj.y, z=r_obj.z, w=r_obj.w),
                    ),
                )
            )
        return ros_utils.TFMessage(transforms=transforms)

    def find_transform_path(self, from_frame: str, to_frame: str) -> Optional[np.ndarray]:
        if from_frame == to_frame:
            return np.eye(4)
        if not self.tf_tree:
            return None
        from collections import deque
        adj: Dict[str, list] = {f: [] for f in self._get_all_tf_frames()}
        for p, children in self.tf_tree.items():
            for c, data in children.items():
                adj[p].append((c, data["transform"]))
                adj[c].append((p, np.linalg.inv(data["transform"])))
        q = deque([(from_frame, np.eye(4))])
        visited = {from_frame}
        while q:
            curr, T = q.popleft()
            if curr == to_frame:
                return T
            for nb, t in adj.get(curr, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, T @ t))
        return None

    def find_transformation_path_frames(
        self, from_frame: str, to_frame: str
    ) -> Optional[List[str]]:
        if from_frame == to_frame:
            return [from_frame]
        if not self.tf_tree:
            return None
        from collections import deque
        adj: Dict[str, list] = {f: [] for f in self._get_all_tf_frames()}
        for p, children in self.tf_tree.items():
            for c in children:
                adj[p].append(c)
                adj[c].append(p)
        q = deque([(from_frame, [from_frame])])
        visited = {from_frame}
        while q:
            curr, path = q.popleft()
            if curr == to_frame:
                return path
            for nb in adj.get(curr, []):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, path + [nb]))
        return None

    def show_tf_graph(self):
        if not self.tf_tree:
            return
        target = self.camera_frame if self._is_lidar_cam else getattr(self, "lidar2_frame", "")
        path = self.find_transformation_path_frames(self.lidar_frame, target)
        self.tf_graph_window = TFGraphWidget(
            self.tf_tree, self.lidar_frame, target, path, parent=self
        )
        self.tf_graph_window.show()

    def _get_all_tf_frames(self) -> List[str]:
        frames = set(self.tf_tree.keys())
        for children in self.tf_tree.values():
            frames.update(children.keys())
        return sorted(frames)

    # ================================================================== #
    #  Calibration widget                                                  #
    # ================================================================== #

    def _create_calibration_widget(self, image_msg, pointcloud_msg):
        # ── Tear down any previous calibration widget ──────────────────
        if self.calib_widget is not None:
            self._vcl.removeWidget(self.calib_widget.view)
            self.calib_widget.view.setParent(None)
            # Reset side panel to placeholder
            _sph = QLabel("Controls appear after processing")
            _sph.setAlignment(Qt.AlignCenter)
            _sph.setWordWrap(True)
            _sph.setStyleSheet("color: #444; padding: 20px;")
            self.side_scroll.setWidget(_sph)
            self.calib_widget.deleteLater()
            self.calib_widget = None

        # Remove the first-run placeholder if still present
        if self._view_placeholder:
            self._vcl.removeWidget(self._view_placeholder)
            self._view_placeholder.deleteLater()
            self._view_placeholder = None

        cinfo = self.active_camerainfo_msg or self._parse_camera_info_yaml(
            os.path.join(os.path.dirname(__file__), "data",
                         f"{self._camera_type_label}_intrinsics.yaml")
        )
        self.calib_widget = CalibrationWidget(
            image_msg, pointcloud_msg, cinfo, ros_utils, self.current_transform
        )
        self.calib_widget.calibration_completed.connect(self.calibration_completed)
        self.calib_widget.extrinsics_updated.connect(self._on_calibration_result_update)

        # Embed the ZoomableView in the left panel
        self.calib_widget.view.setParent(self.view_container)
        self._vcl.addWidget(self.calib_widget.view)

        # Find the QScrollArea child that holds the side panel controls
        side_scroll = None
        for child in self.calib_widget.children():
            if isinstance(child, QScrollArea):
                side_scroll = child
                break
        if side_scroll is not None:
            inner = side_scroll.takeWidget()
            self.side_scroll.setWidget(inner)
        else:
            self.side_scroll.setWidget(self.calib_widget)

    def _on_calibration_result_update(self, T_lidar_cam: np.ndarray):
        """Called when CalibrationWidget finishes calibration or refinement."""
        self.current_transform = T_lidar_cam
        self._sync_transform_inputs_from_matrix()
        self.extrinsics_source_label.setText("Source: Calibration result")

    def _on_calibration_result(self, calibration_results):
        if isinstance(calibration_results, dict):
            extrinsics = calibration_results.get("master_to_camera", np.eye(4))
        elif isinstance(calibration_results, np.ndarray):
            extrinsics = calibration_results
        else:
            return
        # extrinsics is T_cam_lidar; invert to get T_lidar_cam
        T_lidar_cam = np.linalg.inv(extrinsics)
        self._on_calibration_result_update(T_lidar_cam)

    # ================================================================== #
    #  LiDAR-to-LiDAR (unchanged flow)                                    #
    # ================================================================== #

    def proceed_to_lidar_calibration_with_transform(self, initial_transform, pc1, pc2):
        import threading
        threading.Thread(
            target=self._run_lidar_calibration_thread,
            args=(pc1, pc2, initial_transform),
            daemon=True,
        ).start()

    def _run_lidar_calibration_thread(self, pc1, pc2, initial_transform):
        try:
            # Imported lazily: open3d SIGILLs on CPUs without AVX (see note at top).
            from .lidar2lidar_o3d_widget import launch_lidar2lidar_calibration

            launch_lidar2lidar_calibration(pc1, pc2, initial_transform, self._on_lidar_cal_done)
        except Exception as e:
            print(f"[ERROR] LiDAR calibration thread: {e}")

    def _on_lidar_cal_done(self, final_transform: np.ndarray):
        self.calibration_completed.emit(final_transform)

    # ================================================================== #
    #  Export                                                              #
    # ================================================================== #

    def _write_calib_yaml(self, file_path: str):
        T = self.current_transform
        t = T[:3, 3]
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
        key = f"livox_to_{self._camera_type_label}" if self._is_lidar_cam else "lidar_to_lidar"
        timestamp = datetime.date.today().isoformat()
        content = (
            f"{key}:\n"
            f"  parent_frame: {self._parent_frame}\n"
            f"  child_frame: {self._child_frame}\n"
            f"  translation:\n"
            f"    x: {t[0]:.7f}\n"
            f"    y: {t[1]:.7f}\n"
            f"    z: {t[2]:.7f}\n"
            f"  rotation:\n"
            f"    x: {q[0]:.7f}\n"
            f"    y: {q[1]:.7f}\n"
            f"    z: {q[2]:.7f}\n"
            f"    w: {q[3]:.7f}\n"
            f"  metadata:\n"
            f"    source_file: ros2_calib\n"
            f"    conversion_timestamp: '{timestamp}'\n"
        )
        with open(file_path, "w") as f:
            f.write(content)

    # ================================================================== #
    #  Drag & drop                                                         #
    # ================================================================== #

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            self.process_dropped_path(event.mimeData().urls()[0].toLocalFile())
            event.acceptProposedAction()
