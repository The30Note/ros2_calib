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

from functools import partial

import cv2
import matplotlib.cm as cm
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from . import calibration
from .common import AppConstants, Colors, UIStyles
from .lidar_cleaner import LiDARCleaner


class CollapsibleSection(QWidget):
    """A header button + body widget pair that can be toggled open/closed."""

    def __init__(self, title: str, collapsed: bool = False, parent=None):
        super().__init__(parent)
        self._title = title
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(not collapsed)
        self._btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 8px; font-weight: bold;"
            " background: #2d2d2d; border: none; color: #ccc; }"
            " QPushButton:hover { background: #383838; }"
        )
        self._btn.toggled.connect(self._on_toggle)
        root.addWidget(self._btn)

        self.body = QWidget()
        self._body_layout = QVBoxLayout(self.body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        self._body_layout.setSpacing(4)
        root.addWidget(self.body)

        self._on_toggle(not collapsed)

    def _on_toggle(self, expanded: bool):
        self.body.setVisible(expanded)
        arrow = "▼" if expanded else "▶"
        self._btn.setText(f"{arrow}  {self._title}")

    def add_widget(self, w: QWidget):
        self._body_layout.addWidget(w)

    def add_layout(self, layout):
        self._body_layout.addLayout(layout)

    def body_layout(self):
        return self._body_layout


_VORONOI_FILL_MAX_DIST = 20   # pixels at full resolution; controls boundary tightness
_VORONOI_DOWNSAMPLE    = 4    # compute fill at 1/N resolution, then upscale


class PointCloudItem(QGraphicsItem):
    """A QGraphicsItem that draws projected LiDAR points.

    When point_size == 0 the item switches to Voronoi-fill mode: every pixel
    within _VORONOI_FILL_MAX_DIST of a projected point gets the colour of its
    nearest neighbour, producing a solid filled map that stays within the
    actual footprint of the scan without bleeding to image edges.
    """

    def __init__(self, points, colors, point_size, opacity=0.8, img_size=(0, 0)):
        super().__init__()
        self.points = points
        self.point_size = point_size
        self.opacity = opacity
        self.img_h, self.img_w = img_size
        colors_arr = np.array(colors)
        self._colors_u8 = (colors_arr * 255).clip(0, 255).astype(np.uint8)
        self._pixmap = None
        self._pixmap_origin = None
        self._build_pixmap()

    def update_data(self, points, colors, point_size, opacity, img_size=(0, 0)):
        self.prepareGeometryChange()
        self.points = points
        self.point_size = point_size
        self.opacity = opacity
        self.img_h, self.img_w = img_size
        self._colors_u8 = (np.array(colors) * 255).clip(0, 255).astype(np.uint8)
        self._build_pixmap()
        self.update()

    def _build_pixmap(self):
        if self.points.shape[0] == 0:
            self._pixmap = None
            return
        if self.point_size == 0:
            self._build_voronoi_pixmap()
        else:
            self._build_dot_pixmap()

    # ── dot / square rendering (existing behaviour) ─────────────────────
    def _build_dot_pixmap(self):
        s = max(1, int(self.point_size))
        pad = s // 2
        xs = self.points[:, 0].astype(np.int32)
        ys = self.points[:, 1].astype(np.int32)
        x0, y0 = int(xs.min()) - pad, int(ys.min()) - pad
        w = int(xs.max()) - x0 + pad + 1
        h = int(ys.max()) - y0 + pad + 1

        img = np.zeros((h, w, 4), dtype=np.uint8)
        lxs = np.clip(xs - x0, 0, w - 1)
        lys = np.clip(ys - y0, 0, h - 1)
        img[lys, lxs] = self._colors_u8

        if s > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
            img = cv2.dilate(img, kernel)

        qimg = QImage(img.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._pixmap_origin = (x0, y0)

    # ── Voronoi nearest-neighbour fill ───────────────────────────────────
    def _build_voronoi_pixmap(self):
        if self.img_h == 0 or self.img_w == 0:
            self._build_dot_pixmap()   # no image dims — fall back
            return

        sc = _VORONOI_DOWNSAMPLE
        sh, sw = self.img_h // sc, self.img_w // sc

        # Projected point coords scaled to low-res space
        pts_lo = self.points / sc          # (N, 2)

        # Low-res pixel grid as (M, 2) query array  [x, y]
        gx = np.arange(sw, dtype=np.float32) + 0.5
        gy = np.arange(sh, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(gx, gy)          # each (sh, sw)
        query = np.column_stack([grid_x.ravel(), grid_y.ravel()])  # (sh*sw, 2)

        tree = KDTree(pts_lo)
        dists, idxs = tree.query(query, workers=-1)    # use all CPU cores

        # Assign colours; mask pixels too far from any point
        fill = self._colors_u8[idxs].astype(np.float32)          # (sh*sw, 4)
        max_dist_lo = _VORONOI_FILL_MAX_DIST / sc
        fill[dists > max_dist_lo, 3] = 0                          # alpha → 0

        fill_img = fill.reshape(sh, sw, 4).astype(np.uint8)

        # Upscale colour channels with bilinear for smooth gradients;
        # upscale alpha with nearest to keep a hard boundary (no fringe).
        rgb_up = cv2.resize(fill_img[:, :, :3], (self.img_w, self.img_h),
                            interpolation=cv2.INTER_LINEAR)
        a_up   = cv2.resize(fill_img[:, :,  3], (self.img_w, self.img_h),
                            interpolation=cv2.INTER_NEAREST)
        full   = np.dstack([rgb_up, a_up])

        # Opacity is applied at paint time via painter.setOpacity(), not baked here.
        qimg = QImage(full.tobytes(), self.img_w, self.img_h,
                      4 * self.img_w, QImage.Format_RGBA8888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._pixmap_origin = (0, 0)

    # ── QGraphicsItem interface ──────────────────────────────────────────
    def boundingRect(self):
        if self.points.shape[0] == 0:
            return QRectF()
        if self.point_size == 0 and self.img_w > 0:
            return QRectF(0, 0, self.img_w, self.img_h)
        min_coords = np.min(self.points, axis=0)
        max_coords = np.max(self.points, axis=0)
        pad = max(self.point_size // 2, 1)
        return QRectF(
            min_coords[0] - pad,
            min_coords[1] - pad,
            max_coords[0] - min_coords[0] + pad * 2,
            max_coords[1] - min_coords[1] + pad * 2,
        )

    def paint(self, painter: QPainter, option, widget=None):
        if self._pixmap is None:
            return
        painter.setOpacity(self.opacity)
        painter.drawPixmap(int(self._pixmap_origin[0]), int(self._pixmap_origin[1]),
                           self._pixmap)
        painter.setOpacity(1.0)

    def update_opacity(self, opacity: float):
        """Change opacity without rebuilding the pixmap — just triggers a repaint."""
        self.opacity = opacity
        self.update()


class ZoomableView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._on_resize_callback = None

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._on_resize_callback:
            self._on_resize_callback()


class CalibrationWidget(QWidget):
    calibration_completed = Signal(object)   # Signal to emit calibrated transform(s)
    _refinement_done = Signal(object)        # emitted from worker thread with new extrinsics
    extrinsics_updated = Signal(np.ndarray)  # T_lidar_cam after refinement/calibration

    def __init__(
        self,
        image_msg,
        pointcloud_msg,
        camerainfo_msg,
        ros_utils,
        initial_transform,
        second_pointcloud_msg=None,
    ):
        super().__init__()
        self.image_msg = image_msg
        self.pointcloud_msg = pointcloud_msg  # Master point cloud
        self.second_pointcloud_msg = second_pointcloud_msg  # Optional second point cloud
        self.camerainfo_msg = camerainfo_msg
        self.ros_utils = ros_utils
        self.correspondences = {}  # Master LiDAR to camera correspondences
        self.lidar_to_lidar_correspondences = {}  # Second LiDAR to master LiDAR correspondences

        # Use inverse as transform point cloud to camera frame
        self.initial_extrinsics = np.linalg.inv(initial_transform)
        self.extrinsics = np.copy(self.initial_extrinsics)
        self.second_lidar_transform = np.eye(4)  # Transform from master to second LiDAR
        self.occlusion_mask = None
        self.second_occlusion_mask = None

        # Euler convention used for manual tuning of rotations
        self.euler_convention = "xyz"
        self.euler_convention_options = [
            "xyz",  # roll–pitch–yaw / Tait–Bryan
            "zxy",  # alternative to avoid gimbal lock when pitch ≈ ±90°
            "zyz",  # classic Euler angles (same-axis convention)
        ]

        # Image rectification state
        self.original_cv_image = None
        self.is_rectification_enabled = False
        self._is_fisheye = getattr(self.camerainfo_msg, "distortion_model", "").lower() in (
            "fisheye", "kannala_brandt", "equidistant"
        )

        self.selection_mode = None
        self.selected_2d_point = None
        self.temp_2d_marker = []
        self.current_3d_selection = []
        self.highlighted_3d_items = []
        self.selected_3d_items_map = {}

        # Master point cloud visualization
        self.point_cloud_item = None
        self.kdtree = None

        # Second point cloud visualization
        self.second_point_cloud_item = None
        self.second_kdtree = None
        self.has_second_pointcloud = second_pointcloud_msg is not None

        self.default_button_style = UIStyles.DEFAULT_BUTTON
        self._grid_items: list = []
        self._grid_visible: bool = False

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.scene = QGraphicsScene()
        self._bg_pixmap_item = None
        self.view = ZoomableView(self.scene)
        self.view.viewport().installEventFilter(self)
        self.view._on_resize_callback = self._reposition_overlay

        right_scroll = self._setup_controls()

        main_layout.addWidget(self.view, 3)
        main_layout.addWidget(right_scroll, 1)

        self._setup_overlay_buttons()

        self._refinement_done.connect(self._apply_refinement_result)
        self._edge_map_cache = None

        self.display_image()
        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self._update_calibrate_button_highlight()

    def has_significant_distortion(self):
        """Check if camera has significant distortion coefficients."""
        if not hasattr(self.camerainfo_msg, "d"):
            return False

        # Convert distortion coefficients to numpy array
        dist_coeffs = np.array(self.camerainfo_msg.d)

        # Check if the array is empty or all zeros
        if dist_coeffs.size == 0:
            return False

        # Check if any distortion coefficient is significantly non-zero
        # Use a threshold to account for numerical precision
        threshold = 1e-6
        return bool(np.any(np.abs(dist_coeffs) > threshold))

    def _undistort_points_for_calib(self, pts_2d):
        """Return 2D correspondence points in undistorted pinhole pixel space (P=K).

        PnP/projectPoints in calibrate() run a pinhole model with no distortion, so the
        2D points must be expressed in undistorted pixel coordinates. When rectification
        is enabled the stored points are already in that space; when it is disabled they
        are raw distorted pixels and must be undistorted with the appropriate lens model.
        """
        pts = np.asarray(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
        if self.is_rectification_enabled or not self.has_significant_distortion():
            return pts.reshape(-1, 2)
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        D = np.array(self.camerainfo_msg.d)
        if self._is_fisheye:
            und = cv2.fisheye.undistortPoints(pts, K, D[:4].reshape(4, 1), P=K)
        else:
            und = cv2.undistortPoints(pts, K, D, P=K)
        return und.reshape(-1, 2)

    def _remap_correspondences(self, to_rectified: bool):
        """Transform stored 2D correspondence keys between distorted and undistorted pixel space."""
        if not self.correspondences:
            return
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        D = np.array(self.camerainfo_msg.d)

        pts = np.array([[u, v] for u, v in self.correspondences.keys()],
                       dtype=np.float32).reshape(-1, 1, 2)

        if to_rectified:
            # Distorted pixels → undistorted pixels
            if self._is_fisheye:
                remapped = cv2.fisheye.undistortPoints(pts, K, D[:4].reshape(4, 1), P=K)
            else:
                remapped = cv2.undistortPoints(pts, K, D, P=K)
        else:
            # Undistorted pixels → distorted pixels via forward distortion formula
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            xn = (pts[:, 0, 0] - cx) / fx
            yn = (pts[:, 0, 1] - cy) / fy
            if self._is_fisheye:
                k1, k2, k3, k4 = D[:4]
                r = np.sqrt(xn ** 2 + yn ** 2)
                theta = np.arctan(r)
                t2 = theta ** 2
                theta_d = theta * (1 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
                scale = np.where(r < 1e-8, 1.0, theta_d / r)
                xd, yd = scale * xn, scale * yn
            else:
                k1, k2 = D[0], D[1]
                p1, p2 = D[2], D[3]
                k3 = D[4] if len(D) > 4 else 0.0
                r2 = xn ** 2 + yn ** 2
                radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
                xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn ** 2)
                yd = yn * radial + p1 * (r2 + 2 * yn ** 2) + 2 * p2 * xn * yn
            remapped = np.stack([fx * xd + cx, fy * yd + cy], axis=-1).reshape(-1, 1, 2)

        old_keys = list(self.correspondences.keys())
        new_corr = {}
        for old_key, new_pt in zip(old_keys, remapped):
            new_key = (float(new_pt[0, 0]), float(new_pt[0, 1]))
            new_corr[new_key] = self.correspondences[old_key]
        self.correspondences = new_corr
        self.update_corr_list()

    def toggle_rectification(self, enabled):
        """Toggle image rectification on/off."""
        self._remap_correspondences(to_rectified=enabled)
        self.is_rectification_enabled = enabled
        self.display_image()  # Refresh the display

    def rectify_image(self, image):
        """Apply camera undistortion to the image using cv2.undistort."""
        if not self.has_significant_distortion():
            return image

        # Get camera matrix and distortion coefficients
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        dist_coeffs = np.array(self.camerainfo_msg.d)

        # Undistort the image
        try:
            # Use cv2.undistort with the same camera matrix as newCameraMatrix
            # This preserves the same image dimensions and focal length
            if self._is_fisheye:
                rectified_image = cv2.fisheye.undistortImage(image, K, dist_coeffs[:4].reshape(4, 1), None, K)
            else:
                rectified_image = cv2.undistort(image, K, dist_coeffs, None, K)
            return rectified_image
        except Exception as e:
            print(f"[WARNING] Failed to rectify image: {e}")
            return image

    def _setup_controls(self) -> QScrollArea:
        """Build the scrollable side panel with collapsible sections."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        root = QVBoxLayout(container)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # ── View Settings ────────────────────────────────────────────────
        view_sec = CollapsibleSection("View Settings")

        form_w = QWidget()
        view_form = QFormLayout(form_w)
        view_form.setContentsMargins(0, 0, 0, 0)

        self.image_res_label = QLabel("N/A")
        view_form.addRow("Resolution:", self.image_res_label)

        self.point_size_spinbox = QSpinBox()
        self.point_size_spinbox.setRange(0, 10)
        self.point_size_spinbox.setValue(AppConstants.DEFAULT_POINT_SIZE)
        self.point_size_spinbox.setSpecialValueText("Fill")  # shown when value == 0
        view_form.addRow("Point Size:", self.point_size_spinbox)

        self.opacity_spinbox = QDoubleSpinBox()
        self.opacity_spinbox.setRange(0.0, 1.0)
        self.opacity_spinbox.setSingleStep(0.05)
        self.opacity_spinbox.setValue(0.8)
        view_form.addRow("Opacity:", self.opacity_spinbox)

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            ["autumn", "jet", "winter", "summer", "spring", "hot",
             "magma", "inferno", "Spectral", "RdYlGn"]
        )
        self.colormap_combo.setCurrentText(AppConstants.DEFAULT_COLORMAP)
        view_form.addRow("Colormap:", self.colormap_combo)

        self.colorization_mode_combo = QComboBox()
        self.colorization_mode_combo.addItems(
            ["Intensity", "Distance", "LiDAR Edge", "Surface Normals"]
        )
        view_form.addRow("Color Mode:", self.colorization_mode_combo)

        self.min_value_spinbox = QDoubleSpinBox()
        self.min_value_spinbox.setRange(-1e9, 1e9)
        self.min_value_spinbox.setDecimals(2)
        view_form.addRow("Min:", self.min_value_spinbox)

        self.max_value_spinbox = QDoubleSpinBox()
        self.max_value_spinbox.setRange(-1e9, 1e9)
        self.max_value_spinbox.setDecimals(2)
        view_form.addRow("Max:", self.max_value_spinbox)

        # Surface-normals rotation rows (hidden by default)
        self._normal_rot_labels = []
        self._normal_rot_spinboxes = []
        for lbl_text in ("Roll (°):", "Pitch (°):", "Yaw (°):"):
            sb = QDoubleSpinBox()
            sb.setRange(-180.0, 180.0)
            sb.setSingleStep(1.0)
            sb.setDecimals(1)
            view_form.addRow(lbl_text, sb)
            lbl_widget = view_form.itemAt(
                view_form.rowCount() - 1, QFormLayout.LabelRole
            ).widget()
            self._normal_rot_labels.append(lbl_widget)
            self._normal_rot_spinboxes.append(sb)
        self._set_normal_rotation_visible(False)

        self.clean_occlusion_button = QPushButton("Clean Occluded Points")
        self.clean_occlusion_button.clicked.connect(self.run_occlusion_cleaning)

        view_sec.add_widget(form_w)
        view_sec.add_widget(self.clean_occlusion_button)
        root.addWidget(view_sec)

        # ── Correspondences ──────────────────────────────────────────────
        corr_sec = CollapsibleSection("Correspondences")

        if self.has_second_pointcloud:
            self.correspondence_mode_combo = QComboBox()
            self.correspondence_mode_combo.addItems(
                ["Master LiDAR ↔ Camera", "Second LiDAR ↔ Master LiDAR"]
            )
            corr_sec.add_widget(QLabel("Mode:"))
            corr_sec.add_widget(self.correspondence_mode_combo)

        self.add_corr_button = QPushButton("Add Correspondence")
        self.add_corr_button.setCheckable(True)
        self.add_corr_button.toggled.connect(self.toggle_selection_mode)
        corr_sec.add_widget(self.add_corr_button)

        self.confirm_3d_button = QPushButton("Confirm 3D Selection")
        self.confirm_3d_button.setVisible(False)
        self.confirm_3d_button.clicked.connect(self.finalize_correspondence)
        corr_sec.add_widget(self.confirm_3d_button)

        self.corr_list_widget = QListWidget()
        self.corr_list_widget.currentItemChanged.connect(self.highlight_from_list)
        corr_sec.add_widget(self.corr_list_widget)

        self.delete_corr_button = QPushButton("Delete Selected")
        self.delete_corr_button.clicked.connect(self.delete_correspondence)
        corr_sec.add_widget(self.delete_corr_button)

        root.addWidget(corr_sec)

        # ── Calibration ──────────────────────────────────────────────────
        calib_sec = CollapsibleSection("Calibration")

        calib_form_w = QWidget()
        calib_form = QFormLayout(calib_form_w)
        calib_form.setContentsMargins(0, 0, 0, 0)

        self.pnp_solver_combo = QComboBox()
        self.pnp_solver_combo.addItems(["Iterative", "SQPnP", "None"])
        calib_form.addRow("RANSAC:", self.pnp_solver_combo)

        self.lsq_method_combo = QComboBox()
        self.lsq_method_combo.addItems(["lm", "trf", "dogbox"])
        calib_form.addRow("LSQ Method:", self.lsq_method_combo)

        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.run_calibration)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)

        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_calibration_state)

        self.results_label = QLabel("Results:")
        self.results_label.setWordWrap(True)

        calib_sec.add_widget(calib_form_w)
        calib_sec.add_widget(self.calibrate_button)
        calib_sec.add_widget(self.progress_bar)
        calib_sec.add_widget(self.reset_button)
        calib_sec.add_widget(self.results_label)
        root.addWidget(calib_sec)

        # ── Auto Refinement ──────────────────────────────────────────────
        refine_sec = CollapsibleSection("Auto Refinement")

        self.auto_refine_button = QPushButton("Auto Refine (Edge Alignment)")
        self.auto_refine_button.setToolTip(
            "Optimise 6-DOF transform by aligning LiDAR depth edges with image edges.\n"
            "Requires a good initial guess (within ~5° / 5 cm)."
        )
        self.auto_refine_button.clicked.connect(self.run_auto_refinement)

        self.refine_status_label = QLabel("Ready")
        self.refine_status_label.setStyleSheet("color: #888; font-size: 11px;")

        refine_sec.add_widget(self.auto_refine_button)
        refine_sec.add_widget(self.refine_status_label)
        root.addWidget(refine_sec)

        self.export_button = QPushButton("Export Calibration")
        self.export_button.clicked.connect(self.export_calibration)
        root.addWidget(self.export_button)

        root.addStretch()
        scroll.setWidget(container)

        # Signal connections
        self.point_size_spinbox.valueChanged.connect(self._on_view_params_changed)
        self.opacity_spinbox.valueChanged.connect(self._on_opacity_changed)
        self.colormap_combo.currentTextChanged.connect(self._on_view_params_changed)
        self.colorization_mode_combo.currentTextChanged.connect(self._on_colorization_mode_changed)
        for sb in self._normal_rot_spinboxes:
            sb.valueChanged.connect(self._on_view_params_changed)
        self.min_value_spinbox.valueChanged.connect(self._on_view_params_changed)
        self.max_value_spinbox.valueChanged.connect(self._on_view_params_changed)

        return scroll

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.confirm_3d_button.isVisible() and self.confirm_3d_button.isEnabled():
                self.finalize_correspondence()
                event.accept()
                return
        elif event.key() == Qt.Key_Escape:
            # Cancel current correspondence selection (ESC key)
            if self.selection_mode is not None:
                self.reset_selection_mode()
                event.accept()
                return
        elif event.key() == Qt.Key_Backspace:
            # Delete selected correspondence (Backspace key)
            current_item = self.corr_list_widget.currentItem()
            if current_item:
                self.delete_correspondence()
                event.accept()
                return
        super().keyPressEvent(event)

    def run_occlusion_cleaning(self):
        print("Running occlusion cleaning...")
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        h, w = self.camerainfo_msg.height, self.camerainfo_msg.width
        extrinsics_3x4 = self.extrinsics[:3, :]

        cleaner = LiDARCleaner(K, extrinsics_3x4, self.points_xyz.T, h, w)
        self.occlusion_mask = cleaner.run()

        num_removed = np.sum(~self.occlusion_mask)
        print(f"Occlusion cleaning finished. {num_removed} points identified as occluded.")

        self.progress_bar.setVisible(False)
        self.redraw_points()

    def _set_normal_rotation_visible(self, visible: bool):
        for lbl, sb in zip(self._normal_rot_labels, self._normal_rot_spinboxes):
            lbl.setVisible(visible)
            sb.setVisible(visible)

    def _on_opacity_changed(self, value: float):
        """Opacity only needs a repaint — no rebuild."""
        if self.point_cloud_item is not None:
            self.point_cloud_item.update_opacity(value)
        if self.second_point_cloud_item is not None:
            self.second_point_cloud_item.update_opacity(value)

    def _on_view_params_changed(self):
        self.redraw_points()

    def _on_colorization_mode_changed(self):
        self._set_normal_rotation_visible(
            self.colorization_mode_combo.currentText() == "Surface Normals"
        )
        self._update_min_max_values_for_mode()
        self.redraw_points()

    def _update_min_max_values_for_mode(self):
        """Update min/max spinbox values based on current colorization mode."""
        if not hasattr(self, "points_xyz") or self.points_xyz.shape[0] == 0:
            return

        colorization_mode = self.colorization_mode_combo.currentText()
        if colorization_mode == "Distance":
            if hasattr(self, "valid_indices") and len(self.valid_indices) > 0:
                tvec = self.extrinsics[:3, 3]
                points_cam = (self.extrinsics[:3, :3] @ self.points_xyz.T).T + tvec
                valid_points_cam = points_cam[self.valid_indices]
                distances = np.linalg.norm(valid_points_cam, axis=1)
                min_dist, max_dist = np.quantile(distances, [0.01, 0.99])
                self.min_value_spinbox.setValue(min_dist)
                self.max_value_spinbox.setValue(max_dist)
        elif colorization_mode == "LiDAR Edge":
            if hasattr(self, "valid_indices") and len(self.valid_indices) > 0 and hasattr(self, "points_proj_valid"):
                tvec = self.extrinsics[:3, 3]
                pts_cam = (self.extrinsics[:3, :3] @ self.points_xyz[self.valid_indices].T).T + tvec
                scores = self._compute_lidar_edge_scores(pts_cam, self.points_proj_valid)
                self.min_value_spinbox.setValue(0.0)
                self.max_value_spinbox.setValue(float(np.quantile(scores, 0.95)))
        elif colorization_mode == "Surface Normals":
            # No scalar range needed — RGB mapped directly from normal vector
            self.min_value_spinbox.setEnabled(False)
            self.max_value_spinbox.setEnabled(False)
            return
        else:
            if hasattr(self, "intensities") and self.intensities.size > 0:
                min_i, max_i = np.quantile(self.intensities, [0.01, 0.90])
                self.min_value_spinbox.setValue(min_i)
                self.max_value_spinbox.setValue(max_i)
        self.min_value_spinbox.setEnabled(True)
        self.max_value_spinbox.setEnabled(True)

    def _compute_lidar_edge_scores(
        self, pts_cam: np.ndarray, pts_2d: np.ndarray, k: int = 10
    ) -> np.ndarray:
        """Depth gradient in projected 2D space.
        For each point, max(|depth_i - depth_j| / pixel_dist) over k 2D neighbours.
        Spikes at depth discontinuities (box edges against a farther floor).
        """
        n = len(pts_2d)
        if n < 2:
            return np.zeros(n)
        k = min(k, n - 1)
        depths = pts_cam[:, 2]
        tree = KDTree(pts_2d)
        dists_2d, idx = tree.query(pts_2d, k=k + 1)        # (N, k+1)
        neighbor_depths = depths[idx[:, 1:]]                # (N, k)
        depth_diffs = np.abs(depths[:, None] - neighbor_depths)
        pixel_dists = np.maximum(dists_2d[:, 1:], 1.0)     # clamp to ≥1 px
        return (depth_diffs / pixel_dists).max(axis=1)

    def _compute_normals(self, pts_3d: np.ndarray, k: int = 15) -> np.ndarray:
        """Estimate surface normals via PCA on local k-neighbourhoods."""
        n = len(pts_3d)
        if n < k + 1:
            return np.zeros((n, 3))
        k = min(k, n - 1)
        tree = KDTree(pts_3d)
        _, idx = tree.query(pts_3d, k=k + 1)
        neighbors = pts_3d[idx[:, 1:]]                          # (N, k, 3)
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k  # (N, 3, 3)
        _, eigvecs = np.linalg.eigh(cov)                         # (N, 3, 3)
        return eigvecs[:, :, 0]                                  # (N, 3) smallest eigenvector

    # ------------------------------------------------------------------ #
    #  Auto refinement                                                     #
    # ------------------------------------------------------------------ #

    def run_auto_refinement(self):
        if not hasattr(self, "points_xyz") or self.points_xyz.shape[0] == 0:
            self.refine_status_label.setText("No point cloud loaded.")
            return
        if not hasattr(self, "valid_indices") or len(self.valid_indices) == 0:
            self.refine_status_label.setText("No projected points — check extrinsics.")
            return

        self.auto_refine_button.setEnabled(False)
        self.refine_status_label.setText("Running…")
        self._edge_map_cache = None  # invalidate cache when user triggers a new run

        import threading
        threading.Thread(target=self._refinement_worker, daemon=True).start()

    def _refinement_worker(self):
        try:
            new_extrinsics, status = self._run_edge_alignment()
            self._refinement_done.emit((new_extrinsics, status))
        except Exception as e:
            self._refinement_done.emit((None, f"Error: {e}"))

    def _apply_refinement_result(self, payload):
        new_extrinsics, status = payload
        self.auto_refine_button.setEnabled(True)
        self.refine_status_label.setText(status)
        if new_extrinsics is not None:
            self.extrinsics = new_extrinsics
            self.extrinsics_updated.emit(np.linalg.inv(self.extrinsics))
            self.redraw_points()
            self._highlight_export_button()

    def _compute_edge_map(self) -> np.ndarray:
        """Canny edges on the current image, Gaussian-blurred for a smooth cost landscape."""
        if self._edge_map_cache is not None:
            return self._edge_map_cache

        img = self.original_cv_image
        if img is None:
            h, w = self.camerainfo_msg.height, self.camerainfo_msg.width
            return np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if self.is_rectification_enabled:
            K = np.array(self.camerainfo_msg.k).reshape(3, 3)
            D = np.array(self.camerainfo_msg.d)
            try:
                if self._is_fisheye:
                    gray = cv2.fisheye.undistortImage(gray, K, D[:4].reshape(4, 1), None, K)
                else:
                    gray = cv2.undistort(gray, K, D)
            except Exception:
                pass

        edges = cv2.Canny(gray, 30, 100)
        edge_map = cv2.GaussianBlur(edges.astype(np.float32), (0, 0), sigmaX=4.0)
        max_val = edge_map.max()
        if max_val > 0:
            edge_map /= max_val
        self._edge_map_cache = edge_map
        return edge_map

    @staticmethod
    def _bilinear_sample(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of img at float (x, y) locations."""
        x = np.clip(pts[:, 0], 0, img.shape[1] - 2)
        y = np.clip(pts[:, 1], 0, img.shape[0] - 2)
        x0, y0 = x.astype(np.int32), y.astype(np.int32)
        dx, dy = x - x0, y - y0
        return (
            img[y0,     x0    ] * (1 - dx) * (1 - dy)
            + img[y0,     x0 + 1] * dx       * (1 - dy)
            + img[y0 + 1, x0    ] * (1 - dx) * dy
            + img[y0 + 1, x0 + 1] * dx       * dy
        )

    def _run_edge_alignment(self):
        from scipy.optimize import minimize

        K = np.array(self.camerainfo_msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(self.camerainfo_msg.d, dtype=np.float64)
        is_fisheye = self._is_fisheye
        is_rectified = self.is_rectification_enabled

        # When rectification is active the edge map is in undistorted pixel space,
        # so projection inside the objective must also produce undistorted coords.
        # The simplest way is to project with zero distortion (equivalent to
        # projecting onto the rectified image plane with the same K).
        D_proj = np.zeros_like(D) if is_rectified else D
        d4_proj = D_proj[:4].reshape(4, 1)

        edge_map = self._compute_edge_map()
        h, w = edge_map.shape

        # Select top LiDAR edge points visible in current projection
        valid_pts_3d = self.points_xyz[self.valid_indices]
        pts_cam = (self.extrinsics[:3, :3] @ valid_pts_3d.T).T + self.extrinsics[:3, 3]
        edge_scores = self._compute_lidar_edge_scores(pts_cam, self.points_proj_valid)

        n_pts = min(6000, len(self.valid_indices))
        top_idx = np.argsort(edge_scores)[-n_pts:]
        edge_pts_3d = valid_pts_3d[top_idx].astype(np.float64)

        # Initial params: T_lidar→camera [tx, ty, tz, rx, ry, rz] in radians
        T_lc0 = np.linalg.inv(self.extrinsics)
        t0 = T_lc0[:3, 3].copy()
        r0 = Rotation.from_matrix(T_lc0[:3, :3]).as_euler("xyz")
        x0 = np.concatenate([t0, r0])

        # Bounds: ±5 cm translation, ±3° rotation around initial guess
        # Rotation tighter than translation — small angle errors cause large pixel shifts
        dr = np.radians(3)
        bounds = (
            [(t - 0.05, t + 0.05) for t in t0]
            + [(r - dr, r + dr) for r in r0]
        )

        n_edge = len(edge_pts_3d)

        def objective(params):
            tx, ty, tz, rx, ry, rz = params
            R = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
            R_cl = R.T
            t_cl = -(R.T @ np.array([tx, ty, tz]))
            rvec = cv2.Rodrigues(R_cl.astype(np.float64))[0].ravel()
            try:
                if is_fisheye:
                    pts2d, _ = cv2.fisheye.projectPoints(
                        edge_pts_3d.reshape(-1, 1, 3),
                        rvec, t_cl, K, d4_proj,
                    )
                else:
                    pts2d, _ = cv2.projectPoints(
                        edge_pts_3d, rvec, t_cl, K, D_proj
                    )
            except Exception:
                return 1.0

            pts2d = pts2d.reshape(-1, 2)
            in_bounds = (
                (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w - 1)
                & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h - 1)
            )
            # Always divide by n_edge so off-screen points score 0.
            # Using a shrinking denominator was rewarding transforms that pushed
            # points off-screen when the surviving subset had high edge scores.
            all_scores = np.zeros(n_edge)
            if in_bounds.sum() > 0:
                all_scores[in_bounds] = self._bilinear_sample(
                    edge_map, pts2d[in_bounds]
                )
            return -float(all_scores.mean())

        score_before = -objective(x0)

        result = minimize(
            objective, x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-9, "gtol": 1e-6},
        )

        tx, ty, tz, rx, ry, rz = result.x
        R_opt = Rotation.from_euler("xyz", [rx, ry, rz]).as_matrix()
        T_lc_opt = np.eye(4)
        T_lc_opt[:3, :3] = R_opt
        T_lc_opt[:3, 3] = [tx, ty, tz]
        new_extrinsics = np.linalg.inv(T_lc_opt)

        score_after = -result.fun   # result.fun is already the final objective value
        status = (
            f"Done — edge score {score_before:.4f} → {score_after:.4f}"
            + (" ✓" if score_after > score_before else " (no improvement)")
        )
        return new_extrinsics, status

    def _update_calibrate_button_highlight(self):
        # Need at least 4 master LiDAR to camera correspondences for calibration
        # LiDAR-to-LiDAR correspondences are used to solve for the second LiDAR transform
        master_cam_corr_count = len(self.correspondences)
        lidar_lidar_corr_count = len(self.lidar_to_lidar_correspondences)

        if self.has_second_pointcloud:
            # Need both types of correspondences for dual LiDAR mode
            if master_cam_corr_count >= 4 and lidar_lidar_corr_count >= 3:
                self.calibrate_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            else:
                self.calibrate_button.setStyleSheet(self.default_button_style)
        else:
            # Single LiDAR mode
            if master_cam_corr_count >= 4:
                self.calibrate_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            else:
                self.calibrate_button.setStyleSheet(self.default_button_style)

    def _highlight_export_button(self):
        """Highlight export button when calibration data changes."""
        self.export_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)

    def _update_confirm_button_state(self):
        if len(self.current_3d_selection) > 0:
            self.confirm_3d_button.setStyleSheet(UIStyles.HIGHLIGHT_BUTTON)
            self.confirm_3d_button.setText("Confirm 3D Selection (Enter)")
        else:
            self.confirm_3d_button.setStyleSheet(self.default_button_style)
            self.confirm_3d_button.setText("Confirm 3D Selection")

    def toggle_selection_mode(self, checked):
        if checked:
            if self.has_second_pointcloud:
                corr_mode = self.correspondence_mode_combo.currentText()
                if corr_mode == "Master LiDAR ↔ Camera":
                    self.selection_mode = "wait_for_2d_click"
                    self.add_corr_button.setText("1. Click on 2D Image Point")
                else:  # Second LiDAR ↔ Master LiDAR
                    self.selection_mode = "wait_for_second_lidar_click"
                    self.add_corr_button.setText("1. Click on Second LiDAR Point")
            else:
                self.selection_mode = "wait_for_2d_click"
                self.add_corr_button.setText("1. Click on 2D Image Point")
            self.view.setDragMode(QGraphicsView.NoDrag)
        else:
            self.reset_selection_mode()

    def eventFilter(self, source, event):
        if (
            source is self.view.viewport()
            and event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            if self.selection_mode == "wait_for_2d_click":
                self.handle_2d_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_3d_clicks":
                self.handle_3d_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_second_lidar_click":
                self.handle_second_lidar_point_selection(event.pos())
                return True
            elif self.selection_mode == "wait_for_master_lidar_clicks":
                self.handle_master_lidar_point_selection(event.pos())
                return True
        return super().eventFilter(source, event)

    def handle_2d_point_selection(self, pos):
        self.clear_temp_markers()
        scene_pos = self.view.mapToScene(pos)
        self.selected_2d_point = (scene_pos.x(), scene_pos.y())
        self.draw_cross_marker(scene_pos, QColor(Colors.CORRESPONDENCE_2D))
        self.selection_mode = "wait_for_3d_clicks"
        self.add_corr_button.setText("2. Click on LiDAR Point(s)")
        self.confirm_3d_button.setVisible(True)

    def handle_3d_point_selection(self, pos):
        if self.kdtree is None:
            return
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        if idx in self.selected_3d_items_map:
            item_to_remove = self.selected_3d_items_map.pop(idx)
            if item_to_remove in self.scene.items():
                self.scene.removeItem(item_to_remove)
            self.current_3d_selection.remove(item_to_remove)
        else:
            point_2d = self.points_proj_valid[idx]
            point_size = self.point_size_spinbox.value()
            item = QGraphicsEllipseItem(
                point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size
            )
            item.setPen(QPen(QColor(Colors.CORRESPONDENCE_3D), 2))
            item.setBrush(QBrush(QColor(Colors.CORRESPONDENCE_3D)))
            item.setData(0, idx)
            self.scene.addItem(item)
            self.current_3d_selection.append(item)
            self.selected_3d_items_map[idx] = item
        self._update_confirm_button_state()

    def handle_second_lidar_point_selection(self, pos):
        """Handle selection of a point from the second LiDAR point cloud."""
        if self.second_kdtree is None:
            return
        self.clear_temp_markers()
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.second_kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        # Store selected second LiDAR point
        self.selected_second_lidar_point = self.second_points_proj_valid[idx]
        self.selected_second_lidar_3d_idx = self.second_valid_indices[idx]

        # Draw marker on selected point
        point_2d = self.second_points_proj_valid[idx]
        self.draw_cross_marker(
            QPointF(point_2d[0], point_2d[1]), QColor(255, 0, 255)
        )  # Magenta for second LiDAR

        # Move to next mode
        self.selection_mode = "wait_for_master_lidar_clicks"
        self.add_corr_button.setText("2. Click on Master LiDAR Point(s)")
        self.confirm_3d_button.setVisible(True)

    def handle_master_lidar_point_selection(self, pos):
        """Handle selection of points from the master LiDAR point cloud for LiDAR-to-LiDAR correspondence."""
        if self.kdtree is None:
            return
        scene_pos = self.view.mapToScene(pos)
        dist, idx = self.kdtree.query([scene_pos.x(), scene_pos.y()], k=1)
        if dist > self.point_size_spinbox.value() * 1.5:
            return

        if idx in self.selected_3d_items_map:
            item_to_remove = self.selected_3d_items_map.pop(idx)
            if item_to_remove in self.scene.items():
                self.scene.removeItem(item_to_remove)
            self.current_3d_selection.remove(item_to_remove)
        else:
            point_2d = self.points_proj_valid[idx]
            point_size = self.point_size_spinbox.value()
            item = QGraphicsEllipseItem(
                point_2d[0] - point_size / 2, point_2d[1] - point_size / 2, point_size, point_size
            )
            item.setPen(QPen(QColor(0, 255, 0), 2))  # Green for master LiDAR
            item.setBrush(QBrush(QColor(0, 255, 0)))
            item.setData(0, idx)
            self.scene.addItem(item)
            self.current_3d_selection.append(item)
            self.selected_3d_items_map[idx] = item
        self._update_confirm_button_state()

    def finalize_correspondence(self):
        if not self.current_3d_selection:
            self.reset_selection_mode()
            return

        if hasattr(self, "selected_second_lidar_point"):
            # LiDAR-to-LiDAR correspondence
            selected_valid_indices = [item.data(0) for item in self.current_3d_selection]
            original_indices = [self.valid_indices[i] for i in selected_valid_indices]
            mean_3d_point = np.mean(self.points_xyz[original_indices], axis=0)

            # Store correspondence between second LiDAR point and master LiDAR points
            second_lidar_3d = self.second_points_xyz[self.selected_second_lidar_3d_idx]
            self.lidar_to_lidar_correspondences[tuple(second_lidar_3d)] = {
                "master_3d_mean": mean_3d_point,
                "master_3d_points_indices": original_indices,
                "second_lidar_index": self.selected_second_lidar_3d_idx,
            }
        elif self.selected_2d_point is not None:
            # Camera-to-LiDAR correspondence
            selected_valid_indices = [item.data(0) for item in self.current_3d_selection]
            original_indices = [self.valid_indices[i] for i in selected_valid_indices]
            mean_3d_point = np.mean(self.points_xyz[original_indices], axis=0)
            self.correspondences[self.selected_2d_point] = {
                "3d_mean": mean_3d_point,
                "3d_points_indices": original_indices,
            }
        else:
            self.reset_selection_mode()
            return

        self.update_corr_list()
        self.reset_selection_mode()
        self._update_calibrate_button_highlight()
        self._highlight_export_button()

    def reset_calibration_state(self):
        self.correspondences = {}
        self.lidar_to_lidar_correspondences = {}
        self.update_corr_list()
        self.extrinsics = np.copy(self.initial_extrinsics)
        self.second_lidar_transform = np.eye(4)
        self.occlusion_mask = None
        self.second_occlusion_mask = None
        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self.update_results_display()
        self.extrinsics_updated.emit(np.linalg.inv(self.extrinsics))
        self.clear_all_highlighting()
        self.reset_selection_mode()
        self._update_calibrate_button_highlight()

    def reset_selection_mode(self):
        self.selection_mode = None
        self.selected_2d_point = None
        # Clear second LiDAR selection attributes
        if hasattr(self, "selected_second_lidar_point"):
            delattr(self, "selected_second_lidar_point")
        if hasattr(self, "selected_second_lidar_3d_idx"):
            delattr(self, "selected_second_lidar_3d_idx")
        self.clear_temp_markers()
        for item in self.current_3d_selection:
            if item.scene():
                self.scene.removeItem(item)
        self.current_3d_selection = []
        self.selected_3d_items_map = {}
        self.add_corr_button.setChecked(False)
        self.add_corr_button.setText("Add Correspondence")
        self.confirm_3d_button.setVisible(False)
        self._update_confirm_button_state()
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

    def display_image(self):
        # Decode/load the original image if not already done
        if self.original_cv_image is None:
            encoding = getattr(self.image_msg, "encoding", "")
            is_compressed = not encoding or encoding not in self.ros_utils.name_to_dtypes
            if is_compressed:
                np_arr = np.frombuffer(self.image_msg.data, np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                self.original_cv_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                self.original_cv_image = self.ros_utils.image_to_numpy(self.image_msg)

            # Convert BGR to RGB if needed
            if not is_compressed and "bgr" in encoding:
                self.original_cv_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)

        # Apply rectification if enabled
        if self.is_rectification_enabled:
            # Convert back to BGR for OpenCV undistort function
            bgr_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_RGB2BGR)
            rectified_bgr = self.rectify_image(bgr_image)
            self.cv_image = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        else:
            self.cv_image = self.original_cv_image.copy()

        # Update UI and display
        h, w, c = self.cv_image.shape
        self.image_res_label.setText(f"{w} x {h}")

        # Update the background image in-place to avoid deleting other scene items
        q_image = QImage(self.cv_image.data, w, h, 3 * w, QImage.Format_RGB888)
        new_pixmap = QPixmap.fromImage(q_image)
        if self._bg_pixmap_item is None:
            self._bg_pixmap_item = self.scene.addPixmap(new_pixmap)
            self._bg_pixmap_item.setZValue(-1)
        else:
            self._bg_pixmap_item.setPixmap(new_pixmap)

        # Remove stale point cloud items — they will be recreated by project_pointcloud
        if self.point_cloud_item is not None:
            try:
                self.scene.removeItem(self.point_cloud_item)
            except RuntimeError:
                pass
            self.point_cloud_item = None
        if self.second_point_cloud_item is not None:
            try:
                self.scene.removeItem(self.second_point_cloud_item)
            except RuntimeError:
                pass
            self.second_point_cloud_item = None

        # Re-project point cloud with the updated image
        self.project_pointcloud()
        self._apply_lidar_visibility()

        # Refresh grid overlay if active (distortion state may have changed)
        if self._grid_visible:
            self._clear_intrinsics_grid()
            self._draw_intrinsics_grid()

    def redraw_points(self):
        self.project_pointcloud(self.extrinsics, re_read_cloud=False)
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self._apply_lidar_visibility()

    def project_pointcloud(self, extrinsics=None, re_read_cloud=True):
        if extrinsics is not None:
            self.extrinsics = extrinsics
        self.clear_all_highlighting()
        if self.point_cloud_item is not None and self.point_cloud_item.scene():
            self.scene.removeItem(self.point_cloud_item)
        self.point_cloud_item = None

        if re_read_cloud:
            cloud_arr = self.ros_utils.pointcloud2_to_structured_array(self.pointcloud_msg)
            valid_mask = (
                np.isfinite(cloud_arr["x"])
                & np.isfinite(cloud_arr["y"])
                & np.isfinite(cloud_arr["z"])
            )
            cloud_arr = cloud_arr[valid_mask]
            self.points_xyz = np.vstack([cloud_arr["x"], cloud_arr["y"], cloud_arr["z"]]).T
            intensity_field = "intensity" if "intensity" in cloud_arr.dtype.names else "reflectivity"
            self.intensities = cloud_arr[intensity_field].astype(np.float32)
            if self.intensities.size > 0:
                # Set initial min/max values based on current colorization mode
                self._update_min_max_values_for_mode()

        if not hasattr(self, "points_xyz") or self.points_xyz.shape[0] == 0:
            return

        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(self.extrinsics[:3, :3])
        tvec = self.extrinsics[:3, 3]
        d_raw = np.array(self.camerainfo_msg.d)
        if self._is_fisheye:
            d4 = d_raw[:4].reshape(4, 1)
            points_proj_cv, _ = cv2.fisheye.projectPoints(
                self.points_xyz.reshape(-1, 1, 3), rvec, tvec, K, d4
            )
            if self.is_rectification_enabled:
                points_proj_cv = cv2.fisheye.undistortPoints(points_proj_cv, K, d4, P=K)
        else:
            points_proj_cv, _ = cv2.projectPoints(self.points_xyz, rvec, tvec, K, d_raw)
            if self.is_rectification_enabled:
                points_proj_cv = cv2.undistortPoints(
                    points_proj_cv.reshape(-1, 1, 2), K, d_raw, P=K
                )
        points_proj_cv = points_proj_cv.reshape(-1, 2)
        points_cam = (self.extrinsics[:3, :3] @ self.points_xyz.T).T + tvec
        z_cam = points_cam[:, 2]

        mask = (
            (z_cam > 0)
            & (points_proj_cv[:, 0] >= 0)
            & (points_proj_cv[:, 0] < self.camerainfo_msg.width)
            & (points_proj_cv[:, 1] >= 0)
            & (points_proj_cv[:, 1] < self.camerainfo_msg.height)
        )

        if self.occlusion_mask is not None and len(self.occlusion_mask) == len(mask):
            mask = np.logical_and(mask, self.occlusion_mask)

        self.valid_indices = np.where(mask)[0]
        self.points_proj_valid = points_proj_cv[self.valid_indices]
        self.intensities_valid = self.intensities[self.valid_indices]
        if self.points_proj_valid.shape[0] == 0:
            self.kdtree = None
            return

        self.kdtree = KDTree(self.points_proj_valid)
        cmap = cm.get_cmap(self.colormap_combo.currentText())

        colorization_mode = self.colorization_mode_combo.currentText()
        if colorization_mode == "Distance":
            valid_points_cam = points_cam[self.valid_indices]
            distances = np.linalg.norm(valid_points_cam, axis=1)
            min_val, max_val = self.min_value_spinbox.value(), self.max_value_spinbox.value()
            norm_values = np.clip((distances - min_val) / (max_val - min_val + 1e-6), 0, 1)
            colors = cmap(norm_values)
        elif colorization_mode == "LiDAR Edge":
            valid_points_cam = points_cam[self.valid_indices]
            scores = self._compute_lidar_edge_scores(valid_points_cam, self.points_proj_valid)
            min_val, max_val = self.min_value_spinbox.value(), self.max_value_spinbox.value()
            norm_values = np.clip((scores - min_val) / (max_val - min_val + 1e-6), 0, 1)
            colors = cmap(norm_values)
        elif colorization_mode == "Surface Normals":
            normals = self._compute_normals(self.points_xyz[self.valid_indices])
            roll_deg, pitch_deg, yaw_deg = (
                sb.value() for sb in self._normal_rot_spinboxes
            )
            if roll_deg != 0.0 or pitch_deg != 0.0 or yaw_deg != 0.0:
                R_color = Rotation.from_euler(
                    "xyz", [roll_deg, pitch_deg, yaw_deg], degrees=True
                ).as_matrix()
                normals = normals @ R_color.T
            rgb = np.abs(normals).clip(0, 1)
            colors = np.column_stack([rgb, np.ones(len(rgb))])
        else:
            # Intensity (default)
            min_val, max_val = self.min_value_spinbox.value(), self.max_value_spinbox.value()
            norm_values = np.clip(
                (self.intensities_valid - min_val) / (max_val - min_val + 1e-6), 0, 1
            )
            colors = cmap(norm_values)

        colors[:, 3] = 0.8
        img_size = (self.camerainfo_msg.height, self.camerainfo_msg.width)
        pt_sz = self.point_size_spinbox.value()
        opacity = self.opacity_spinbox.value()

        if self.point_cloud_item is not None and self.point_cloud_item.scene():
            self.point_cloud_item.update_data(
                self.points_proj_valid, colors, pt_sz, opacity, img_size=img_size
            )
        else:
            self.point_cloud_item = PointCloudItem(
                self.points_proj_valid, colors, pt_sz,
                opacity=opacity, img_size=img_size
            )
            self.scene.addItem(self.point_cloud_item)

    def project_second_pointcloud(self, transform=None):
        """Project the second point cloud using the current transformation."""
        if not self.has_second_pointcloud:
            return

        if transform is not None:
            self.second_lidar_transform = transform

        # Remove existing second point cloud
        if self.second_point_cloud_item is not None and self.second_point_cloud_item.scene():
            self.scene.removeItem(self.second_point_cloud_item)
        self.second_point_cloud_item = None

        # Extract second point cloud data
        cloud_arr = self.ros_utils.pointcloud2_to_structured_array(self.second_pointcloud_msg)
        valid_mask = (
            np.isfinite(cloud_arr["x"]) & np.isfinite(cloud_arr["y"]) & np.isfinite(cloud_arr["z"])
        )
        cloud_arr = cloud_arr[valid_mask]
        second_points_xyz = np.vstack([cloud_arr["x"], cloud_arr["y"], cloud_arr["z"]]).T
        second_intensities = cloud_arr["intensity"]

        if second_points_xyz.shape[0] == 0:
            return

        # Transform second LiDAR points to master LiDAR frame
        second_points_homogeneous = np.hstack(
            [second_points_xyz, np.ones((second_points_xyz.shape[0], 1))]
        )
        transformed_points = (self.second_lidar_transform @ second_points_homogeneous.T).T[:, :3]

        # Project using master LiDAR to camera transform
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)
        rvec, _ = cv2.Rodrigues(self.extrinsics[:3, :3])
        tvec = self.extrinsics[:3, 3]
        d_raw = np.array(self.camerainfo_msg.d)
        if self._is_fisheye:
            d4 = d_raw[:4].reshape(4, 1)
            points_proj_cv, _ = cv2.fisheye.projectPoints(
                transformed_points.reshape(-1, 1, 3), rvec, tvec, K, d4
            )
            if self.is_rectification_enabled:
                points_proj_cv = cv2.fisheye.undistortPoints(points_proj_cv, K, d4, P=K)
        else:
            points_proj_cv, _ = cv2.projectPoints(transformed_points, rvec, tvec, K, d_raw)
            if self.is_rectification_enabled:
                points_proj_cv = cv2.undistortPoints(
                    points_proj_cv.reshape(-1, 1, 2), K, d_raw, P=K
                )
        points_proj_cv = points_proj_cv.reshape(-1, 2)

        # Transform to camera coordinates to check visibility
        points_cam = (self.extrinsics[:3, :3] @ transformed_points.T).T + tvec
        z_cam = points_cam[:, 2]

        # Filter points within image bounds and in front of camera
        mask = (
            (z_cam > 0)
            & (points_proj_cv[:, 0] >= 0)
            & (points_proj_cv[:, 0] < self.camerainfo_msg.width)
            & (points_proj_cv[:, 1] >= 0)
            & (points_proj_cv[:, 1] < self.camerainfo_msg.height)
        )

        # Apply occlusion mask if available
        if self.second_occlusion_mask is not None and len(self.second_occlusion_mask) == len(mask):
            mask = np.logical_and(mask, self.second_occlusion_mask)

        self.second_valid_indices = np.where(mask)[0]
        self.second_points_proj_valid = points_proj_cv[self.second_valid_indices]
        self.second_intensities_valid = second_intensities[self.second_valid_indices]
        self.second_points_xyz = second_points_xyz  # Store original coordinates

        if self.second_points_proj_valid.shape[0] == 0:
            self.second_kdtree = None
            return

        self.second_kdtree = KDTree(self.second_points_proj_valid)

        # Use different colormap for second point cloud (e.g., warm colors vs cool colors)
        second_cmap = cm.get_cmap("plasma")  # Different from master point cloud colormap

        # Color by intensity with different range
        min_val = np.quantile(self.second_intensities_valid, 0.01)
        max_val = np.quantile(self.second_intensities_valid, 0.90)
        norm_values = np.clip(
            (self.second_intensities_valid - min_val) / (max_val - min_val + 1e-6), 0, 1
        )

        colors = second_cmap(norm_values)
        colors[:, 3] = 0.7  # Slightly more transparent to distinguish from master

        if self.second_point_cloud_item is not None and self.second_point_cloud_item.scene():
            self.second_point_cloud_item.update_data(
                self.second_points_proj_valid, colors,
                self.point_size_spinbox.value(), self.opacity_spinbox.value()
            )
        else:
            self.second_point_cloud_item = PointCloudItem(
                self.second_points_proj_valid, colors, self.point_size_spinbox.value()
            )
            self.scene.addItem(self.second_point_cloud_item)

    def update_corr_list(self):
        self.corr_list_widget.clear()

        # Add master LiDAR to camera correspondences
        for p2d, corr_data in self.correspondences.items():
            p3d = corr_data["3d_mean"]
            item_text = f"Cam ({p2d[0]:.1f}, {p2d[1]:.1f}) ↔ Master ({p3d[0]:.2f}, {p3d[1]:.2f}, {p3d[2]:.2f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ("master_cam", p2d))
            self.corr_list_widget.addItem(item)

        # Add LiDAR-to-LiDAR correspondences
        for second_3d, corr_data in self.lidar_to_lidar_correspondences.items():
            master_3d = corr_data["master_3d_mean"]
            item_text = f"Second ({second_3d[0]:.2f}, {second_3d[1]:.2f}, {second_3d[2]:.2f}) ↔ Master ({master_3d[0]:.2f}, {master_3d[1]:.2f}, {master_3d[2]:.2f})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, ("lidar_lidar", second_3d))
            self.corr_list_widget.addItem(item)

    def delete_correspondence(self):
        current_item = self.corr_list_widget.currentItem()
        if current_item:
            corr_data = current_item.data(Qt.UserRole)
            if corr_data[0] == "master_cam":
                p2d_key = corr_data[1]
                if p2d_key in self.correspondences:
                    del self.correspondences[p2d_key]
            elif corr_data[0] == "lidar_lidar":
                second_3d_key = corr_data[1]
                if second_3d_key in self.lidar_to_lidar_correspondences:
                    del self.lidar_to_lidar_correspondences[second_3d_key]
            self.update_corr_list()
            self.clear_all_highlighting()
            self._update_calibrate_button_highlight()
            self._highlight_export_button()

    def highlight_from_list(self, current_item, previous_item):
        self.clear_all_highlighting()
        if not current_item:
            return

        corr_data = current_item.data(Qt.UserRole)
        point_size = self.point_size_spinbox.value()

        if corr_data[0] == "master_cam":
            # Highlight master LiDAR to camera correspondence
            p2d_key = corr_data[1]
            corr = self.correspondences.get(p2d_key)
            if not corr:
                return
            self.draw_cross_marker(
                QPointF(p2d_key[0], p2d_key[1]), QColor(Colors.CORRESPONDENCE_3D)
            )
            original_to_valid_idx_map = {
                orig_idx: valid_idx for valid_idx, orig_idx in enumerate(self.valid_indices)
            }
            for original_point_idx in corr["3d_points_indices"]:
                valid_idx = original_to_valid_idx_map.get(original_point_idx)
                if valid_idx is not None and valid_idx < len(self.points_proj_valid):
                    point_2d = self.points_proj_valid[valid_idx]
                    item = QGraphicsEllipseItem(
                        point_2d[0] - point_size / 2,
                        point_2d[1] - point_size / 2,
                        point_size,
                        point_size,
                    )
                    item.setPen(QPen(QColor(Colors.CORRESPONDENCE_3D), 2))
                    item.setBrush(QBrush(QColor(Colors.CORRESPONDENCE_3D)))
                    self.scene.addItem(item)
                    self.highlighted_3d_items.append(item)

        elif corr_data[0] == "lidar_lidar":
            # Highlight LiDAR-to-LiDAR correspondence
            second_3d_key = corr_data[1]
            corr = self.lidar_to_lidar_correspondences.get(second_3d_key)
            if not corr:
                return

            # Highlight second LiDAR point (if visible)
            second_idx = corr["second_lidar_index"]
            if hasattr(self, "second_valid_indices") and hasattr(self, "second_points_proj_valid"):
                second_valid_idx_map = {
                    orig_idx: valid_idx
                    for valid_idx, orig_idx in enumerate(self.second_valid_indices)
                }
                second_valid_idx = second_valid_idx_map.get(second_idx)
                if second_valid_idx is not None and second_valid_idx < len(
                    self.second_points_proj_valid
                ):
                    point_2d = self.second_points_proj_valid[second_valid_idx]
                    self.draw_cross_marker(
                        QPointF(point_2d[0], point_2d[1]), QColor(255, 0, 255)
                    )  # Magenta

            # Highlight master LiDAR points
            original_to_valid_idx_map = {
                orig_idx: valid_idx for valid_idx, orig_idx in enumerate(self.valid_indices)
            }
            for original_point_idx in corr["master_3d_points_indices"]:
                valid_idx = original_to_valid_idx_map.get(original_point_idx)
                if valid_idx is not None and valid_idx < len(self.points_proj_valid):
                    point_2d = self.points_proj_valid[valid_idx]
                    item = QGraphicsEllipseItem(
                        point_2d[0] - point_size / 2,
                        point_2d[1] - point_size / 2,
                        point_size,
                        point_size,
                    )
                    item.setPen(QPen(QColor(0, 255, 0), 2))  # Green
                    item.setBrush(QBrush(QColor(0, 255, 0)))
                    self.scene.addItem(item)
                    self.highlighted_3d_items.append(item)

    def draw_cross_marker(self, center, color):
        pen = QPen(color, 2)
        size = 10
        l1 = self.scene.addLine(center.x() - size, center.y(), center.x() + size, center.y(), pen)
        l2 = self.scene.addLine(center.x(), center.y() - size, center.x(), center.y() + size, pen)
        self.temp_2d_marker.extend([l1, l2])

    def clear_all_highlighting(self):
        self.clear_temp_markers()
        self.clear_highlighted_3d_points()

    def clear_temp_markers(self):
        for item in self.temp_2d_marker:
            if item.scene():
                self.scene.removeItem(item)
        self.temp_2d_marker = []

    def clear_highlighted_3d_points(self):
        for item in self.highlighted_3d_items:
            if item.scene():
                self.scene.removeItem(item)
        self.highlighted_3d_items = []
        for item in self.current_3d_selection:
            if item.scene():
                self.scene.removeItem(item)
        self.current_3d_selection = []
        self.selected_3d_items_map = {}

    def run_calibration(self):
        if len(self.correspondences) < AppConstants.MIN_CORRESPONDENCES:
            return
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        ransac_method_str = self.pnp_solver_combo.currentText()
        pnp_flag = {"SQPnP": cv2.SOLVEPNP_SQPNP, "Iterative": cv2.SOLVEPNP_ITERATIVE}.get(
            ransac_method_str
        )
        lsq_method = self.lsq_method_combo.currentText()
        K = np.array(self.camerainfo_msg.k).reshape(3, 3)

        # Express 2D correspondences in undistorted pinhole pixel space so the
        # pinhole-model PnP in calibrate() is valid regardless of lens distortion.
        raw_pts = list(self.correspondences.keys())
        und_pts = self._undistort_points_for_calib(raw_pts)
        und_map = {p2d: tuple(map(float, u)) for p2d, u in zip(raw_pts, und_pts)}
        print(
            f"[run_calibration] {len(raw_pts)} correspondences | "
            f"rectification={'ON' if self.is_rectification_enabled else 'OFF'} | "
            f"fisheye={self._is_fisheye} | "
            f"distortion_model={getattr(self.camerainfo_msg, 'distortion_model', '')!r} | "
            f"D={np.array(self.camerainfo_msg.d).tolist()}"
        )

        if self.has_second_pointcloud and len(self.lidar_to_lidar_correspondences) >= 3:
            # Dual LiDAR calibration
            master_cam_corr = [
                (und_map[p2d], corr["3d_mean"]) for p2d, corr in self.correspondences.items()
            ]
            new_extrinsics, new_second = calibration.calibrate_dual_lidar(
                master_cam_corr, self.lidar_to_lidar_correspondences, K, pnp_flag, lsq_method
            )
        else:
            # Single LiDAR calibration
            calib_corr = [
                (und_map[p2d], corr["3d_mean"]) for p2d, corr in self.correspondences.items()
            ]
            new_extrinsics = calibration.calibrate(calib_corr, K, pnp_flag, lsq_method)
            new_second = None

        self.progress_bar.setVisible(False)

        if new_extrinsics is None:
            # Calibration failed — keep the previous transform instead of zeroing out.
            print("[run_calibration] Calibration failed — keeping previous extrinsics.")
            self.results_label.setText(
                "Calibration failed (see console). Keeping previous values."
            )
            return

        self.extrinsics = new_extrinsics
        if new_second is not None:
            self.second_lidar_transform = new_second

        self.project_pointcloud()
        if self.has_second_pointcloud:
            self.project_second_pointcloud()
        self.update_results_display()
        self.extrinsics_updated.emit(np.linalg.inv(self.extrinsics))
        self._highlight_export_button()

    def update_results_display(self):
        self.results_label.setText("Calibration parameters updated")

    def export_calibration(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "", "YAML Files (*.yaml)"
        )
        if file_path:
            t = self.extrinsics[:3, 3]
            q = Rotation.from_matrix(self.extrinsics[:3, :3]).as_quat()
            with open(file_path, "w") as f:
                f.write("# LiDAR-Camera Extrinsic Calibration (T_camera_lidar)\n")
                f.write(f"translation:\n  x: {t[0]:.8f}\n  y: {t[1]:.8f}\n  z: {t[2]:.8f}\n")
                f.write(
                    f"rotation:\n  x: {q[0]:.8f}\n  y: {q[1]:.8f}\n  z: {q[2]:.8f}\n  w: {q[3]:.8f}\n"
                )
            print(f"Calibration saved to {file_path}")

    def view_calibration_results(self):
        """Emit signal to view calibration results in main window."""
        self.export_button.setStyleSheet(self.default_button_style)

        if self.has_second_pointcloud:
            calibration_results = {
                "mode": "dual_lidar",
                "master_to_camera": self.extrinsics,
                "master_to_second_lidar": self.second_lidar_transform,
            }
        else:
            calibration_results = {"mode": "single_lidar", "master_to_camera": self.extrinsics}

        self.calibration_completed.emit(calibration_results)

    # ------------------------------------------------------------------ #
    #  Public update API (called by MainWindow)                           #
    # ------------------------------------------------------------------ #

    def update_extrinsics(self, T_lidar_cam: np.ndarray):
        """Update the displayed transform from the extrinsics panel."""
        self.extrinsics = np.linalg.inv(T_lidar_cam)
        self._edge_map_cache = None
        self.project_pointcloud(re_read_cloud=False)
        self._apply_lidar_visibility()
        if self._grid_visible:
            self._clear_intrinsics_grid()
            self._draw_intrinsics_grid()

    def update_intrinsics(self, camerainfo_msg):
        """Update camera intrinsics (called when intrinsics panel changes)."""
        self.camerainfo_msg = camerainfo_msg
        self._is_fisheye = getattr(camerainfo_msg, "distortion_model", "").lower() in (
            "fisheye", "kannala_brandt", "equidistant"
        )
        has_distortion = self.has_significant_distortion()
        self.btn_rectify.setEnabled(has_distortion)
        self._edge_map_cache = None
        self.display_image()

    # ------------------------------------------------------------------ #
    #  Overlay buttons                                                     #
    # ------------------------------------------------------------------ #

    _OVERLAY_STYLE = (
        "QPushButton { background: rgba(50,50,50,210); border-radius: 4px;"
        " color: #ddd; font-weight: bold; font-size: 12px; border: 1px solid #555; }"
        " QPushButton:checked { background: rgba(214,72,20,220); color: white; border-color: #e96030; }"
        " QPushButton:hover { background: rgba(80,80,80,220); }"
        " QPushButton:checked:hover { background: rgba(240,100,40,230); }"
    )

    def _setup_overlay_buttons(self):
        self._overlay = QWidget(self.view)
        ol = QVBoxLayout(self._overlay)
        ol.setContentsMargins(4, 4, 4, 4)
        ol.setSpacing(4)

        self.btn_rectify = QPushButton("R")
        self.btn_rectify.setCheckable(True)
        self.btn_rectify.setFixedSize(32, 32)
        self.btn_rectify.setToolTip("Rectify Image")
        self.btn_rectify.setStyleSheet(self._OVERLAY_STYLE)

        self.btn_show_lidar = QPushButton("L")
        self.btn_show_lidar.setCheckable(True)
        self.btn_show_lidar.setChecked(True)
        self.btn_show_lidar.setFixedSize(32, 32)
        self.btn_show_lidar.setToolTip("Show LiDAR Points")
        self.btn_show_lidar.setStyleSheet(self._OVERLAY_STYLE)

        self.btn_show_grid = QPushButton("G")
        self.btn_show_grid.setCheckable(True)
        self.btn_show_grid.setFixedSize(32, 32)
        self.btn_show_grid.setToolTip("Show Intrinsics Grid")
        self.btn_show_grid.setStyleSheet(self._OVERLAY_STYLE)

        ol.addWidget(self.btn_rectify)
        ol.addWidget(self.btn_show_lidar)
        ol.addWidget(self.btn_show_grid)
        self._overlay.adjustSize()
        self._overlay.raise_()

        # Set initial rectification state
        has_distortion = self.has_significant_distortion()
        self.btn_rectify.setEnabled(has_distortion)
        if has_distortion:
            self.is_rectification_enabled = True
            self.btn_rectify.blockSignals(True)
            self.btn_rectify.setChecked(True)
            self.btn_rectify.blockSignals(False)

        self.btn_rectify.toggled.connect(self.toggle_rectification)
        self.btn_show_lidar.toggled.connect(self._on_lidar_toggle)
        self.btn_show_grid.toggled.connect(self._on_grid_toggle)

        self._reposition_overlay()

    def _reposition_overlay(self):
        if not hasattr(self, "_overlay"):
            return
        hint = self._overlay.sizeHint()
        self._overlay.setGeometry(
            self.view.width() - hint.width() - 8, 8, hint.width(), hint.height()
        )
        self._overlay.raise_()

    def _apply_lidar_visibility(self):
        """Sync point cloud item visibility with the L overlay button state."""
        if not hasattr(self, "btn_show_lidar"):
            return
        visible = self.btn_show_lidar.isChecked()
        if self.point_cloud_item:
            self.point_cloud_item.setVisible(visible)
        if self.second_point_cloud_item:
            self.second_point_cloud_item.setVisible(visible)

    def _on_lidar_toggle(self, checked: bool):
        if self.point_cloud_item:
            self.point_cloud_item.setVisible(checked)
        if self.second_point_cloud_item:
            self.second_point_cloud_item.setVisible(checked)

    # ------------------------------------------------------------------ #
    #  Intrinsics grid overlay                                             #
    # ------------------------------------------------------------------ #

    def _on_grid_toggle(self, checked: bool):
        self._grid_visible = checked
        if checked:
            self._draw_intrinsics_grid()
        else:
            self._clear_intrinsics_grid()

    def _draw_intrinsics_grid(self):
        """Draw a 9×9 projected grid at 1 m depth as QGraphicsLine items."""
        if not hasattr(self, "camerainfo_msg"):
            return
        K = np.array(self.camerainfo_msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(self.camerainfo_msg.d, dtype=np.float64)
        is_fisheye = self._is_fisheye
        rectified = self.is_rectification_enabled

        h, w = self.camerainfo_msg.height, self.camerainfo_msg.width
        n_lines, n_pts, depth = 9, 40, 1.0
        fx, fy = K[0, 0], K[1, 1]
        x_half = (w / 2) / fx * 1.05
        y_half = (h / 2) / fy * 1.05

        xs = np.linspace(-x_half, x_half, n_lines)
        ys = np.linspace(-y_half, y_half, n_lines)
        along_x = np.linspace(-x_half, x_half, n_pts)
        along_y = np.linspace(-y_half, y_half, n_pts)
        D_use = np.zeros_like(D) if rectified else D
        rvec = tvec = np.zeros(3)

        pen = QPen(QColor(0, 220, 0), 1)

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
                x1, y1 = pts2d[i]
                x2, y2 = pts2d[i + 1]
                if max(abs(x1), abs(x2)) < w * 3 and max(abs(y1), abs(y2)) < h * 3:
                    item = self.scene.addLine(x1, y1, x2, y2, pen)
                    item.setZValue(0.5)
                    self._grid_items.append(item)

        for y_val in ys:
            pts = np.column_stack([along_x, np.full(n_pts, y_val), np.full(n_pts, depth)])
            draw_poly(project(pts))
        for x_val in xs:
            pts = np.column_stack([np.full(n_pts, x_val), along_y, np.full(n_pts, depth)])
            draw_poly(project(pts))

    def _clear_intrinsics_grid(self):
        for item in self._grid_items:
            if item.scene():
                self.scene.removeItem(item)
        self._grid_items = []
