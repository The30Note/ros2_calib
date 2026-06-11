#!/usr/bin/env python3
"""Camera intrinsic calibration from AprilGrid (tag36h11) images.

Usage:
    python calibrate.py <image_dir> --context   # fisheye lens
    python calibrate.py <image_dir> --zoom       # standard / plumb_bob lens
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from pupil_apriltags import Detector


@dataclass
class TagDetection:
    """Mirrors the pupil-apriltags Detection interface for remapped detections."""
    tag_id: int
    corners: np.ndarray   # (4, 2) float32, in original distorted image coords
    center: np.ndarray    # (2,)   float32

# ── AprilGrid board parameters (Tangram Vision 6×6, tag36h11) ───────────────
TAG_FAMILY        = "tag36h11"
TAG_SIZE          = 0.1121    # meters, physical side length of one tag
TAG_SPACING_RATIO = 0.25      # gap / tag_size  →  gap = 0.02802 m
GRID_COLS         = 6
GRID_ROWS         = 6
TAG_ID_OFFSET     = 0
STEP              = TAG_SIZE * (1.0 + TAG_SPACING_RATIO)  # 0.140125 m

# Tag IDs to skip regardless of detection quality.
# Grid layout (row-major, 6×6):
#   0  1  2  3  4  5
#   6  7  8  9 10 11
#  12 13 14 15 16 17
#  18 19 20 21 22 23
#  24 25 26 27 28 29
#  30 31 32 33 34 35
# Keep only the inner 4×4 block (rows 1–4, cols 1–4); ignore all outer-ring tags.
# Middle 4×4 IDs:  7  8  9 10
#                 13 14 15 16
#                 19 20 21 22
#                 25 26 27 28
_keep = {row * GRID_COLS + col for row in range(1, 5) for col in range(1, 5)}
IGNORED_TAG_IDS: set = set(range(GRID_ROWS * GRID_COLS)) - _keep
# ────────────────────────────────────────────────────────────────────────────

MIN_TAGS_DEFAULT  = 4   # per image; 4 tags = 16 point pairs


# ── Prior-calibration undistort helpers ─────────────────────────────────────

def load_prior_calibration(yaml_path: Path):
    """Return (K, D) from an existing calibration YAML, or (None, None)."""
    if not yaml_path.exists():
        return None, None
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    raw = data["distortion_coefficients"]["data"]
    model = data.get("distortion_model") or data["distortion_coefficients"].get("model", "plumb_bob")
    if model == "fisheye":
        D = np.array(raw, dtype=np.float64).reshape(4, 1)
    else:
        D = np.array(raw, dtype=np.float64)
    return K, D


def setup_undistort(K: np.ndarray, D: np.ndarray, image_size: tuple):
    """Build fisheye remap tables. Returns (K_new, map1, map2).

    Uses K itself as K_new: cv2.fisheye.estimateNewCameraMatrixForUndistortRectify
    is broken on some OpenCV builds (returns fx≈0), and for detection we just need
    distortion removed — preserving the original focal length is fine.
    """
    W, H = image_size
    K_new = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (W, H), cv2.CV_16SC2
    )
    return K_new, map1, map2


def remap_detections(dets, K_new: np.ndarray, K: np.ndarray, D: np.ndarray):
    """Map corners from undistorted image space back to original distorted image coords."""
    result = []
    for d in dets:
        corners_u = d.corners.astype(np.float64)
        # Normalise to undistorted rays
        x_n = (corners_u[:, 0] - K_new[0, 2]) / K_new[0, 0]
        y_n = (corners_u[:, 1] - K_new[1, 2]) / K_new[1, 1]
        pts_3d = np.column_stack([x_n, y_n, np.ones(4)]).reshape(-1, 1, 3)
        # Re-project through fisheye model back to original pixel coords
        corners_d, _ = cv2.fisheye.projectPoints(
            pts_3d, np.zeros((3, 1)), np.zeros((3, 1)), K, D
        )
        corners_d = corners_d.reshape(-1, 2).astype(np.float32)
        result.append(TagDetection(d.tag_id, corners_d, corners_d.mean(axis=0)))
    return result


def tag_world_corners(tag_id: int) -> np.ndarray:
    """Return the 4 world-space corners of a tag (Z=0 board plane).

    Corner order matches apriltag3: top-left, top-right, bottom-right,
    bottom-left.  Origin is the top-left corner of tag 0.
    """
    local_id = tag_id - TAG_ID_OFFSET
    row, col  = divmod(local_id, GRID_COLS)
    ox, oy    = col * STEP, row * STEP
    s         = TAG_SIZE
    # apriltag3 corner order: top-left, top-right, bottom-right, bottom-left
    return np.array([
        [ox,     oy,     0.0],  # top-left
        [ox + s, oy,     0.0],  # top-right
        [ox + s, oy + s, 0.0],  # bottom-right
        [ox,     oy + s, 0.0],  # bottom-left
    ], dtype=np.float32)


def filter_detections(all_dets):
    """Split raw detections into (accepted, ignored) lists."""
    valid_ids = range(TAG_ID_OFFSET, TAG_ID_OFFSET + GRID_ROWS * GRID_COLS)
    accepted = [d for d in all_dets if d.tag_id in valid_ids and d.tag_id not in IGNORED_TAG_IDS]
    ignored  = [d for d in all_dets if d.tag_id in valid_ids and d.tag_id in IGNORED_TAG_IDS]
    return accepted, ignored


def build_point_arrays(detections, gray: np.ndarray):
    """Convert a list of Detection objects to (obj_pts, img_pts) float32 arrays."""
    if not detections:
        return None, None
    obj_pts = np.vstack([tag_world_corners(d.tag_id) for d in detections])
    img_pts = np.vstack([d.corners for d in detections]).astype(np.float32)
    return obj_pts, img_pts


def write_debug_image(path: Path, img: np.ndarray, accepted, ignored, debug_dir: Path):
    """Save annotated image: accepted tags green, ignored tags red."""
    out = img.copy()

    for dets, color in [(accepted, (0, 210, 0)), (ignored, (0, 60, 255))]:
        for d in dets:
            pts = d.corners.astype(int).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, 3)
            cv2.putText(out, str(d.tag_id), tuple(d.center.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    label = f"green={len(accepted)} accepted   red={len(ignored)} ignored"
    cv2.putText(out, label, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,   0,   0), 4)
    cv2.putText(out, label, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / path.name), out)


_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


def _refine_corners(det, gray):
    """Snap corner positions to the nearest sub-pixel edge in the full-res image.

    Half-res detection often reports corners 1-4 px from the true tag corner
    (it locks onto the middle of the border pixel rather than the outer edge);
    cornerSubPix uses the local gradient to pull each corner to the correct spot.
    """
    pts = det.corners.astype(np.float32).reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), _SUBPIX_CRITERIA)
    refined = pts.reshape(-1, 2)
    # Drop the refinement if it pulled any corner more than 4 px — that means
    # cornerSubPix latched onto an adjacent feature rather than this tag's corner.
    if np.max(np.linalg.norm(refined - det.corners, axis=1)) > 4.0:
        return det
    return TagDetection(det.tag_id, refined.astype(np.float32), refined.mean(axis=0).astype(np.float32))


def detect_with_fallback(detectors, gray, min_keep):
    """Run the primary detector; only fall back to coarser detectors if needed.

    Coarser (decimated) passes find more tags on small/tilted boards but place
    corners less precisely. Using them only when the primary pass is short of
    `min_keep` keeps corner quality high on easy images and recovers tags on
    hard ones. Newly-found tags from fallback passes get cornerSubPix-refined
    against the full-res image so their corners aren't half-res-quantized.
    """
    primary, *fallbacks = detectors
    out  = list(primary.detect(gray))
    seen = {d.tag_id for d in out}

    accepted = [d for d in out
                if d.tag_id < TAG_ID_OFFSET + GRID_ROWS * GRID_COLS
                and d.tag_id not in IGNORED_TAG_IDS]
    if len(accepted) >= min_keep:
        return out

    for det in fallbacks:
        for d in det.detect(gray):
            if d.tag_id in seen:
                continue
            seen.add(d.tag_id)
            out.append(_refine_corners(d, gray))
    return out


def load_images(image_dir: Path, detectors, min_tags: int,
                visualize: bool, debug_dir: Path | None,
                undistort_setup: tuple | None = None):
    """Walk image_dir, run detection, return usable (obj, img) lists + image_size."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"No images found in {image_dir}")

    all_obj, all_img, used_paths = [], [], []
    image_size = None

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [skip] could not read  {path.name}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (W, H)

        # Full-res detect; only fall back to half-res if too few accepted tags.
        all_dets = detect_with_fallback(detectors, gray, min_tags)

        # If a prior calibration is available and we still need more tags,
        # also detect on the undistorted image and merge in new tag IDs.
        if undistort_setup is not None:
            K_new, map1, map2, K_prior, D_prior, detectors_undist = undistort_setup
            undist   = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)
            und_dets = detect_with_fallback(detectors_undist, undist, min_tags)
            und_dets = remap_detections(und_dets, K_new, K_prior, D_prior)
            seen = {d.tag_id for d in all_dets}
            all_dets = all_dets + [d for d in und_dets if d.tag_id not in seen]

        accepted, filtered = filter_detections(all_dets)
        n_tags             = len(accepted)

        if debug_dir is not None:
            write_debug_image(path, img, accepted, filtered, debug_dir)

        if visualize:
            for d in accepted:
                cv2.polylines(img, [d.corners.astype(int).reshape(-1, 1, 2)],
                              True, (0, 255, 0), 2)
            cv2.imshow("detections", img)
            cv2.waitKey(300)

        if n_tags < min_tags:
            print(f"  [skip] {path.name}: {n_tags} accepted tags (need {min_tags},"
                  f" {len(filtered)} ignored)")
            continue

        obj_pts, img_pts = build_point_arrays(accepted, gray)
        all_obj.append(obj_pts)
        all_img.append(img_pts)
        used_paths.append(path)
        print(f"  [ok]   {path.name}: {n_tags} tags ({len(filtered)} ignored)")

    if visualize:
        cv2.destroyAllWindows()

    return all_obj, all_img, used_paths, image_size


# ── Fisheye calibration ──────────────────────────────────────────────────────

def calibrate_fisheye(all_obj, all_img, image_size):
    """Run cv2.fisheye.calibrate; returns (K, D, rvecs, tvecs, rms)."""
    # fisheye requires shape (N, 1, 3) / (N, 1, 2)
    obj_f = [o.reshape(-1, 1, 3) for o in all_obj]
    img_f = [i.reshape(-1, 1, 2) for i in all_img]

    W, H = image_size
    # Seed focal length from equidistant model: f = r_max / theta_max
    # Assuming ~185° FOV: theta_max ≈ pi/2, r_max ≈ min(W,H)/2
    f0 = (min(W, H) / 2.0) / (np.pi / 2.0)
    K = np.array([
        [f0,  0.0, W / 2.0],
        [0.0, f0,  H / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Small non-zero seed avoids the D=0 saddle point in the fisheye optimizer
    D = np.full((4, 1), 0.01, dtype=np.float64)

    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
        cv2.fisheye.CALIB_FIX_SKEW             |
        cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    )

    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_f, img_f, image_size, K, D,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
    )
    return K, D, rvecs, tvecs, rms


def per_image_errors_fisheye(all_obj, all_img, K, D, rvecs, tvecs, used_paths):
    print("\nPer-image reprojection errors (fisheye):")
    for obj, img, rvec, tvec, path in zip(all_obj, all_img, rvecs, tvecs, used_paths):
        proj, _ = cv2.fisheye.projectPoints(
            obj.reshape(-1, 1, 3), rvec, tvec, K, D
        )
        err = np.linalg.norm(img - proj.reshape(-1, 2), axis=1).mean()
        print(f"  {path.name}: {err:.4f} px")


class _InlineList(list):
    pass

def _inline_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

yaml.add_representer(_InlineList, _inline_list_representer)


def _round(v, decimals=8):
    return round(float(v), decimals)


def _projection_matrix(K):
    """Build the 3×4 projection matrix P for a monocular camera (Tx=0)."""
    return _InlineList([
        _round(K[0, 0]), 0.0, _round(K[0, 2]), 0.0,
        0.0, _round(K[1, 1]), _round(K[1, 2]), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ])


def save_fisheye(K, D, image_size, rms, out_path):
    d = D.flatten()
    result = {
        "image_width":  image_size[0],
        "image_height": image_size[1],
        "camera_name":  "context_main",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": _InlineList([_round(v) for row in K for v in row]),
        },
        "distortion_model": "fisheye",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(d),
            "data": _InlineList([_round(v) for v in d]),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": _InlineList([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": _projection_matrix(K),
        },
        "rms_reprojection_error_px": float(rms),
    }
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {out_path}")


# ── Plumb_bob (standard) calibration ────────────────────────────────────────

def calibrate_zoom(all_obj, all_img, image_size):
    """Run cv2.calibrateCamera; returns (K, dist, rvecs, tvecs, rms)."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None
    )
    return K, dist, rvecs, tvecs, rms


def per_image_errors_zoom(all_obj, all_img, K, dist, rvecs, tvecs, used_paths):
    print("\nPer-image reprojection errors (plumb_bob):")
    for obj, img, rvec, tvec, path in zip(all_obj, all_img, rvecs, tvecs, used_paths):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(img - proj.reshape(-1, 2), axis=1).mean()
        print(f"  {path.name}: {err:.4f} px")


def save_zoom(K, dist, image_size, rms, out_path):
    d = dist.flatten()
    result = {
        "image_width":  image_size[0],
        "image_height": image_size[1],
        "camera_name":  "zoom_main",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": _InlineList([_round(v) for row in K for v in row]),
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": len(d),
            "data": _InlineList([_round(v) for v in d]),
        },
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": _InlineList([1, 0, 0, 0, 1, 0, 0, 0, 1]),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": _projection_matrix(K),
        },
        "rms_reprojection_error_px": float(rms),
    }
    with open(out_path, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {out_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AprilGrid camera intrinsic calibration"
    )
    parser.add_argument("image_dir", type=Path,
                        help="Folder containing calibration images")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--context", action="store_true",
                      help="Fisheye / wide-angle lens (equidistant model)")
    mode.add_argument("--zoom",    action="store_true",
                      help="Standard / zoom lens (plumb_bob model)")

    parser.add_argument("--output",   type=Path,  default=None,
                        help="Output YAML path (default: calibration_<model>.yaml)")
    parser.add_argument("--initial-guess", type=Path, default=None,
                        help="Path to an intrinsics YAML used as a prior (--context only): "
                             "the images are undistorted before detection so more tags are "
                             "found and corners land more accurately. Pass a previous "
                             "calibration here to refine it.")
    parser.add_argument("--min-tags", type=int,   default=MIN_TAGS_DEFAULT,
                        help=f"Min detected tags per image (default: {MIN_TAGS_DEFAULT})")
    parser.add_argument("--visualize", action="store_true",
                        help="Display tag detections while processing")
    parser.add_argument("--debug-images", action="store_true",
                        help="Save annotated debug images to debug_images/ in the image folder")
    args = parser.parse_args()

    model_name = "fisheye" if args.context else "plumb_bob"
    out_path   = args.output or Path(f"calibration_{model_name}.yaml")

    print(f"\nCamera model : {model_name}")
    print(f"Image folder : {args.image_dir}")
    print(f"Output       : {out_path}\n")

    # Primary full-res detector + half-res fallback. Half-res catches small/tilted
    # tags the full-res quad finder misses, but its corners are less precise so we
    # only use it when full-res is short of min_tags (and refine its corners).
    detectors = [
        Detector(
            families=TAG_FAMILY, nthreads=2,
            quad_decimate=1.0, quad_sigma=0.8,
            refine_edges=1, decode_sharpening=0.5,
        ),
        Detector(
            families=TAG_FAMILY, nthreads=2,
            quad_decimate=2.0, quad_sigma=0.0,
            refine_edges=1, decode_sharpening=0.5,
        ),
    ]

    debug_dir = args.image_dir / "debug_images" if args.debug_images else None
    if debug_dir:
        print(f"Debug images  : {debug_dir}\n")

    # If an --initial-guess intrinsics YAML is provided, undistort before detection.
    # Straight-edged tags let the detector find more corners and place them better.
    undistort_setup = None
    if args.initial_guess is not None:
        if not args.context:
            sys.exit("--initial-guess is only supported with --context (fisheye).")
        if not args.initial_guess.exists():
            sys.exit(f"--initial-guess file not found: {args.initial_guess}")
        K_prior, D_prior = load_prior_calibration(args.initial_guess)
        if K_prior is not None:
            first_img = next(
                (p for p in sorted(args.image_dir.iterdir())
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}),
                None,
            )
            if first_img is not None:
                probe = cv2.imread(str(first_img))
                if probe is not None:
                    H_img, W_img = probe.shape[:2]
                    K_new, map1, map2 = setup_undistort(K_prior, D_prior, (W_img, H_img))
                    # On undistorted images edges are straight — use full-res, no blur,
                    # plus a half-res pass for small tags
                    detectors_undist = [
                        Detector(families=TAG_FAMILY, nthreads=2,
                                 quad_decimate=1.0, quad_sigma=0.0,
                                 refine_edges=1, decode_sharpening=0.25),
                        Detector(families=TAG_FAMILY, nthreads=2,
                                 quad_decimate=2.0, quad_sigma=0.0,
                                 refine_edges=1, decode_sharpening=0.25),
                    ]
                    undistort_setup = (K_new, map1, map2, K_prior, D_prior, detectors_undist)
                    print(f"Initial-guess intrinsics loaded from {args.initial_guess} — undistorting before detection\n")

    all_obj, all_img, used_paths, image_size = load_images(
        args.image_dir, detectors, args.min_tags, args.visualize, debug_dir, undistort_setup
    )

    print(f"\nUsable images: {len(used_paths)} / "
          f"{len(list(args.image_dir.iterdir()))}")

    if len(used_paths) < 3:
        sys.exit("Need at least 3 usable images for calibration.")

    print("Running calibration …")

    if args.context:
        K, D, rvecs, tvecs, rms = calibrate_fisheye(all_obj, all_img, image_size)
        print(f"\nOverall RMS reprojection error: {rms:.4f} px")
        per_image_errors_fisheye(all_obj, all_img, K, D, rvecs, tvecs, used_paths)
        save_fisheye(K, D, image_size, rms, out_path)
    else:
        K, dist, rvecs, tvecs, rms = calibrate_zoom(all_obj, all_img, image_size)
        print(f"\nOverall RMS reprojection error: {rms:.4f} px")
        per_image_errors_zoom(all_obj, all_img, K, dist, rvecs, tvecs, used_paths)
        save_zoom(K, dist, image_size, rms, out_path)


if __name__ == "__main__":
    main()
