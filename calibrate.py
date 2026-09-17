#!/usr/bin/env python3
"""Camera intrinsic calibration from AprilGrid (tag36h11) images.

Writes ROS camera_info YAMLs straight into the device-config layout:

    devices/<serial>/ip_camera_processing_cpp/
        context_camera_info.yaml        zoom_camera_info.yaml
        debug_context_camera_info.yaml  debug_zoom_camera_info.yaml

Usage:
    python calibrate.py --serial vss_00000041 --context <image_dir>
    python calibrate.py --serial vss_00000041 --zoom <image_dir>
    python calibrate.py --serial vss_00000041 --context <dir> --zoom <dir>
"""

import argparse
import atexit
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


_SUBPIX_WIN = 5


def _refine_corners(det, gray):
    """Snap corner positions to the nearest sub-pixel edge in the full-res image.

    Half-res detection often reports corners 1-4 px from the true tag corner
    (it locks onto the middle of the border pixel rather than the outer edge);
    cornerSubPix uses the local gradient to pull each corner to the correct spot.
    """
    h, w = gray.shape[:2]
    # cornerSubPix samples a (2*win+1)² neighbourhood around each corner and
    # raises an assertion if that window falls outside the image. Fisheye frames
    # routinely place a tag hard against the frame edge, so guard the bounds and
    # keep the raw corner rather than crashing the whole run.
    m = _SUBPIX_WIN + 2
    c = det.corners
    if (c[:, 0] < m).any() or (c[:, 0] >= w - m).any() or \
       (c[:, 1] < m).any() or (c[:, 1] >= h - m).any():
        return det if isinstance(det, TagDetection) else \
            TagDetection(det.tag_id, c.astype(np.float32), c.mean(axis=0).astype(np.float32))

    pts = c.astype(np.float32).reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(gray, pts, (_SUBPIX_WIN, _SUBPIX_WIN), (-1, -1), _SUBPIX_CRITERIA)
    refined = pts.reshape(-1, 2)
    # Drop the refinement if it pulled any corner more than 4 px — that means
    # cornerSubPix latched onto an adjacent feature rather than this tag's corner.
    if np.max(np.linalg.norm(refined - c, axis=1)) > 4.0:
        return det if isinstance(det, TagDetection) else \
            TagDetection(det.tag_id, c.astype(np.float32), c.mean(axis=0).astype(np.float32))
    return TagDetection(det.tag_id, refined.astype(np.float32), refined.mean(axis=0).astype(np.float32))


def release_detectors(detectors):
    """Defuse the pupil-apriltags destructor, which double-frees the tag family.

    Detector.__del__ calls tag36h11_destroy(family) and *then*
    apriltag_detector_destroy(), whose clear_families() runs quick_decode_uninit()
    on that already-freed family — a use-after-free that segfaults at interpreter
    exit (backtrace: quick_decode_uninit → apriltag_detector_clear_families →
    apriltag_detector_destroy). Emptying tag_families skips the first free and
    leaves apriltag_detector_destroy as the only cleanup path. The family struct
    itself (a few hundred bytes per detector) is leaked; the process is exiting.
    """
    for d in detectors:
        try:
            d.tag_families.clear()
        except AttributeError:
            pass


def detect_multi(detectors, gray):
    """Run every detector pass and merge detections by tag id (finest-wins).

    Each pass uses a different quad_decimate/quad_sigma and therefore sees a
    different subset of the board: large near tags resolve best at low
    decimation, while small/far/tilted tags only appear at high decimation or
    after blurring. On the fisheye set the union of all passes finds ~2× the tags
    of any single pass, which is what makes hard frames usable at all.

    `detectors` MUST be ordered so the pass most likely to give the *best corners*
    for a tag comes first (ascending quad_decimate). The first pass to report a
    tag id supplies its corners; later passes only add ids not yet seen. Every
    kept detection is cornerSubPix-refined against the full-res image so corners
    from decimated passes are not left quantized.
    """
    merged = {}
    for det in detectors:
        for d in det.detect(gray):
            if d.tag_id not in merged:
                merged[d.tag_id] = d
    return [_refine_corners(d, gray) for d in merged.values()]


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

        # Merge detections across all decimation/blur passes (see detect_multi).
        all_dets = detect_multi(detectors, gray)

        # If a prior calibration is available, also detect on the undistorted
        # image and merge in any tag IDs the distorted passes missed.
        if undistort_setup is not None:
            K_new, map1, map2, K_prior, D_prior, detectors_undist = undistort_setup
            undist   = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR)
            und_dets = detect_multi(detectors_undist, undist)
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


# ── Robust calibration (iterative outlier rejection) ────────────────────────

def _view_reproj_errors(all_obj, all_img, rvecs, tvecs, K, D, project):
    """Mean per-view reprojection error (px) for each view."""
    errs = []
    for obj, img, rvec, tvec in zip(all_obj, all_img, rvecs, tvecs):
        proj = project(obj, rvec, tvec, K, D)
        errs.append(float(np.linalg.norm(img - proj, axis=1).mean()))
    return errs


def _drop_outlier_tags(obj, img, paths, rvecs, tvecs, K, D, project,
                       pt_abs_thresh, pt_rel_thresh, min_tags):
    """Remove individual mislocalised tags; drop views left with too few.

    Returns (obj, img, paths, n_tags_dropped, n_views_dropped, thresh).

    Most frames are not globally bad — they carry one or two tags whose corners
    landed on the wrong feature (a coarse decimation pass whose corners survived
    the cornerSubPix guard, or a grazing-angle tag). On the context set the median
    corner error is 0.5px while p99 is 10px, so view-level rejection either keeps
    those corners or throws away 60 otherwise-good ones with them. A tag is dropped
    whole (all 4 corners) so the surviving points stay a set of complete tags.
    """
    thresh = None
    new_obj, new_img, new_paths = [], [], []
    n_tags = n_views = 0
    errs = [np.linalg.norm(im - project(ob, r, t, K, D), axis=1)
            for ob, im, r, t in zip(obj, img, rvecs, tvecs)]
    med = float(np.median(np.concatenate(errs)))
    thresh = max(pt_abs_thresh, pt_rel_thresh * med)

    for ob, im, path, e in zip(obj, img, paths, errs):
        keep = e.reshape(-1, 4).max(axis=1) <= thresh  # worst corner decides the tag
        n_tags += int((~keep).sum())
        if keep.sum() < min_tags:
            n_views += 1
            continue
        new_obj.append(ob.reshape(-1, 4, 3)[keep].reshape(-1, 3))
        new_img.append(im.reshape(-1, 4, 2)[keep].reshape(-1, 2))
        new_paths.append(path)
    return new_obj, new_img, new_paths, n_tags, n_views, thresh


def calibrate_robust(all_obj, all_img, used_paths, image_size, calib_fn, project,
                     abs_thresh=1.0, rel_thresh=2.5, min_frac=0.6, max_iters=6,
                     pt_abs_thresh=1.0, pt_rel_thresh=3.0, min_tags=2):
    """Calibrate, drop the worst-fitting tags and views, refit — until stable.

    Two levels of rejection run per pass. Individual tags whose worst corner
    exceeds max(pt_abs_thresh, pt_rel_thresh*median) are dropped first — a single
    mislocalised corner otherwise dominates the RMS and drags the intrinsics with
    it (on the context set this alone takes RMS from 2.32px to 0.35px without
    moving fx/fy). Then whole views whose mean error exceeds max(abs_thresh,
    rel_thresh*median) are dropped: motion blur or a near-degenerate board pose
    makes every corner in the frame unreliable. At least `min_frac` of the
    original views are always kept so a uniformly noisy set can't be whittled
    down to a degenerate handful. Returns
    (K, D, rvecs, tvecs, rms, kept_obj, kept_img, kept_paths, errs).
    """
    obj, img, paths = list(all_obj), list(all_img), list(used_paths)
    min_keep = max(4, int(round(len(all_obj) * min_frac)))
    result = None
    for it in range(max_iters):
        K, D, rvecs, tvecs, rms = calib_fn(obj, img, image_size)
        errs = _view_reproj_errors(obj, img, rvecs, tvecs, K, D, project)
        result = (K, D, rvecs, tvecs, rms, obj, img, paths, errs)

        obj2, img2, paths2, n_tags, n_views, pt_thresh = _drop_outlier_tags(
            obj, img, paths, rvecs, tvecs, K, D, project,
            pt_abs_thresh, pt_rel_thresh, min_tags)
        if n_tags and len(obj2) >= min_keep:
            print(f"  [reject] pass {it + 1}: rms {rms:.3f}px, dropping {n_tags} tag(s)"
                  f" > {pt_thresh:.2f}px"
                  + (f" and {n_views} view(s) left under {min_tags} tags" if n_views else ""))
            obj, img, paths = obj2, img2, paths2
            continue

        if len(obj) <= min_keep:
            break
        med = float(np.median(errs))
        thresh = max(abs_thresh, rel_thresh * med)
        keep = [i for i, e in enumerate(errs) if e <= thresh]
        if len(keep) == len(obj):
            break  # converged — nothing exceeds either threshold
        if len(keep) < min_keep:  # keep the min_keep lowest-error views
            keep = sorted(int(i) for i in np.argsort(errs)[:min_keep])
        dropped = [paths[i].name for i in range(len(paths)) if i not in keep]
        print(f"  [reject] pass {it + 1}: rms {rms:.3f}px, dropping "
              f"{len(obj) - len(keep)} view(s) > {thresh:.2f}px: {', '.join(dropped)}")
        obj = [obj[i] for i in keep]
        img = [img[i] for i in keep]
        paths = [paths[i] for i in keep]
    return result


def _project_zoom(obj, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def _project_fisheye(obj, rvec, tvec, K, D):
    proj, _ = cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K, D)
    return proj.reshape(-1, 2)


def print_view_errors(used_paths, errs, model_name):
    print(f"\nPer-image reprojection errors ({model_name}):")
    for path, err in zip(used_paths, errs):
        print(f"  {path.name}: {err:.4f} px")


# ── Fisheye calibration ──────────────────────────────────────────────────────

def _fisheye_seed_focals(image_size):
    """Candidate seed focal lengths, from the equidistant model f = r_max / θ_max.

    One seed is not enough: cv2.fisheye's InitExtrinsics is razor-sensitive to the
    guess and aborts with "(-215) fabs(norm_u1) > 0" on seeds that happen to make a
    view's init homography degenerate — on the context set the 185°-FOV seed (618.79)
    dies while 619.0 and 630.0 both succeed. Every seed that survives converges to
    the same optimum, so the seed only decides whether the solver starts at all.
    Spans 185° down to 60° FOV plus a few nearby values so a dead seed has neighbours.
    """
    r_max = min(image_size) / 2.0
    fovs = [185.0, 170.0, 150.0, 130.0, 120.0, 100.0, 90.0, 75.0, 60.0]
    return [r_max / np.deg2rad(fov / 2.0) for fov in fovs]


def calibrate_fisheye(all_obj, all_img, image_size):
    """Run cv2.fisheye.calibrate; returns (K, D, rvecs, tvecs, rms).

    Retries over a ladder of seed focal lengths, then without the intrinsic guess,
    and returns the first fit that survives (see _fisheye_seed_focals).
    """
    # fisheye requires shape (N, 1, 3) / (N, 1, 2), float64 for the solver
    obj_f = [o.reshape(-1, 1, 3).astype(np.float64) for o in all_obj]
    img_f = [i.reshape(-1, 1, 2).astype(np.float64) for i in all_img]

    W, H = image_size
    base_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    attempts = [(f0, base_flags | cv2.fisheye.CALIB_USE_INTRINSIC_GUESS)
                for f0 in _fisheye_seed_focals(image_size)]
    # Last resort: let OpenCV derive its own initial intrinsics.
    attempts.append((_fisheye_seed_focals(image_size)[0], base_flags))

    last_err = None
    for f0, flags in attempts:
        K = np.array([
            [f0,  0.0, W / 2.0],
            [0.0, f0,  H / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        # Small non-zero seed avoids the D=0 saddle point in the fisheye optimizer
        D = np.full((4, 1), 0.01, dtype=np.float64)
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_f, img_f, image_size, K, D, flags=flags, criteria=criteria,
            )
        except cv2.error as e:
            last_err = e
            continue
        return K, D, rvecs, tvecs, rms

    raise RuntimeError(
        "cv2.fisheye.calibrate failed for every seed focal length; the views are "
        f"likely degenerate (too few tags, or the board too small in frame). "
        f"Last OpenCV error: {last_err}"
    )


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


def _scale_K(K, image_size, target_size):
    """Scale an intrinsic matrix from image_size to target_size (both (w, h))."""
    sx = target_size[0] / image_size[0]
    sy = target_size[1] / image_size[1]
    K_s = K.copy()
    K_s[0, 0] *= sx
    K_s[0, 2] *= sx
    K_s[1, 1] *= sy
    K_s[1, 2] *= sy
    return K_s


def camera_info(K, D, image_size, camera_name, distortion_model, rms=None):
    """Build a ROS camera_info dict in the device-config format."""
    d = np.asarray(D).flatten()
    info = {
        "image_width":  int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_name":  camera_name,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": _InlineList([_round(v) for row in K for v in row]),
        },
        "distortion_model": distortion_model,
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
    }
    if rms is not None:
        info["rms_reprojection_error_px"] = float(rms)
    return info


def write_yaml(data, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  wrote {out_path}")


# ── Plumb_bob (standard) calibration ────────────────────────────────────────

def calibrate_zoom(all_obj, all_img, image_size):
    """Run cv2.calibrateCamera; returns (K, dist, rvecs, tvecs, rms)."""
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, image_size, None, None
    )
    return K, dist, rvecs, tvecs, rms


# ── Device-config output layout ──────────────────────────────────────────────

DEVICES_DIRNAME  = "devices"
CAMERA_SUBDIR    = "ip_camera_processing_cpp"

# Per-camera output settings. debug_size is the downscaled stream the device
# publishes alongside the full-resolution image; its intrinsics are the main
# ones scaled to that resolution (distortion coefficients are resolution
# independent, so they carry over unchanged).
CAMERAS = {
    "context": {
        "model":      "fisheye",
        "main_name":  "context_main",
        "debug_name": "context_debug",
        "debug_size": (320, 240),
    },
    "zoom": {
        "model":      "plumb_bob",
        "main_name":  "zoom_main",
        "debug_name": "zoom_debug",
        "debug_size": (352, 240),
    },
}


def normalize_serial(serial: str) -> str:
    """Accept 'vss_00000041', '00000041' or '41' → 'vss_00000041'."""
    s = serial.strip()
    if s.isdigit():
        return f"vss_{int(s):08d}"
    return s


def output_paths(device_dir: Path, cam: str) -> tuple[Path, Path]:
    """(main yaml, debug yaml) for one camera inside a device folder."""
    d = device_dir / CAMERA_SUBDIR
    return d / f"{cam}_camera_info.yaml", d / f"debug_{cam}_camera_info.yaml"


def confirm_overwrite(existing: list[Path], assume_yes: bool) -> bool:
    if not existing:
        return True
    print("\nWARNING: these calibration files already exist and will be overwritten:")
    for p in existing:
        print(f"  {p}")
    if assume_yes:
        print("(--yes given, overwriting)")
        return True
    try:
        answer = input("Continue? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    return answer in ("y", "yes")


# ── Per-camera calibration run ───────────────────────────────────────────────

def build_detectors(sharpening: float):
    """Multi-pass detector set, ordered finest-corners-first.

    No single decimation is enough: near/large tags resolve best at low
    decimation while small/far/tilted tags only appear at high decimation or
    after a blur. Merging the union of these passes (in detect_multi) roughly
    doubles the tags found per frame on the fisheye set and is what keeps hard
    frames usable. Ordered ascending so the pass with the best raw corners wins
    each tag id; all kept corners are then cornerSubPix-refined at full res.
    """
    def _dec(d, sigma=0.0):
        return Detector(
            families=TAG_FAMILY, nthreads=2,
            quad_decimate=d, quad_sigma=sigma,
            refine_edges=1, decode_sharpening=sharpening,
        )
    dets = [_dec(1.0), _dec(1.5), _dec(2.0), _dec(3.0), _dec(4.0), _dec(2.0, 0.8)]
    atexit.register(release_detectors, dets)
    return dets


def make_undistort_setup(image_dir: Path, guess_path: Path):
    """Prepare pre-detection undistortion from a prior intrinsics YAML.

    Straight-edged tags let the detector find more corners and place them better.
    Returns None if the prior cannot be used.
    """
    if not guess_path.exists():
        sys.exit(f"--initial-guess file not found: {guess_path}")
    K_prior, D_prior = load_prior_calibration(guess_path)
    if K_prior is None:
        return None
    first_img = next(
        (p for p in sorted(image_dir.iterdir())
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}),
        None,
    )
    if first_img is None:
        return None
    probe = cv2.imread(str(first_img))
    if probe is None:
        return None
    H_img, W_img = probe.shape[:2]
    K_new, map1, map2 = setup_undistort(K_prior, D_prior, (W_img, H_img))
    # Undistorted edges are straight, but tags span the same size range, so use
    # the same ascending multi-pass set as the distorted detectors.
    detectors_undist = build_detectors(0.25)
    print(f"Initial-guess intrinsics loaded from {guess_path} — undistorting before detection\n")
    return (K_new, map1, map2, K_prior, D_prior, detectors_undist)


def run_camera(cam: str, image_dir: Path, device_dir: Path, args):
    """Calibrate one camera and write its main + debug camera_info YAMLs."""
    spec = CAMERAS[cam]
    main_path, debug_path = output_paths(device_dir, cam)

    print(f"\n{'=' * 70}")
    print(f"Camera       : {cam}  ({spec['model']})")
    print(f"Image folder : {image_dir}")
    print(f"Output       : {main_path}")
    print(f"               {debug_path}\n")

    if not image_dir.is_dir():
        sys.exit(f"Image folder not found: {image_dir}")

    detectors = build_detectors(0.5)

    debug_dir = image_dir / "debug_images" if args.debug_images else None
    if debug_dir:
        print(f"Debug images : {debug_dir}\n")

    undistort_setup = None
    if args.initial_guess is not None and cam == "context":
        undistort_setup = make_undistort_setup(image_dir, args.initial_guess)

    all_obj, all_img, used_paths, image_size = load_images(
        image_dir, detectors, args.min_tags, args.visualize, debug_dir, undistort_setup
    )

    print(f"\nUsable images: {len(used_paths)} / {len(list(image_dir.iterdir()))}")
    if len(used_paths) < 3:
        sys.exit("Need at least 3 usable images for calibration.")

    print("Running calibration …")
    calib_fn, project = ((calibrate_fisheye, _project_fisheye) if cam == "context"
                         else (calibrate_zoom, _project_zoom))
    K, D, _rvecs, _tvecs, rms, _o, _i, kept_paths, errs = calibrate_robust(
        all_obj, all_img, used_paths, image_size, calib_fn, project,
    )
    print(f"\nKept {len(kept_paths)} / {len(used_paths)} views after rejection")
    print(f"Overall RMS reprojection error: {rms:.4f} px")
    print_view_errors(kept_paths, errs, spec["model"])

    write_yaml(
        camera_info(K, D, image_size, spec["main_name"], spec["model"], rms),
        main_path,
    )
    debug_size = spec["debug_size"]
    write_yaml(
        camera_info(_scale_K(K, image_size, debug_size), D, debug_size,
                    spec["debug_name"], spec["model"]),
        debug_path,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AprilGrid camera intrinsic calibration → device config layout"
    )
    parser.add_argument("--serial", required=True,
                        help="Device serial, e.g. vss_00000041")
    parser.add_argument("--context", type=Path, default=None, metavar="IMAGE_DIR",
                        help="Folder of context-camera images (fisheye model)")
    parser.add_argument("--zoom", type=Path, default=None, metavar="IMAGE_DIR",
                        help="Folder of zoom-camera images (plumb_bob model)")

    parser.add_argument("--devices-root", type=Path,
                        default=Path(__file__).resolve().parent / DEVICES_DIRNAME,
                        help="Root of the devices/ tree (default: ./devices next to this script)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Overwrite existing calibration files without asking")
    parser.add_argument("--initial-guess", type=Path, default=None,
                        help="Path to an intrinsics YAML used as a prior (--context only): "
                             "the images are undistorted before detection so more tags are "
                             "found and corners land more accurately. Pass a previous "
                             "calibration here to refine it.")
    parser.add_argument("--min-tags", type=int, default=MIN_TAGS_DEFAULT,
                        help=f"Min detected tags per image (default: {MIN_TAGS_DEFAULT})")
    parser.add_argument("--visualize", action="store_true",
                        help="Display tag detections while processing")
    parser.add_argument("--debug-images", action="store_true",
                        help="Save annotated debug images to debug_images/ in the image folder")
    args = parser.parse_args()

    requested = [(cam, d) for cam, d in (("context", args.context), ("zoom", args.zoom))
                 if d is not None]
    if not requested:
        parser.error("pass --context IMAGE_DIR and/or --zoom IMAGE_DIR")
    if args.initial_guess is not None and args.context is None:
        sys.exit("--initial-guess is only supported with --context (fisheye).")

    serial     = normalize_serial(args.serial)
    device_dir = args.devices_root / serial

    print(f"\nDevice       : {serial}")
    print(f"Device folder: {device_dir / CAMERA_SUBDIR}")

    existing = [p for cam, _ in requested
                for p in output_paths(device_dir, cam) if p.exists()]
    if not confirm_overwrite(existing, args.yes):
        sys.exit("Aborted — existing calibration left untouched.")

    (device_dir / CAMERA_SUBDIR).mkdir(parents=True, exist_ok=True)

    for cam, image_dir in requested:
        run_camera(cam, image_dir, device_dir, args)

    print(f"\nDone. Calibration written to {device_dir / CAMERA_SUBDIR}")


if __name__ == "__main__":
    main()
