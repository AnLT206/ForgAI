# -*- coding: utf-8 -*-
"""
src/realtime.py
Bộ máy phân tích kỹ thuật tập Gym thời gian thực (Bước 7 - ForgAI):
  1. MediaPipeRealtimeTracker: Trích xuất 13 khớp động học (wx, wy, wz, ix, iy, vis) ở RunningMode.VIDEO.
  2. CycleRepCounter: Máy trạng thái có trễ Hysteresis + EMA đếm chu kỳ rep cho 8 bài Nhóm A.
  3. ForgAIRealtimeScorer: Nạp checkpoint Bước E.2 và chấm điểm Đúng/Sai + giải thích lỗi tiếng Việt
     theo đúng pipeline src/features.py + src/dataset.py + src/models.py + src/rules.py (0 lệch Train-Serve).
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from configs.config import (
    CHECKPOINTS_DIR,
    EXERCISE_TO_ID,
    GROUP_A_EXERCISES,
    MIN_VISIBILITY_THRESHOLD,
    NUM_FRAMES,
    NUM_JOINTS,
    POSE_MODEL_PATH,
)
from src.constants import J_IDX, RAW_POINTS_13
from src.dataset import dynamic_strided_sampling
from src.features import (
    EXERCISE_BIOMECH_SPEC,
    _triplet_angle_series_deg,
    _visibility_weighted_side,
    compute_117d_biomechanical_mix,
    compute_biomechanical_tabular_features,
    compute_torso_scale,
    extract_anatomical_golden_matrix,
    handle_zero_frames,
    normalize_skeleton_geometry,
)
from src.models import build_model


# Ngưỡng động học chuẩn hóa từ thống kê thực tế trên 327 rep Đúng của Nhóm A.
# min_rom đủ nhạy để bắt trọn cả rep sai lỗi partial_rom & góc quay chính diện.
REP_CYCLE_SPECS: dict[str, dict[str, Any]] = {
    "pull up": {"driver": "elbow_angle", "mode": "min", "unit": "°", "start_ref": 158.0, "peak_ref": 81.0, "min_rom": 24.0, "rest_bounds": (105.0, 178.0)},
    "push-up": {"driver": "elbow_angle", "mode": "min", "unit": "°", "start_ref": 159.4, "peak_ref": 85.8, "min_rom": 24.0, "rest_bounds": (110.0, 178.0)},
    "leg extension": {"driver": "knee_angle", "mode": "max", "unit": "°", "start_ref": 87.3, "peak_ref": 163.2, "min_rom": 22.0, "rest_bounds": (45.0, 128.0)},
    "squat": {"driver": "knee_angle", "mode": "min", "unit": "°", "start_ref": 163.6, "peak_ref": 65.1, "min_rom": 24.0, "rest_bounds": (115.0, 178.0)},
    "chest fly machine": {"driver": "wrist_squeeze_dist", "mode": "min", "unit": "", "start_ref": 2.30, "peak_ref": 0.32, "min_rom": 0.60, "rest_bounds": (0.95, 3.00)},
    "hip thrust": {"driver": "hip_angle", "mode": "max", "unit": "°", "start_ref": 105.7, "peak_ref": 165.1, "min_rom": 20.0, "rest_bounds": (60.0, 140.0)},
    "t bar row": {"driver": "elbow_angle", "mode": "min", "unit": "°", "start_ref": 156.5, "peak_ref": 99.5, "min_rom": 18.0, "rest_bounds": (115.0, 178.0)},
    "leg raises": {"driver": "hip_angle", "mode": "min", "unit": "°", "start_ref": 168.2, "peak_ref": 55.0, "min_rom": 26.0, "rest_bounds": (115.0, 178.0)},
}

_DRIVER_TRIPLETS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "knee_angle": (
        (J_IDX["left_hip"], J_IDX["left_knee"], J_IDX["left_ankle"]),
        (J_IDX["right_hip"], J_IDX["right_knee"], J_IDX["right_ankle"]),
    ),
    "hip_angle": (
        (J_IDX["left_shoulder"], J_IDX["left_hip"], J_IDX["left_knee"]),
        (J_IDX["right_shoulder"], J_IDX["right_hip"], J_IDX["right_knee"]),
    ),
    "elbow_angle": (
        (J_IDX["left_shoulder"], J_IDX["left_elbow"], J_IDX["left_wrist"]),
        (J_IDX["right_shoulder"], J_IDX["right_elbow"], J_IDX["right_wrist"]),
    ),
}


@dataclass
class RepCompletedEvent:
    """Gói dữ liệu 1 rep vừa kết thúc chu kỳ để gửi sang ForgAIRealtimeScorer."""

    rep_index: int
    exercise: str
    pose_seq: np.ndarray      # (L, 13, 6)
    duration_sec: float
    observed_rom: float


class MediaPipeRealtimeTracker:
    """Bộ theo dõi khung xương 13 khớp theo từng frame ở chế độ RunningMode.VIDEO."""

    def __init__(self, conf_threshold: float = 0.25):
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=POSE_MODEL_PATH.read_bytes()),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=conf_threshold,
            min_pose_presence_confidence=conf_threshold,
            min_tracking_confidence=conf_threshold,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
        self._last_ts_ms = 0

    def __enter__(self) -> "MediaPipeRealtimeTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def process_frame(self, bgr_frame: np.ndarray, ts_ms: int | None = None) -> tuple[np.ndarray, bool]:
        """Trích xuất vector (13, 6) [wx, wy, wz, ix, iy, vis] và cờ có phát hiện người tập."""
        if self._landmarker is None:
            raise RuntimeError("MediaPipeRealtimeTracker đã bị đóng.")

        if ts_ms is None:
            ts_ms = int(time.perf_counter() * 1000.0)
        safe_ts = max(int(ts_ms), self._last_ts_ms + 1)
        self._last_ts_ms = safe_ts

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect_for_video(mp_img, safe_ts)

        joints_mat = np.zeros((NUM_JOINTS, 6), dtype=np.float32)
        if not (res.pose_landmarks and res.pose_world_landmarks):
            return joints_mat, False

        pl = res.pose_landmarks[0]
        wl = res.pose_world_landmarks[0]
        for j_out, mp_idx in enumerate(RAW_POINTS_13):
            wlm = wl[mp_idx]
            ilm = pl[mp_idx]
            vis = float(getattr(wlm, "visibility", getattr(ilm, "visibility", 1.0)) or 0.0)
            joints_mat[j_out] = [float(wlm.x), float(wlm.y), float(wlm.z), float(ilm.x), float(ilm.y), vis]
        return joints_mat, True


def compute_driver_signal_with_vis(frame_joints: np.ndarray, exercise: str) -> tuple[float, bool]:
    """Tính nhanh chỉ đúng 1 tín hiệu khớp chủ đạo của bài tập cùng cờ hiển thị rõ ràng."""
    spec = EXERCISE_BIOMECH_SPEC[exercise]
    driver = spec["driver"]
    coords_1 = frame_joints[None, :, :3].astype(np.float32)
    vis_1 = frame_joints[None, :, 5].astype(np.float32)
    nz = np.abs(frame_joints[:, :3]).sum(axis=-1) > 1e-6

    if driver == "wrist_squeeze_dist":
        req = (J_IDX["left_wrist"], J_IDX["right_wrist"], J_IDX["left_shoulder"], J_IDX["right_shoulder"])
        if not all(bool(nz[i] and vis_1[0, i] >= MIN_VISIBILITY_THRESHOLD) for i in req):
            return 0.0, False
        scale = float(np.linalg.norm(coords_1[0, J_IDX["left_shoulder"], :] - coords_1[0, J_IDX["right_shoulder"], :]))
        scale = max(scale * 1.45, compute_torso_scale(coords_1) if bool(nz[J_IDX["left_hip"]] and nz[J_IDX["right_hip"]]) else 0.15)
        dist = float(np.linalg.norm(coords_1[0, J_IDX["left_wrist"], :] - coords_1[0, J_IDX["right_wrist"], :]) / scale)
        return dist, True

    trip_l, trip_r = _DRIVER_TRIPLETS[driver]
    l_ok = all(bool(nz[i] and vis_1[0, i] >= MIN_VISIBILITY_THRESHOLD) for i in trip_l)
    r_ok = all(bool(nz[i] and vis_1[0, i] >= MIN_VISIBILITY_THRESHOLD) for i in trip_r)
    if not (l_ok or r_ok):
        return 0.0, False

    ang_l = _triplet_angle_series_deg(coords_1, trip_l)
    ang_r = _triplet_angle_series_deg(coords_1, trip_r)
    if l_ok and not r_ok:
        return float(ang_l[0]), True
    if r_ok and not l_ok:
        return float(ang_r[0]), True
    combined, _ = _visibility_weighted_side(ang_l, ang_r, vis_1, trip_l, trip_r)
    return float(combined[0]), True


def compute_instant_driver_signal(frame_joints: np.ndarray, exercise: str) -> float:
    """Trả về giá trị khớp chủ đạo tức thời của 1 frame."""
    val, _ = compute_driver_signal_with_vis(frame_joints, exercise)
    return val


def has_valid_exercise_posture(frame_joints: np.ndarray, exercise: str, strict_ready: bool = False) -> bool:
    """
    Kiểm tra tư thế khung xương tối thiểu theo đặc thù bài tập để loại bỏ tình huống
    người tập ngồi gần webcam chỉ thấy đầu/vai/tay rồi khua tay giả lập rep.
    """
    nz = np.abs(frame_joints[:, :3]).sum(axis=-1) > 1e-6
    vis = frame_joints[:, 5]
    iy = frame_joints[:, 4]

    hip_pts = [
        frame_joints[i, :3]
        for i in (J_IDX["left_hip"], J_IDX["right_hip"])
        if nz[i] and vis[i] >= MIN_VISIBILITY_THRESHOLD
    ]
    if not hip_pts:
        return False

    sh_pts = [
        frame_joints[i, :3]
        for i in (J_IDX["left_shoulder"], J_IDX["right_shoulder"])
        if nz[i] and vis[i] >= MIN_VISIBILITY_THRESHOLD
    ]
    if exercise != "leg extension" and not sh_pts:
        return False

    if exercise == "pull up":
        # Kéo xà: ít nhất một cổ tay nhìn thấy phải ở trên hoặc ngang vùng vai (không phải khua tay ngang bụng/ngực).
        max_dy = 0.06 if strict_ready else 0.12
        overhead_ok = False
        for wi, si in ((J_IDX["left_wrist"], J_IDX["left_shoulder"]), (J_IDX["right_wrist"], J_IDX["right_shoulder"])):
            if (
                nz[wi]
                and nz[si]
                and vis[wi] >= MIN_VISIBILITY_THRESHOLD
                and vis[si] >= MIN_VISIBILITY_THRESHOLD
                and float(iy[wi]) <= float(iy[si]) + max_dy
            ):
                overhead_ok = True
                break
        if not overhead_ok:
            return False

    if exercise == "push-up" and sh_pts and hip_pts:
        # Chống đẩy: thân người nằm nghiêng/ngang thay vì ngồi/đứng thẳng trục đứng trước webcam.
        d = np.mean(sh_pts, axis=0) - np.mean(hip_pts, axis=0)
        norm = float(np.linalg.norm(d))
        if norm > 1e-4 and (abs(float(d[1])) / norm) > 0.78:
            return False

    return True


class CycleRepCounter:
    """
    Máy trạng thái phân đoạn rep theo chu kỳ (IDLE -> IN_REP -> COMPLETED).
    Dùng lọc trung vị 3-frame + EMA làm mượt tín hiệu khớp dẫn, neo nghỉ thích ứng,
    xác thực tư thế chuẩn bị (khi bật require_ready_pose), khóa ổn định sau mất khung xương
    và hậu kiểm thời gian vận động thực tế để chống nhận nhầm rep giả khi khua tay.
    """

    def __init__(
        self,
        exercise: str,
        ema_alpha: float = 0.35,
        pre_roll_frames: int = 8,
        min_rep_frames: int = 10,
        min_rep_sec: float = 0.55,
        min_active_sec: float = 0.35,
        min_active_frames: int = 5,
        max_rep_sec: float = 12.0,
        max_lost_frames: int = 18,
        reacquire_warmup_frames: int = 8,
        min_valid_ratio: float = 0.75,
        min_inflection_frames: int = 2,
        require_ready_pose: bool = False,
        ready_min_frames: int = 5,
        ready_hold_sec: float = 0.50,
    ):
        if exercise not in REP_CYCLE_SPECS:
            raise ValueError(f"Bài tập không thuộc Nhóm A: {exercise}")
        self.ema_alpha = float(ema_alpha)
        self.pre_roll_frames = int(pre_roll_frames)
        self.min_rep_frames = int(min_rep_frames)
        self.min_rep_sec = float(min_rep_sec)
        self.min_active_sec = float(min_active_sec)
        self.min_active_frames = int(min_active_frames)
        self.max_rep_sec = float(max_rep_sec)
        self.max_lost_frames = int(max_lost_frames)
        self.reacquire_warmup_frames = int(reacquire_warmup_frames)
        self.min_valid_ratio = float(min_valid_ratio)
        self.min_inflection_frames = int(min_inflection_frames)
        self.require_ready_pose = bool(require_ready_pose)
        self.ready_min_frames = int(ready_min_frames)
        self.ready_hold_sec = float(ready_hold_sec)
        self.reset(exercise)

    def reset(self, exercise: str | None = None) -> None:
        if exercise is not None:
            if exercise not in REP_CYCLE_SPECS:
                raise ValueError(f"Bài tập không thuộc Nhóm A: {exercise}")
            self.exercise = exercise
        self.spec = REP_CYCLE_SPECS[self.exercise]
        self.state = "IDLE"
        self.rep_count = 0
        self.ema_val: float | None = None
        self.prev_ema: float | None = None
        self.rest_anchor = float(self.spec["start_ref"])
        self.last_progress = 0.0

        self._pre_roll: deque[tuple[np.ndarray, float, float]] = deque(maxlen=self.pre_roll_frames)
        self._rest_win: deque[float] = deque(maxlen=6)
        self._raw_win: deque[float] = deque(maxlen=3)
        self._rep_frames: list[np.ndarray] = []
        self._rep_ts: list[float] = []
        self._rep_valid_flags: list[bool] = []
        self._rep_lost_total = 0
        self._active_start_ts: float | None = None
        self._movement_start_ts: float | None = None
        self._active_frame_count = 0
        self._rep_start_val: float | None = None
        self._peak_val: float | None = None
        self._max_progress = 0.0
        self._enter_streak = 0
        self._return_streak = 0
        self._inflection_streak = 0
        self._peak_confirmed = False
        self._lost_streak = 0
        self._vis_lost_streak = 0
        self._combined_lost_streak = 0
        self._reacquire_lockout = 0
        self._needs_reanchor = False
        self._ready_confirmed = not self.require_ready_pose
        self._ready_streak = 0
        self._ready_start_ts: float | None = None
        self._cooldown = 0

    @property
    def is_tracking_lost(self) -> bool:
        return self._combined_lost_streak > 0

    @property
    def is_stabilizing(self) -> bool:
        return self.state == "IDLE" and self._combined_lost_streak == 0 and self._reacquire_lockout > 0

    @property
    def is_waiting_ready(self) -> bool:
        return (
            self.require_ready_pose
            and not self._ready_confirmed
            and self.state == "IDLE"
            and self._combined_lost_streak == 0
            and self._reacquire_lockout == 0
        )

    def _compute_progress(self, val: float) -> float:
        ref_span = max(abs(self.rest_anchor - self.spec["peak_ref"]), self.spec["min_rom"], 1e-4)
        if self.spec["mode"] == "min":
            raw = (self.rest_anchor - val) / ref_span
        else:
            raw = (val - self.rest_anchor) / ref_span
        return float(np.clip(raw, -0.2, 1.4))

    def _arm_reacquire_lockout(self) -> None:
        self._pre_roll.clear()
        self._rest_win.clear()
        self._raw_win.clear()
        self.ema_val = None
        self.prev_ema = None
        self.last_progress = 0.0
        self._enter_streak = 0
        self._movement_start_ts = None
        self._reacquire_lockout = max(self._reacquire_lockout, self.reacquire_warmup_frames)
        self._needs_reanchor = True
        self._ready_confirmed = not self.require_ready_pose
        self._ready_streak = 0
        self._ready_start_ts = None

    def _abort_active_rep(self) -> None:
        self.state = "IDLE"
        self._rep_frames.clear()
        self._rep_ts.clear()
        self._rep_valid_flags.clear()
        self._rep_lost_total = 0
        self._pre_roll.clear()
        self._rest_win.clear()
        self._active_start_ts = None
        self._movement_start_ts = None
        self._active_frame_count = 0
        self._rep_start_val = None
        self._peak_val = None
        self._max_progress = 0.0
        self._enter_streak = 0
        self._return_streak = 0
        self._inflection_streak = 0
        self._peak_confirmed = False
        self.last_progress = 0.0

    def update(self, frame_joints: np.ndarray, detected: bool, timestamp_sec: float) -> RepCompletedEvent | None:
        """Cập nhật 1 frame mới; trả về RepCompletedEvent khi kết thúc 1 rep hợp lệ."""
        posture_ok = bool(detected and has_valid_exercise_posture(frame_joints, self.exercise))
        raw_val, vis_ok = (
            compute_driver_signal_with_vis(frame_joints, self.exercise) if posture_ok else (0.0, False)
        )
        if not detected or not posture_ok or not vis_ok:
            if not detected:
                self._lost_streak += 1
            else:
                self._lost_streak = 0
                self._vis_lost_streak += 1
            self._combined_lost_streak += 1

            if self.state == "IN_REP":
                self._rep_lost_total += 1
                lost_too_long = (
                    self._lost_streak > self.max_lost_frames
                    or self._vis_lost_streak > self.max_lost_frames
                    or self._combined_lost_streak > self.max_lost_frames
                    or self._rep_lost_total > self.max_lost_frames
                    or (not self._peak_confirmed and self._combined_lost_streak >= 4)
                )
                if lost_too_long:
                    self._abort_active_rep()
                    self._arm_reacquire_lockout()
                else:
                    self._rep_frames.append(frame_joints.copy())
                    self._rep_ts.append(float(timestamp_sec))
                    self._rep_valid_flags.append(False)
            else:
                self._enter_streak = 0
                self._movement_start_ts = None
                self._ready_streak = 0
                self._ready_start_ts = None
                if self._combined_lost_streak >= 6:
                    self._arm_reacquire_lockout()
            return None

        self._lost_streak = 0
        self._vis_lost_streak = 0
        self._combined_lost_streak = 0

        self._raw_win.append(float(raw_val))
        if len(self._raw_win) >= 3:
            filt_val = float(np.median(self._raw_win))
        elif len(self._raw_win) == 2 and self.ema_val is not None:
            filt_val = float(min(self._raw_win, key=lambda x: abs(x - self.ema_val)))
        else:
            filt_val = float(raw_val)

        self.prev_ema = self.ema_val
        if self.ema_val is None:
            self.ema_val = filt_val
        else:
            jump_cap = 0.8 if self.spec["unit"] == "" else 40.0
            alpha = 0.15 if abs(filt_val - self.ema_val) > jump_cap else self.ema_alpha
            self.ema_val = alpha * filt_val + (1.0 - alpha) * self.ema_val
        val = float(self.ema_val)

        if self.state == "IDLE":
            self._rest_win.append(val)
            lo_r, hi_r = self.spec["rest_bounds"]
            flat_span = 0.15 * self.spec["min_rom"]
            in_rest_bounds = lo_r <= val <= hi_r
            win_span = (max(self._rest_win) - min(self._rest_win)) if self._rest_win else 999.0
            if len(self._rest_win) >= 4 and in_rest_bounds:
                if win_span <= flat_span or (self._needs_reanchor and win_span <= 0.45 * self.spec["min_rom"]):
                    self.rest_anchor = float(np.median(self._rest_win))
                    self._needs_reanchor = False

            if self._reacquire_lockout > 0:
                self._reacquire_lockout -= 1
                self._enter_streak = 0
                self._movement_start_ts = None
                self._ready_streak = 0
                self._ready_start_ts = None
                self._pre_roll.clear()
                self.last_progress = 0.0
                return None

            if self.require_ready_pose and not self._ready_confirmed:
                ready_posture_ok = has_valid_exercise_posture(frame_joints, self.exercise, strict_ready=True)
                stable_ready = len(self._rest_win) >= 4 and win_span <= max(flat_span, 0.30 * self.spec["min_rom"])
                if ready_posture_ok and in_rest_bounds and stable_ready:
                    if self._ready_streak == 0 or self._ready_start_ts is None:
                        self._ready_start_ts = float(timestamp_sec)
                    self._ready_streak += 1
                    ready_elapsed = float(timestamp_sec) - float(self._ready_start_ts)
                    if self._ready_streak >= self.ready_min_frames and ready_elapsed >= self.ready_hold_sec:
                        self._ready_confirmed = True
                        self.rest_anchor = float(np.median(self._rest_win))
                        self._needs_reanchor = False
                else:
                    self._ready_streak = 0
                    self._ready_start_ts = None

                self._enter_streak = 0
                self._movement_start_ts = None
                self._pre_roll.append((frame_joints.copy(), float(timestamp_sec), val))
                self.last_progress = 0.0
                if not self._ready_confirmed:
                    return None

            self._pre_roll.append((frame_joints.copy(), float(timestamp_sec), val))
            progress_now = self._compute_progress(val)
            self.last_progress = float(np.clip(progress_now, 0.0, 1.0))
            if self._cooldown > 0:
                self._cooldown -= 1
                self._enter_streak = 0
                self._movement_start_ts = None
                return None

            excursion = (self.rest_anchor - val) if self.spec["mode"] == "min" else (val - self.rest_anchor)

            if excursion >= 0.35 * self.spec["min_rom"] and progress_now >= 0.14:
                if self._enter_streak == 0 or self._movement_start_ts is None:
                    self._movement_start_ts = float(timestamp_sec)
                self._enter_streak += 1
            else:
                self._enter_streak = 0
                self._movement_start_ts = None

            if self._enter_streak >= 2:
                self.state = "IN_REP"
                self._rep_frames = [f for f, _, _ in self._pre_roll]
                self._rep_ts = [t for _, t, _ in self._pre_roll]
                self._rep_valid_flags = [True] * len(self._rep_frames)
                self._rep_lost_total = 0
                pre_vals = [v for _, _, v in self._pre_roll] or [self.rest_anchor]
                self._rep_start_val = float(max(pre_vals) if self.spec["mode"] == "min" else min(pre_vals))
                self._active_start_ts = self._rep_ts[0] if self._rep_ts else float(timestamp_sec)
                if self._movement_start_ts is None:
                    self._movement_start_ts = float(timestamp_sec)
                self._active_frame_count = 1
                self._peak_val = val
                self._max_progress = progress_now
                self._return_streak = 0
                self._enter_streak = 0
                self._inflection_streak = 0
                self._peak_confirmed = False
            return None

        # State == IN_REP
        self._rep_frames.append(frame_joints.copy())
        self._rep_ts.append(float(timestamp_sec))
        self._rep_valid_flags.append(True)
        self._active_frame_count += 1
        dur = float(timestamp_sec - (self._active_start_ts or timestamp_sec))

        if dur > self.max_rep_sec:
            self._abort_active_rep()
            return None

        if self.spec["mode"] == "min":
            self._peak_val = val if self._peak_val is None else min(self._peak_val, val)
        else:
            self._peak_val = val if self._peak_val is None else max(self._peak_val, val)

        start_v = float(self._rep_start_val if self._rep_start_val is not None else self.rest_anchor)
        ref_extreme = max(start_v, val) if self.spec["mode"] == "min" else min(start_v, val)
        progress_now = self._compute_progress(val)
        self._max_progress = max(self._max_progress, progress_now)
        self.last_progress = float(np.clip(progress_now, 0.0, 1.0))

        obs_rom = abs(ref_extreme - float(self._peak_val))
        start_rom = abs(start_v - float(self._peak_val))
        return_excursion = abs(val - float(self._peak_val))
        return_frac = return_excursion / max(start_rom if start_rom >= self.spec["min_rom"] else obs_rom, 1e-4)

        if obs_rom >= self.spec["min_rom"]:
            self._inflection_streak += 1
            if self._inflection_streak >= self.min_inflection_frames:
                self._peak_confirmed = True
        else:
            self._inflection_streak = 0

        reached_inflection = self._peak_confirmed and obs_rom >= self.spec["min_rom"]

        if reached_inflection and (progress_now <= 0.16 or (return_frac >= 0.85 and return_excursion >= self.spec["min_rom"])):
            self._return_streak += 1
        else:
            self._return_streak = 0

        if reached_inflection and (progress_now <= 0.10 or self._return_streak >= 2):
            return self.flush_if_valid()
        return None

    def flush_if_valid(self) -> RepCompletedEvent | None:
        """Kết thúc rep đang mở (hữu ích khi kết thúc video replay) nếu đã đủ biên độ, tỷ lệ frame hợp lệ & hồi về."""
        n_total = len(self._rep_frames)
        n_valid = int(sum(self._rep_valid_flags)) if self._rep_valid_flags else n_total
        valid_ratio = n_valid / max(1, n_total)
        if (
            self.state != "IN_REP"
            or not self._peak_confirmed
            or n_valid < self.min_rep_frames
            or valid_ratio < self.min_valid_ratio
            or self._active_frame_count < self.min_active_frames
        ):
            self._abort_active_rep()
            if valid_ratio < self.min_valid_ratio:
                self._arm_reacquire_lockout()
            return None

        valid_ts = (
            [t for t, ok in zip(self._rep_ts, self._rep_valid_flags) if ok]
            if self._rep_valid_flags
            else self._rep_ts
        )
        dur = max(float(valid_ts[-1] - valid_ts[0]), n_valid / 30.0) if len(valid_ts) >= 2 else 0.0
        active_dur = (
            max(float(valid_ts[-1] - float(self._movement_start_ts)), self._active_frame_count / 30.0)
            if (self._movement_start_ts is not None and valid_ts)
            else dur
        )
        start_v = float(self._rep_start_val if self._rep_start_val is not None else self.rest_anchor)
        val_end = float(self.ema_val) if self.ema_val is not None else start_v
        ref_extreme = max(start_v, val_end) if self.spec["mode"] == "min" else min(start_v, val_end)
        obs_rom = abs(ref_extreme - float(self._peak_val)) if self._peak_val is not None else 0.0
        return_frac = abs(val_end - float(self._peak_val)) / max(obs_rom, 1e-4) if self._peak_val is not None else 0.0

        if dur < self.min_rep_sec or active_dur < self.min_active_sec or obs_rom < self.spec["min_rom"] or return_frac < 0.45:
            self._abort_active_rep()
            return None

        self.rep_count += 1
        event = RepCompletedEvent(
            rep_index=self.rep_count,
            exercise=self.exercise,
            pose_seq=np.stack(self._rep_frames, axis=0).astype(np.float32),
            duration_sec=dur,
            observed_rom=obs_rom,
        )
        self._abort_active_rep()
        self._cooldown = 5
        return event


class ForgAIRealtimeScorer:
    """Nạp checkpoint Bước E.2 (1 Fold) và suy luận 1 rep khớp 100% pipeline huấn luyện."""

    def __init__(self, fold: int = 2, device: str | torch.device | None = None, checkpoints_dir: Path = CHECKPOINTS_DIR):
        self.fold = int(fold)
        self.device = torch.device(device or "cpu")
        self.ckpt_dir = Path(checkpoints_dir) / f"fold_{self.fold}"
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục checkpoint: {self.ckpt_dir}")

        with open(self.ckpt_dir / "tabular_and_ensemble.pkl", "rb") as f:
            pkg = pickle.load(f)

        self.rf_model = pkg["rf_model"]
        if hasattr(self.rf_model, "rf"):
            self.rf_model.rf.n_jobs = 1
        if hasattr(self.rf_model, "et"):
            self.rf_model.et.n_jobs = 1
        self.hybrid_ensemble = pkg["hybrid_ensemble"]
        self.rule_engine = pkg["rule_engine"]
        self.golden_bank = pkg["golden_bank"]
        self.seq_scaler = pkg["seq_scaler"]
        self.tab_scaler = pkg["tab_scaler"]

        self.cnn_model = self._load_dl_model("cnn_bilstm", self.ckpt_dir / "cnn_bilstm_best.pt")
        self.trans_model = self._load_dl_model("transformer", self.ckpt_dir / "transformer_best.pt")

    def _load_dl_model(self, model_name: str, pt_path: Path) -> torch.nn.Module:
        ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
        dropout = float(ckpt.get("hparams", {}).get("dropout", 0.42))
        model = build_model(
            model_name,
            seq_dim=int(ckpt["seq_dim"]),
            tab_dim=int(ckpt["tab_dim"]),
            dropout=dropout,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    def score_rep(self, pose_seq: np.ndarray, exercise: str, duration_sec: float) -> dict[str, Any]:
        """Chấm điểm Đúng/Sai + giải thích mã lỗi tiếng Việt cho 1 rep (L, 13, 6)."""
        if exercise not in EXERCISE_TO_ID:
            raise ValueError(f"Bài tập không nằm trong Nhóm A: {exercise}")

        t0 = time.perf_counter()
        coords_clean, vis_clean = handle_zero_frames(pose_seq, min_vis=MIN_VISIBILITY_THRESHOLD, smooth_window=3)
        coords_norm = normalize_skeleton_geometry(coords_clean)

        c32 = dynamic_strided_sampling(coords_norm, target_len=NUM_FRAMES, is_train=False)
        v32 = dynamic_strided_sampling(vis_clean, target_len=NUM_FRAMES, is_train=False)
        s117 = compute_117d_biomechanical_mix(c32)
        a10 = extract_anatomical_golden_matrix(c32, v32)

        ex_id = int(EXERCISE_TO_ID[exercise])
        ex_arr = np.asarray([ex_id], dtype=np.int64)
        m117, std117, manat, sanat = self.golden_bank.get_prototype(ex_id)
        gdev = self.golden_bank.transform_sequence_deviations(s117[None], a10[None], ex_arr)[0]

        seq_120 = np.concatenate([s117, gdev], axis=-1)[None]
        x_seq_sc = self.seq_scaler.transform(seq_120)

        x_tab_raw = compute_biomechanical_tabular_features(
            coords_seq=c32,
            vis_seq=v32,
            exercise_name=exercise,
            seq_117=s117,
            golden_mean_117=m117,
            golden_std_117=std117,
            anat_golden_seq=a10,
            golden_mean_anat=manat,
            golden_std_anat=sanat,
            duration_sec=float(duration_sec),
        )
        x_tab_sc = self.tab_scaler.transform(x_tab_raw[None])

        if hasattr(self.rf_model, "predict_unshifted_proba"):
            p_rf_for_hybrid = self.rf_model.predict_unshifted_proba(
                x_tab_sc, tab_raw=x_tab_raw[None], exercises=[exercise]
            )
            p_rf = self.rf_model.apply_prior_shift(p_rf_for_hybrid)
        else:
            p_rf = self.rf_model.predict_proba(x_tab_sc, tab_raw=x_tab_raw[None], exercises=[exercise])
            p_rf_for_hybrid = p_rf

        with torch.inference_mode():
            xs_t = torch.from_numpy(x_seq_sc).to(self.device)
            xt_t = torch.from_numpy(x_tab_sc).to(self.device)
            ex_t = torch.from_numpy(ex_arr).to(self.device)
            p_c = float(torch.sigmoid(self.cnn_model(xs_t, ex_t, xt_t)).item())
            p_t = float(torch.sigmoid(self.trans_model(xs_t, ex_t, xt_t)).item())

        probs_dict = {
            "rf": p_rf_for_hybrid,
            "cnn_bilstm": np.asarray([[1.0 - p_c, p_c]], dtype=np.float32),
            "transformer": np.asarray([[1.0 - p_t, p_t]], dtype=np.float32),
        }
        p_hyb = float(self.hybrid_ensemble.predict_proba(probs_dict)[0, 1])
        threshold = float(self.hybrid_ensemble.threshold)
        is_incorrect = bool(p_hyb >= threshold)

        explain = self.rule_engine.explain_rep(
            x_tab_raw,
            exercise_name=exercise,
            p_incorrect=p_hyb,
            is_incorrect=is_incorrect,
            top_k=3,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "exercise": exercise,
            "duration_sec": round(float(duration_sec), 2),
            "num_frames": int(pose_seq.shape[0]),
            "p_rf": round(float(p_rf[0, 1]), 4),
            "p_cnn_bilstm": round(p_c, 4),
            "p_transformer": round(p_t, 4),
            "p_incorrect": round(p_hyb, 4),
            "threshold": threshold,
            "is_incorrect": is_incorrect,
            "label_vi": "SAI KỸ THUẬT" if is_incorrect else "ĐÚNG KỸ THUẬT",
            "infer_ms": round(infer_ms, 2),
            **explain,
        }
