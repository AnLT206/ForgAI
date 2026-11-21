# -*- coding: utf-8 -*-
"""
src/features.py
Module tiền xử lý hình học khung xương & trích xuất đặc trưng cơ sinh học cho ForgAI Task 2:
  1. handle_zero_frames(): Nội suy khớp bị che khuất (min_vis=0.20) + khử rung 3-frame.
  2. normalize_skeleton_geometry(): Đưa gốc tọa độ về tâm hông & chia cho trung vị chiều dài lưng (Cải tiến #1).
  3. horizontal_flip_skeleton(): Phép lật gương Trái-Phải an toàn bảo toàn 100% góc khớp (SWAP_MAP_13).
  4. compute_117d_biomechanical_mix(): Vector đặc trưng SkelGym 117 chiều (39 tọa độ 3D + 78 góc nghiêng 3D).
  5. compute_anatomical_angle_series() & compute_biomechanical_tabular_features():
     Tích hợp trọn vẹn Cải tiến #2 (chọn bên rõ + bất đối xứng), #3 (ảnh chụp đáy Inflection Frame),
     #4 (vận tốc, tỷ lệ pha Âm/Dương, gia tốc giật đáy & độ trễ lưng/hông), #5 (độ lệch Golden-Rep).
"""

from __future__ import annotations

import numpy as np

from configs.config import (
    EXERCISE_TO_ID,
    GROUP_A_EXERCISES,
    MIN_VISIBILITY_THRESHOLD,
    NUM_EXERCISES,
    NUM_JOINTS,
)
from src.constants import J_IDX, PAIRWISE_JOINT_INDICES_78, SWAP_MAP_13

# ==============================================================================
# HẰNG SỐ CHỈ SỐ KHỚP & ĐẶC TẢ CƠ SINH HỌC CHO 8 BÀI TẬP NHÓM A
# ==============================================================================
_L_SH, _R_SH = J_IDX["left_shoulder"], J_IDX["right_shoulder"]
_L_EL, _R_EL = J_IDX["left_elbow"], J_IDX["right_elbow"]
_L_WR, _R_WR = J_IDX["left_wrist"], J_IDX["right_wrist"]
_L_HP, _R_HP = J_IDX["left_hip"], J_IDX["right_hip"]
_L_KN, _R_KN = J_IDX["left_knee"], J_IDX["right_knee"]
_L_AN, _R_AN = J_IDX["left_ankle"], J_IDX["right_ankle"]
_NOSE = J_IDX["nose"]

_PAIRWISE_IDX_78 = np.asarray(PAIRWISE_JOINT_INDICES_78, dtype=np.int64)

_TRIPLETS = {
    "knee_L": (_L_HP, _L_KN, _L_AN),
    "knee_R": (_R_HP, _R_KN, _R_AN),
    "hip_L": (_L_SH, _L_HP, _L_KN),
    "hip_R": (_R_SH, _R_HP, _R_KN),
    "elbow_L": (_L_SH, _L_EL, _L_WR),
    "elbow_R": (_R_SH, _R_EL, _R_WR),
    "shoulder_L": (_L_HP, _L_SH, _L_EL),
    "shoulder_R": (_R_HP, _R_SH, _R_EL),
    "body_L": (_L_SH, _L_HP, _L_AN),
    "body_R": (_R_SH, _R_HP, _R_AN),
}

# Quy định chuỗi góc chủ đạo, điểm cực trị chuyển pha (min hay max) và pha đầu là Hạ (Eccentric) hay Nâng (Concentric)
EXERCISE_BIOMECH_SPEC = {
    "pull up": {"driver": "elbow_angle", "mode": "min", "phase1_is_eccentric": False},
    "push-up": {"driver": "elbow_angle", "mode": "min", "phase1_is_eccentric": True},
    "leg extension": {"driver": "knee_angle", "mode": "max", "phase1_is_eccentric": False},
    "squat": {"driver": "knee_angle", "mode": "min", "phase1_is_eccentric": True},
    "chest fly machine": {"driver": "wrist_squeeze_dist", "mode": "min", "phase1_is_eccentric": False},
    "hip thrust": {"driver": "hip_angle", "mode": "max", "phase1_is_eccentric": False},
    "t bar row": {"driver": "elbow_angle", "mode": "min", "phase1_is_eccentric": False},
    "leg raises": {"driver": "hip_angle", "mode": "min", "phase1_is_eccentric": False},
}

# Khung định nghĩa cố định tên cột đặc trưng bảng (100 chiều) bảo đảm đồng nhất tuyệt đối
_CORE_SERIES_NAMES = (
    "knee_angle",
    "hip_angle",
    "elbow_angle",
    "shoulder_angle",
    "body_line_angle",
    "torso_tilt",
    "neck_angle",
    "shin_tilt",
)
_CORE_STATS = ("min", "max", "rom", "mean", "std", "at_inflection")

_DOMAIN_STAT_SPECS = [
    ("waist_sag_signed", ("min", "max", "at_inflection", "mean")),
    ("hip_below_bridge", ("min", "at_inflection", "mean")),
    ("elbow_drop_vs_sh", ("max", "at_inflection", "mean")),
    ("chin_above_wrist", ("max", "at_inflection")),
    ("wrist_to_knee_dist", ("min", "mean")),
    ("elbow_flare_ratio", ("max", "at_inflection")),
    ("knee_valgus_ratio", ("min", "at_inflection")),
    ("wrist_squeeze_dist", ("min", "at_inflection", "max")),
    ("spine_compression", ("min", "at_inflection")),
    ("neck_to_sh_dist", ("min", "at_inflection")),
]

_ASYM_NAMES = (
    "elbow_asym_max",
    "elbow_asym_mean",
    "shoulder_y_asym_max",
    "knee_asym_max",
)

_DYNAMIC_NAMES = (
    "inflection_time_frac",
    "rep_duration_sec",
    "ecc_duration_sec",
    "driver_phase1_speed",
    "driver_phase2_speed",
    "ecc_conc_speed_ratio",
    "driver_peak_speed",
    "driver_speed_bouncing",
    "torso_tilt_peak_speed",
    "torso_tilt_speed_bouncing",
    "hips_rising_first_lag",
    "snaking_lag",
    "driver_peak_to_mean_speed",
    "driver_dimless_jerk",
    "torso_driver_corr",
    "hip_vs_wrist_2d_x_rom",
    "hip_vs_wrist_2d_x_std",
    "hip_vs_wrist_2d_x_speed",
    "ank_vs_wrist_2d_x_rom",
    "ank_vs_wrist_2d_y_rom",
    "torso_2d_len_ratio_rom",
)

_GOLDEN_STAT_NAMES = (
    "golden_coord_dev_mean",
    "golden_angle_dev_mean",
    "golden_dev_at_inflection",
    "golden_dev_max",
    "golden_dev_std",
)

ANATOMICAL_GOLDEN_KEYS = (
    "knee_angle",
    "hip_angle",
    "elbow_angle",
    "shoulder_angle",
    "body_line_angle",
    "torso_tilt",
    "neck_angle",
    "shin_tilt",
    "waist_sag_signed",
    "hip_below_bridge",
)
ANATOMICAL_GOLDEN_MIN_STD = np.asarray(
    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.08, 0.08], dtype=np.float32
)


def extract_anatomical_golden_matrix(coords: np.ndarray, vis: np.ndarray | None = None) -> np.ndarray:
    """Trả về ma trận (T, 10) các đường cong giải phẫu đã chọn bên rõ để tính Khuôn Mẫu Chuẩn."""
    series = compute_anatomical_angle_series(coords, vis)
    return np.stack([series[k] for k in ANATOMICAL_GOLDEN_KEYS], axis=-1).astype(np.float32)


def get_tabular_feature_names() -> list[str]:
    """Trả về danh sách tên 100 đặc trưng cơ sinh học theo đúng thứ tự cột."""
    names: list[str] = []
    for s_name in _CORE_SERIES_NAMES:
        for stat in _CORE_STATS:
            names.append(f"{s_name}_{stat}")
    for s_name, stats in _DOMAIN_STAT_SPECS:
        for stat in stats:
            names.append(f"{s_name}_{stat}")
    names.extend(_ASYM_NAMES)
    names.extend(_DYNAMIC_NAMES)
    names.extend(_GOLDEN_STAT_NAMES)
    for ex_name in GROUP_A_EXERCISES:
        names.append(f"ex_onehot_{ex_name.replace(' ', '_').replace('-', '_')}")
    return names


TABULAR_FEATURE_NAMES = get_tabular_feature_names()
TABULAR_FEATURE_DIM = len(TABULAR_FEATURE_NAMES)


# ==============================================================================
# 1. NỘI SUY KHUNG HÌNH KHUẤT & LÀM MƯỢT CHỐNG RUNG (CẢI TIẾN VISIBILITY-GATED)
# ==============================================================================
def _extract_vis_and_det_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tách mảng visibility [L, 13] và mặt nạ frame phát hiện được người tập từ (L, 13, C)."""
    vis = arr[:, :, 5].copy() if arr.shape[2] >= 6 else np.ones(arr.shape[:2], dtype=np.float64)
    vis = np.clip(vis, 0.0, 1.0)
    det_mask = np.abs(arr[:, :, :3]).sum(axis=(1, 2)) > 1e-6
    return vis, det_mask


def _interpolate_and_smooth_coords(
    coords: np.ndarray,
    vis: np.ndarray,
    det_mask: np.ndarray,
    min_vis: float,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy tuyến tính theo thời gian trên các khớp che khuất và làm mượt chống rung."""
    num_frames = coords.shape[0]
    n_axes = coords.shape[2]
    vis_out = vis.copy()
    coords_out = coords.copy()
    if num_frames == 0:
        return coords_out.astype(np.float32), vis_out.astype(np.float32)

    if not np.any(det_mask):
        return np.zeros_like(coords_out, dtype=np.float32), np.zeros_like(vis_out, dtype=np.float32)

    t_all = np.arange(num_frames, dtype=np.float64)
    det_count = int(det_mask.sum())
    min_valid_joint_frames = min(5, max(2, det_count // 4))

    for j in range(NUM_JOINTS):
        if det_count < num_frames:
            vis_out[:, j] = np.clip(np.interp(t_all, t_all[det_mask], vis_out[det_mask, j]), 0.0, 1.0)

        joint_valid = det_mask & (vis_out[:, j] >= min_vis)
        use_mask = joint_valid if int(joint_valid.sum()) >= min_valid_joint_frames else det_mask

        if int(use_mask.sum()) == 1:
            coords_out[:, j, :] = coords_out[use_mask, j, :][0]
        else:
            t_src = t_all[use_mask]
            for axis in range(n_axes):
                coords_out[:, j, axis] = np.interp(t_all, t_src, coords_out[use_mask, j, axis])

    # Làm mượt nhẹ khử rung MediaPipe (tuyệt đối giữ nguyên đầu-cuối nhờ edge padding)
    if smooth_window == 3 and num_frames >= 3:
        padded = np.pad(coords_out, ((1, 1), (0, 0), (0, 0)), mode="edge")
        coords_out = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    elif smooth_window > 3 and num_frames >= smooth_window:
        pad = smooth_window // 2
        kernel = np.ones(smooth_window, dtype=np.float64) / float(smooth_window)
        padded = np.pad(coords_out, ((pad, pad), (0, 0), (0, 0)), mode="edge")
        coords_out = np.tensordot(
            np.lib.stride_tricks.sliding_window_view(padded, smooth_window, axis=0),
            kernel,
            axes=([-1], [0]),
        )

    return coords_out.astype(np.float32), vis_out.astype(np.float32)


def handle_zero_frames(
    pose_seq: np.ndarray,
    min_vis: float = MIN_VISIBILITY_THRESHOLD,
    smooth_window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hàn gắn các frame mất người hoặc khớp bị che khuất (visibility < min_vis)
    bằng nội suy tuyến tính theo thời gian, sau đó khử rung bằng lọc trượt 3-frame.

    Args:
        pose_seq: Mảng shape (L, 13, 6) [wx, wy, wz, ix, iy, vis] hoặc (L, 13, 3).
        min_vis: Ngưỡng độ rõ dưới mức này được coi là bị che khuất tạm thời.
        smooth_window: Độ dài cửa sổ làm mượt [0.25, 0.50, 0.25] nếu bằng 3.

    Returns:
        coords_clean: Mảng 3D shape (L, 13, 3) float32.
        vis_clean: Mảng độ rõ shape (L, 13) float32.
    """
    arr = np.asarray(pose_seq, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != NUM_JOINTS:
        raise ValueError(f"Expected pose_seq shape (L, {NUM_JOINTS}, C), got {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vis, det_mask = _extract_vis_and_det_mask(arr)
    return _interpolate_and_smooth_coords(arr[:, :, :3], vis, det_mask, min_vis, smooth_window)


def handle_zero_frames_2d(
    pose_seq: np.ndarray,
    min_vis: float = MIN_VISIBILITY_THRESHOLD,
    smooth_window: int = 3,
) -> np.ndarray | None:
    """
    Trích xuất và nội suy làm mượt chuỗi tọa độ khung hình 2D [ix, iy] shape (L, 13, 2)
    từ pose_seq (L, 13, 6) để đo chuyển động con lắc/vung thân so với điểm tựa (thanh xà).
    Trả về None nếu đầu vào không chứa kênh 2D.
    """
    arr = np.asarray(pose_seq, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != NUM_JOINTS or arr.shape[2] < 5:
        return None

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    coords_2d = arr[:, :, 3:5]
    if not np.any(np.abs(coords_2d).sum(axis=(1, 2)) > 1e-6):
        return None

    vis, det_mask = _extract_vis_and_det_mask(arr)
    coords_2d_clean, _ = _interpolate_and_smooth_coords(coords_2d, vis, det_mask, min_vis, smooth_window)
    return coords_2d_clean


# ==============================================================================
# 2. CHUẨN HÓA TÂM HÔNG & CHIỀU DÀI THÂN LƯNG (CẢI TIẾN #1 - SKELGYM + SHAHMIR2004)
# ==============================================================================
def compute_torso_scale(coords: np.ndarray, min_torso_len: float = 0.15) -> float:
    """Tính trung vị độ dài thân lưng (khoảng cách từ trung điểm 2 hông lên trung điểm 2 vai)."""
    hip_mid = 0.5 * (coords[:, _L_HP, :] + coords[:, _R_HP, :])
    sh_mid = 0.5 * (coords[:, _L_SH, :] + coords[:, _R_SH, :])
    lengths = np.linalg.norm(sh_mid - hip_mid, axis=-1)
    scale = float(np.median(lengths)) if lengths.size > 0 else min_torso_len
    return max(scale, float(min_torso_len))


def normalize_skeleton_geometry(coords: np.ndarray, min_torso_len: float = 0.15) -> np.ndarray:
    """
    Dời gốc tọa độ (0,0,0) về trung điểm hai khớp hông tại từng frame (Hip-Centering),
    sau đó chia cho trung vị chiều dài thân lưng cả rep (Torso-Length Normalization)
    để chuẩn hóa chiều cao mọi người tập về 1.0 mà vẫn bảo toàn độ co/gù cột sống.
    """
    arr = np.asarray(coords, dtype=np.float32)
    hip_mid = 0.5 * (arr[:, _L_HP : _L_HP + 1, :] + arr[:, _R_HP : _R_HP + 1, :])
    centered = arr - hip_mid
    scale = compute_torso_scale(arr, min_torso_len=min_torso_len)
    return (centered / np.float32(scale)).astype(np.float32)


# ==============================================================================
# 3. LẬT GƯƠNG TRÁI <-> PHẢI AN TOÀN (BẢO TOÀN 100% GÓC KỸ THUẬT)
# ==============================================================================
def horizontal_flip_skeleton(
    coords: np.ndarray,
    vis: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Đảo dấu trục ngang X và hoán đổi vị trí các cặp khớp Trái-Phải theo SWAP_MAP_13.
    Bảo toàn 100% góc khớp và góc nghiêng so với phương trọng lực.
    """
    flipped = np.array(coords, dtype=np.float32, copy=True)
    flipped[:, :, 0] *= -1.0
    flipped = flipped[:, SWAP_MAP_13, :]

    if vis is not None:
        flipped_vis = np.array(vis, dtype=np.float32, copy=True)[:, SWAP_MAP_13]
        return flipped, flipped_vis
    return flipped


def horizontal_flip_2d(coords_2d: np.ndarray) -> np.ndarray:
    """Lật gương ngang tọa độ khung hình 2D [ix, iy] (1.0 - ix) đồng bộ SWAP_MAP_13."""
    flipped = np.array(coords_2d, dtype=np.float32, copy=True)
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    return flipped[:, SWAP_MAP_13, :]


# ==============================================================================
# 4. ĐẶC TRƯNG SKELGYM 117-D BIOMECHANICAL MIX CHO NHÁNH DEEP LEARNING
# ==============================================================================
def compute_117d_biomechanical_mix(coords: np.ndarray) -> np.ndarray:
    """
    Biến đổi chuỗi tọa độ chuẩn hóa (T, 13, 3) thành ma trận đặc trưng (T, 117):
      - 39 chiều đầu: Tọa độ 3D tương đối (13 khớp x 3 trục X, Y, Z).
      - 78 chiều sau: Góc nghiêng 3D [0..pi rad] so với phương thẳng đứng Y của tất cả
        C(13, 2) = 78 đoạn nối từng cặp khớp trong PAIRWISE_JOINT_INDICES_78.
    """
    arr = np.asarray(coords, dtype=np.float32)
    t_len = arr.shape[0]
    idx_i = _PAIRWISE_IDX_78[:, 0]
    idx_j = _PAIRWISE_IDX_78[:, 1]

    vecs = arr[:, idx_j, :] - arr[:, idx_i, :]          # (T, 78, 3)
    norms = np.linalg.norm(vecs, axis=-1)               # (T, 78)
    cos_y = np.clip(vecs[:, :, 1] / (norms + 1e-6), -1.0, 1.0)
    angles_78 = np.arccos(cos_y).astype(np.float32)     # (T, 78)

    coords_39 = arr.reshape(t_len, NUM_JOINTS * 3)      # (T, 39)
    return np.concatenate([coords_39, angles_78], axis=-1).astype(np.float32)


# ==============================================================================
# 5. CHUỖI GÓC GIẢI PHẪU & ĐẶC TRƯNG BẢNG CƠ SINH HỌC (5 CẢI TIẾN)
# ==============================================================================
def _triplet_angle_series_deg(coords: np.ndarray, triplet: tuple[int, int, int]) -> np.ndarray:
    """Tính chuỗi góc 3D tại đỉnh b trong bộ ba khớp (a, b, c) bằng độ [0..180]."""
    a, b, c = triplet
    v1 = coords[:, a, :] - coords[:, b, :]
    v2 = coords[:, c, :] - coords[:, b, :]
    norm_prod = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    cos_val = np.clip(dot / (norm_prod + 1e-6), -1.0, 1.0)
    # Khi hai đoạn ngắn về 0, gán góc mặc định 180 độ tránh nan
    angles = np.degrees(np.arccos(cos_val))
    angles = np.where(norm_prod > 1e-8, angles, 180.0)
    return angles.astype(np.float32)


def _tilt_from_vertical_up_deg(vecs: np.ndarray) -> np.ndarray:
    """
    Góc nghiêng của vector so với hướng thẳng đứng LÊN TRÊN (0, -1, 0) (vì +Y MediaPipe hướng xuống).
    0 độ = thẳng đứng hướng lên; 90 độ = nằm ngang; 180 độ = chúc ngược xuống.
    """
    norms = np.linalg.norm(vecs, axis=-1)
    cos_up = np.clip(-vecs[:, 1] / (norms + 1e-6), -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_up))
    return np.where(norms > 1e-6, angles, 0.0).astype(np.float32)


def _visibility_weighted_side(
    left_vals: np.ndarray,
    right_vals: np.ndarray,
    vis: np.ndarray,
    left_idxs: tuple[int, ...],
    right_idxs: tuple[int, ...],
    dom_margin: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cải tiến #2 (từ shervinnkh/gym-assistant):
      - Nếu một bên rõ hơn hẳn bên kia (|w_L - w_R| > 0.25), lấy hoàn toàn góc của bên rõ.
      - Nếu cả hai bên đều rõ tương đương (quay chính diện/45°), lấy trung bình có trọng số.
      - Đồng thời trả về chênh lệch |Trái - Phải| được điều tiết theo độ rõ chung để đo bất đối xứng.
    """
    w_l = vis[:, left_idxs].min(axis=1)
    w_r = vis[:, right_idxs].min(axis=1)

    w_sum = w_l + w_r
    blended = np.where(w_sum > 1e-6, (w_l * left_vals + w_r * right_vals) / np.maximum(w_sum, 1e-6), 0.5 * (left_vals + right_vals))
    combined = np.where((w_l - w_r) > dom_margin, left_vals, blended)
    combined = np.where((w_r - w_l) > dom_margin, right_vals, combined)

    both_clear_weight = np.clip((np.minimum(w_l, w_r) - 0.15) / 0.25, 0.0, 1.0)
    asym = np.abs(left_vals - right_vals) * both_clear_weight
    return combined.astype(np.float32), asym.astype(np.float32)


def _signed_hip_sag_vs_line_y(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """
    Độ võng (hoặc nhô) có dấu của hông (tại gốc 0,0,0) so với đường thẳng nối A và B
    trong mặt phẳng ngang (X, Z) + trọng lực (+Y hướng xuống):
      - Giá trị > 0: Hông tụt XUỐNG THẤP HƠN đường A-B (waist_sagging / hip_below_bridge).
      - Giá trị < 0: Hông nhô LÊN CAO HƠN đường A-B (hips_too_high / hyperextended).
    Bất biến 100% với phép xoay camera quanh trục thẳng đứng & lật gương.
    """
    d_a = np.hypot(p_a[:, 0], p_a[:, 2])
    d_b = np.hypot(p_b[:, 0], p_b[:, 2])
    u = d_a / (d_a + d_b + 1e-6)
    y_line = (1.0 - u) * p_a[:, 1] + u * p_b[:, 1]
    return (-y_line).astype(np.float32)


def compute_anatomical_angle_series(
    coords: np.ndarray,
    vis: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Tính toàn bộ chuỗi thời gian các góc giải phẫu & tỷ số cơ sinh học từ tọa độ chuẩn hóa (T, 13, 3).
    """
    arr = np.asarray(coords, dtype=np.float32)
    t_len = arr.shape[0]
    vis_arr = np.ones((t_len, NUM_JOINTS), dtype=np.float32) if vis is None else np.asarray(vis, dtype=np.float32)

    sh_mid = 0.5 * (arr[:, _L_SH, :] + arr[:, _R_SH, :])
    hp_mid = 0.5 * (arr[:, _L_HP, :] + arr[:, _R_HP, :])
    kn_mid = 0.5 * (arr[:, _L_KN, :] + arr[:, _R_KN, :])
    an_mid = 0.5 * (arr[:, _L_AN, :] + arr[:, _R_AN, :])
    el_mid = 0.5 * (arr[:, _L_EL, :] + arr[:, _R_EL, :])
    wr_mid = 0.5 * (arr[:, _L_WR, :] + arr[:, _R_WR, :])
    nose = arr[:, _NOSE, :]

    # 1. Năm cặp góc 3-điểm Trái/Phải + kết hợp theo visibility
    knee_l = _triplet_angle_series_deg(arr, _TRIPLETS["knee_L"])
    knee_r = _triplet_angle_series_deg(arr, _TRIPLETS["knee_R"])
    knee_angle, knee_asym = _visibility_weighted_side(knee_l, knee_r, vis_arr, _TRIPLETS["knee_L"], _TRIPLETS["knee_R"])

    hip_l = _triplet_angle_series_deg(arr, _TRIPLETS["hip_L"])
    hip_r = _triplet_angle_series_deg(arr, _TRIPLETS["hip_R"])
    hip_angle, _ = _visibility_weighted_side(hip_l, hip_r, vis_arr, _TRIPLETS["hip_L"], _TRIPLETS["hip_R"])

    elbow_l = _triplet_angle_series_deg(arr, _TRIPLETS["elbow_L"])
    elbow_r = _triplet_angle_series_deg(arr, _TRIPLETS["elbow_R"])
    elbow_angle, elbow_asym = _visibility_weighted_side(elbow_l, elbow_r, vis_arr, _TRIPLETS["elbow_L"], _TRIPLETS["elbow_R"])

    sh_l = _triplet_angle_series_deg(arr, _TRIPLETS["shoulder_L"])
    sh_r = _triplet_angle_series_deg(arr, _TRIPLETS["shoulder_R"])
    shoulder_angle, _ = _visibility_weighted_side(sh_l, sh_r, vis_arr, _TRIPLETS["shoulder_L"], _TRIPLETS["shoulder_R"])

    body_l = _triplet_angle_series_deg(arr, _TRIPLETS["body_L"])
    body_r = _triplet_angle_series_deg(arr, _TRIPLETS["body_R"])
    body_line_angle, _ = _visibility_weighted_side(body_l, body_r, vis_arr, _TRIPLETS["body_L"], _TRIPLETS["body_R"])

    # 2. Góc nghiêng thân lưng & cẳng chân so với phương thẳng đứng, góc gập/ngửa cổ
    torso_tilt = _tilt_from_vertical_up_deg(sh_mid - hp_mid)
    shin_l = _tilt_from_vertical_up_deg(arr[:, _L_KN, :] - arr[:, _L_AN, :])
    shin_r = _tilt_from_vertical_up_deg(arr[:, _R_KN, :] - arr[:, _R_AN, :])
    shin_tilt, _ = _visibility_weighted_side(shin_l, shin_r, vis_arr, (_L_KN, _L_AN), (_R_KN, _R_AN))

    # Góc cổ: góc tại Vai_giữa tạo bởi (Hông_giữa - Vai_giữa) và (Mũi - Vai_giữa)
    v_to_hip = hp_mid - sh_mid
    v_to_nose = nose - sh_mid
    neck_norm = np.linalg.norm(v_to_hip, axis=-1) * np.linalg.norm(v_to_nose, axis=-1)
    neck_cos = np.clip(np.sum(v_to_hip * v_to_nose, axis=-1) / (neck_norm + 1e-6), -1.0, 1.0)
    neck_angle = np.where(neck_norm > 1e-8, np.degrees(np.arccos(neck_cos)), 180.0).astype(np.float32)

    # 3. Các chỉ số hình học chuyên biệt cho 32 mã lỗi
    waist_sag_signed = _signed_hip_sag_vs_line_y(sh_mid, an_mid)
    hip_below_bridge = _signed_hip_sag_vs_line_y(sh_mid, kn_mid)
    elbow_drop_vs_sh = (el_mid[:, 1] - sh_mid[:, 1]).astype(np.float32)
    chin_above_wrist = (wr_mid[:, 1] - nose[:, 1]).astype(np.float32)

    d_wl_kl = np.linalg.norm(arr[:, _L_WR, :] - arr[:, _L_KN, :], axis=-1)
    d_wr_kr = np.linalg.norm(arr[:, _R_WR, :] - arr[:, _R_KN, :], axis=-1)
    wrist_to_knee_dist, _ = _visibility_weighted_side(d_wl_kl, d_wr_kr, vis_arr, (_L_WR, _L_KN), (_R_WR, _R_KN))

    sh_width = np.linalg.norm(arr[:, _L_SH, :] - arr[:, _R_SH, :], axis=-1)
    el_width = np.linalg.norm(arr[:, _L_EL, :] - arr[:, _R_EL, :], axis=-1)
    kn_width = np.linalg.norm(arr[:, _L_KN, :] - arr[:, _R_KN, :], axis=-1)
    an_width = np.linalg.norm(arr[:, _L_AN, :] - arr[:, _R_AN, :], axis=-1)
    wr_width = np.linalg.norm(arr[:, _L_WR, :] - arr[:, _R_WR, :], axis=-1)

    elbow_flare_ratio = np.clip(el_width / np.maximum(sh_width, 0.15), 0.0, 4.0).astype(np.float32)
    knee_valgus_ratio = np.clip(kn_width / np.maximum(an_width, 0.10), 0.0, 4.0).astype(np.float32)
    wrist_squeeze_dist = wr_width.astype(np.float32)
    spine_compression = np.linalg.norm(sh_mid - hp_mid, axis=-1).astype(np.float32)
    neck_to_sh_dist = np.linalg.norm(nose - sh_mid, axis=-1).astype(np.float32)

    sh_y_both_clear = np.clip((np.minimum(vis_arr[:, _L_SH], vis_arr[:, _R_SH]) - 0.15) / 0.25, 0.0, 1.0)
    shoulder_y_asym = (np.abs(arr[:, _L_SH, 1] - arr[:, _R_SH, 1]) * sh_y_both_clear).astype(np.float32)

    return {
        "knee_angle": knee_angle,
        "hip_angle": hip_angle,
        "elbow_angle": elbow_angle,
        "shoulder_angle": shoulder_angle,
        "body_line_angle": body_line_angle,
        "torso_tilt": torso_tilt,
        "neck_angle": neck_angle,
        "shin_tilt": shin_tilt,
        "waist_sag_signed": waist_sag_signed,
        "hip_below_bridge": hip_below_bridge,
        "elbow_drop_vs_sh": elbow_drop_vs_sh,
        "chin_above_wrist": chin_above_wrist,
        "wrist_to_knee_dist": wrist_to_knee_dist,
        "elbow_flare_ratio": elbow_flare_ratio,
        "knee_valgus_ratio": knee_valgus_ratio,
        "wrist_squeeze_dist": wrist_squeeze_dist,
        "spine_compression": spine_compression,
        "neck_to_sh_dist": neck_to_sh_dist,
        "elbow_asym": elbow_asym,
        "knee_asym": knee_asym,
        "shoulder_y_asym": shoulder_y_asym,
    }


def find_inflection_frame(series_dict: dict[str, np.ndarray], exercise_name: str) -> int:
    """
    Cải tiến #3 (từ solakoglukoray/form-check):
    Tìm chỉ số khung hình t* tại đỉnh/đáy chuyển pha chính của rep theo từng bài tập.
    Tìm trong khoảng [1, T-2] để tránh nhiễu ở 2 biên cắt video.
    """
    spec = EXERCISE_BIOMECH_SPEC[exercise_name]
    driver = series_dict[spec["driver"]]
    t_len = len(driver)
    if t_len <= 2:
        return int(np.argmin(driver) if spec["mode"] == "min" else np.argmax(driver))

    sub = driver[1 : t_len - 1]
    rel_idx = int(np.argmin(sub)) if spec["mode"] == "min" else int(np.argmax(sub))
    return 1 + rel_idx


def warp_sequence_to_inflection(seq_arr: np.ndarray, t_star: int, target_mid: int = 15) -> np.ndarray:
    """Căn chỉnh pha chuyển động (Piecewise Linear Registration) đưa điểm cực trị t* về đúng khung target_mid."""
    arr = np.asarray(seq_arr, dtype=np.float32)
    t_len = arr.shape[0]
    if t_len <= 4:
        return arr.copy()
    t_s = int(np.clip(t_star, 2, t_len - 3))
    mid = int(np.clip(target_mid, 1, t_len - 2))
    g1 = np.linspace(0.0, float(t_s), mid + 1, dtype=np.float32)
    g2 = np.linspace(float(t_s), float(t_len - 1), t_len - mid, dtype=np.float32)[1:]
    grid = np.concatenate([g1, g2], axis=0)
    idx0 = np.floor(grid).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_len - 1)
    w = (grid - idx0).astype(np.float32)
    if arr.ndim > 1:
        w = w.reshape((t_len,) + (1,) * (arr.ndim - 1))
    return ((1.0 - w) * arr[idx0] + w * arr[idx1]).astype(np.float32)


def unwarp_sequence_from_inflection(proto_arr: np.ndarray, t_star: int, source_mid: int = 15) -> np.ndarray:
    """Ánh xạ ngược Khuôn Mẫu Chuẩn (từ vị trí source_mid) về đúng pha chuyển đổi t* của từng mẫu."""
    arr = np.asarray(proto_arr, dtype=np.float32)
    t_len = arr.shape[0]
    if t_len <= 4:
        return arr.copy()
    t_s = int(np.clip(t_star, 2, t_len - 3))
    mid = int(np.clip(source_mid, 1, t_len - 2))
    g1 = np.linspace(0.0, float(mid), t_s + 1, dtype=np.float32)
    g2 = np.linspace(float(mid), float(t_len - 1), t_len - t_s, dtype=np.float32)[1:]
    grid = np.concatenate([g1, g2], axis=0)
    idx0 = np.floor(grid).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_len - 1)
    w = (grid - idx0).astype(np.float32)
    if arr.ndim > 1:
        w = w.reshape((t_len,) + (1,) * (arr.ndim - 1))
    return ((1.0 - w) * arr[idx0] + w * arr[idx1]).astype(np.float32)


def _extract_stat(arr: np.ndarray, stat_name: str, t_star: int) -> float:
    if stat_name == "min":
        return float(np.min(arr))
    if stat_name == "max":
        return float(np.max(arr))
    if stat_name == "rom":
        return float(np.max(arr) - np.min(arr))
    if stat_name == "mean":
        return float(np.mean(arr))
    if stat_name == "std":
        return float(np.std(arr))
    if stat_name == "at_inflection":
        return float(arr[t_star])
    raise ValueError(f"Unknown stat_name: {stat_name}")


def compute_biomechanical_tabular_features(
    coords_seq: np.ndarray,
    vis_seq: np.ndarray | None,
    exercise_name: str,
    seq_117: np.ndarray | None = None,
    golden_mean_117: np.ndarray | None = None,
    golden_std_117: np.ndarray | None = None,
    anat_golden_seq: np.ndarray | None = None,
    golden_mean_anat: np.ndarray | None = None,
    golden_std_anat: np.ndarray | None = None,
    duration_sec: float | None = None,
    coords2d_seq: np.ndarray | None = None,
) -> np.ndarray:
    """
    Trích xuất vector đặc trưng bảng cơ sinh học (TABULAR_FEATURE_DIM = 111 chiều):
      - Nhóm A (48): 8 chuỗi góc chính x 6 thống kê (gồm ảnh chụp tại đáy t*).
      - Nhóm B (29): 10 chỉ số chuyên biệt cho 32 mã lỗi + 4 chỉ số bất đối xứng Trái-Phải.
      - Nhóm C (21): Thời lượng thực (rep, eccentric), vận tốc/gia tốc theo thời gian thực (°/s, °/s²),
                     tỷ lệ pha Âm/Dương, độ trễ Hông/Vai, LDLJ, tương quan thân-khớp & 6 chỉ số lắc 2D.
      - Nhóm D (5): Độ lệch so với Khuôn Mẫu Chuẩn (Phase-Aligned Golden-Rep Prototype).
      - Nhóm E (8): Vector One-Hot bài tập để Random Forest phân tách ngưỡng từng bài.
    """
    if exercise_name not in EXERCISE_TO_ID:
        raise ValueError(f"Bài tập không nằm trong Nhóm A: {exercise_name}")

    series = compute_anatomical_angle_series(coords_seq, vis_seq)
    t_len = coords_seq.shape[0]
    t_star = find_inflection_frame(series, exercise_name)
    spec = EXERCISE_BIOMECH_SPEC[exercise_name]

    vals: list[float] = []

    # Nhóm A: 8 chuỗi góc chính x 6 thống kê
    for s_name in _CORE_SERIES_NAMES:
        arr = series[s_name]
        for stat in _CORE_STATS:
            vals.append(_extract_stat(arr, stat, t_star))

    # Nhóm B: 10 chuỗi chuyên biệt + 4 bất đối xứng
    for s_name, stats in _DOMAIN_STAT_SPECS:
        arr = series[s_name]
        for stat in stats:
            vals.append(_extract_stat(arr, stat, t_star))

    vals.append(float(np.max(series["elbow_asym"])))
    vals.append(float(np.mean(series["elbow_asym"])))
    vals.append(float(np.max(series["shoulder_y_asym"])))
    vals.append(float(np.max(series["knee_asym"])))

    # Nhóm C: Động học thời gian thực (°/s, °/s²), độ mượt LDLJ & dao động vung thân 2D
    safe_dur = (
        float(duration_sec)
        if (duration_sec is not None and np.isfinite(duration_sec) and float(duration_sec) > 0.0)
        else float(max(1, t_len - 1)) / 30.0
    )
    safe_dur = max(0.25, safe_dur)
    dt = safe_dur / float(max(1, t_len - 1))

    driver = series[spec["driver"]]
    vel_step = np.diff(driver) if t_len > 1 else np.zeros(1, dtype=np.float32)
    acc_step = np.diff(vel_step) if len(vel_step) > 1 else np.zeros(1, dtype=np.float32)
    jerk_step = np.diff(acc_step) if len(acc_step) > 1 else np.zeros(1, dtype=np.float32)

    vel = vel_step / dt
    abs_vel = np.abs(vel)
    acc = np.diff(vel) / dt if len(vel) > 1 else np.zeros(1, dtype=np.float32)

    p1_vel = float(np.mean(abs_vel[: max(1, t_star)]))
    p2_vel = float(np.mean(abs_vel[min(len(abs_vel) - 1, t_star) :]))
    if spec["phase1_is_eccentric"]:
        ecc_speed, conc_speed = p1_vel, p2_vel
    else:
        conc_speed, ecc_speed = p1_vel, p2_vel

    t_frac = float(t_star / max(1, t_len - 1))
    p1_dur = safe_dur * t_frac
    p2_dur = safe_dur * (1.0 - t_frac)
    ecc_duration_sec = p1_dur if spec["phase1_is_eccentric"] else p2_dur

    conc_pad = 0.25 if spec["driver"] == "wrist_squeeze_dist" else 5.0
    ecc_conc_ratio = float(np.clip(ecc_speed / (conc_speed + conc_pad), 0.0, 10.0))
    driver_peak_speed = float(np.max(abs_vel))
    min_mean_vel = 0.30 if spec["driver"] == "wrist_squeeze_dist" else 4.0
    driver_peak_to_mean_speed = float(np.clip(driver_peak_speed / max(float(np.mean(abs_vel)), min_mean_vel), 0.0, 20.0))
    rom_min = 0.15 if spec["driver"] == "wrist_squeeze_dist" else 5.0
    rom_safe = max(float(np.max(driver) - np.min(driver)), rom_min)
    driver_dimless_jerk = float(np.log1p(float(np.mean(jerk_step ** 2)) * (float(max(1, t_len)) ** 5) / (rom_safe ** 2)))

    if np.std(driver) > 1e-4 and np.std(series["torso_tilt"]) > 1e-4:
        corr_val = float(np.corrcoef(driver, series["torso_tilt"])[0, 1])
        torso_driver_corr = abs(corr_val) if np.isfinite(corr_val) else 0.0
    else:
        torso_driver_corr = 0.0

    # Gia tốc đổi chiều quanh vùng chuyển pha [t* - 3, t* + 3] (°/s²)
    lo_a = max(0, t_star - 3)
    hi_a = min(len(acc), t_star + 3)
    driver_bounce = float(np.max(np.abs(acc[lo_a:hi_a]))) if hi_a > lo_a else 0.0

    torso_vel = (np.diff(series["torso_tilt"]) / dt) if t_len > 1 else np.zeros(1, dtype=np.float32)
    torso_acc = (np.diff(torso_vel) / dt) if len(torso_vel) > 1 else np.zeros(1, dtype=np.float32)
    torso_peak_speed = float(np.max(np.abs(torso_vel)))
    torso_bounce = float(np.max(np.abs(torso_acc))) if len(torso_acc) > 0 else 0.0

    # Độ trễ hông bật trước vai (hips_rising_first) & đẩy người uốn sóng (snaking_push_up) ngay sau đáy t*
    t_post = min(t_len - 1, t_star + max(2, t_len // 6))
    hips_rising_first_lag = float(series["torso_tilt"][t_post] - series["torso_tilt"][t_star])
    snaking_lag = float(series["waist_sag_signed"][t_post] - series["waist_sag_signed"][t_star])

    # Chuyển động vung 2D tương đối trên khung hình (bất biến với phép lật gương Trái-Phải & khoảng cách camera)
    if (
        coords2d_seq is not None
        and np.asarray(coords2d_seq).shape[:2] == (t_len, NUM_JOINTS)
        and np.asarray(coords2d_seq).shape[2] >= 2
        and float(np.abs(coords2d_seq).sum()) > 1e-6
    ):
        c2 = np.asarray(coords2d_seq, dtype=np.float32)[:, :, :2]
        sh2 = 0.5 * (c2[:, _L_SH, :] + c2[:, _R_SH, :])
        hp2 = 0.5 * (c2[:, _L_HP, :] + c2[:, _R_HP, :])
        wr2 = 0.5 * (c2[:, _L_WR, :] + c2[:, _R_WR, :])
        an2 = 0.5 * (c2[:, _L_AN, :] + c2[:, _R_AN, :])
        torso_2d_lens = np.linalg.norm(sh2 - hp2, axis=-1)
        t2_scale = max(float(np.median(torso_2d_lens)), 0.05)

        hip_wr_x = (hp2[:, 0] - wr2[:, 0]) / t2_scale
        ank_wr_x = (an2[:, 0] - wr2[:, 0]) / t2_scale
        ank_wr_y = (an2[:, 1] - wr2[:, 1]) / t2_scale
        torso_2d_ratio = torso_2d_lens / t2_scale

        hip_vs_wrist_2d_x_rom = float(np.ptp(hip_wr_x))
        hip_vs_wrist_2d_x_std = float(np.std(hip_wr_x))
        hip_vs_wrist_2d_x_speed = float(np.max(np.abs(np.diff(hip_wr_x) / dt))) if t_len > 1 else 0.0
        ank_vs_wrist_2d_x_rom = float(np.ptp(ank_wr_x))
        ank_vs_wrist_2d_y_rom = float(np.ptp(ank_wr_y))
        torso_2d_len_ratio_rom = float(np.ptp(torso_2d_ratio))
    else:
        hip_vs_wrist_2d_x_rom = 0.0
        hip_vs_wrist_2d_x_std = 0.0
        hip_vs_wrist_2d_x_speed = 0.0
        ank_vs_wrist_2d_x_rom = 0.0
        ank_vs_wrist_2d_y_rom = 0.0
        torso_2d_len_ratio_rom = 0.0

    vals.extend([
        t_frac,
        safe_dur,
        ecc_duration_sec,
        p1_vel,
        p2_vel,
        ecc_conc_ratio,
        driver_peak_speed,
        driver_bounce,
        torso_peak_speed,
        torso_bounce,
        hips_rising_first_lag,
        snaking_lag,
        driver_peak_to_mean_speed,
        driver_dimless_jerk,
        torso_driver_corr,
        hip_vs_wrist_2d_x_rom,
        hip_vs_wrist_2d_x_std,
        hip_vs_wrist_2d_x_speed,
        ank_vs_wrist_2d_x_rom,
        ank_vs_wrist_2d_y_rom,
        torso_2d_len_ratio_rom,
    ])

    # Nhóm D: Độ lệch so với Khuôn Mẫu Chuẩn Căn Chỉnh Pha (KIMORE / Liao et al., 2020)
    coord_dev = 0.0
    if seq_117 is not None and golden_mean_117 is not None and golden_std_117 is not None:
        seq_117_w = warp_sequence_to_inflection(seq_117, t_star, target_mid=15)
        z_117 = np.clip(np.abs(seq_117_w - golden_mean_117) / np.maximum(golden_std_117, 0.05), 0.0, 10.0)
        coord_dev = float(z_117[:, : NUM_JOINTS * 3].mean())

    if golden_mean_anat is not None and golden_std_anat is not None:
        anat_mat = (
            anat_golden_seq
            if anat_golden_seq is not None
            else np.stack([series[k] for k in ANATOMICAL_GOLDEN_KEYS], axis=-1).astype(np.float32)
        )
        anat_w = warp_sequence_to_inflection(anat_mat, t_star, target_mid=15)
        z_anat = np.clip(
            np.abs(anat_w - golden_mean_anat) / np.maximum(golden_std_anat, ANATOMICAL_GOLDEN_MIN_STD),
            0.0,
            10.0,
        )
        angle_dev = float(z_anat.mean())
        frame_dev = z_anat.mean(axis=1)
        mid_idx = 15 if len(frame_dev) >= 17 else (len(frame_dev) // 2)
        lo_g = max(0, mid_idx - 1)
        hi_g = min(len(frame_dev), mid_idx + 2)
        dev_inflection = float(np.mean(frame_dev[lo_g:hi_g]))
        dev_max = float(np.max(frame_dev))
        dev_std = float(np.std(frame_dev))
        vals.extend([coord_dev, angle_dev, dev_inflection, dev_max, dev_std])
    else:
        vals.extend([coord_dev, 0.0, 0.0, 0.0, 0.0])

    # Nhóm E: One-hot mã bài tập (8 chiều)
    ex_id = EXERCISE_TO_ID[exercise_name]
    one_hot = [1.0 if i == ex_id else 0.0 for i in range(NUM_EXERCISES)]
    vals.extend(one_hot)

    out = np.asarray(vals, dtype=np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
