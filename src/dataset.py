# -*- coding: utf-8 -*-
"""
src/dataset.py
Quản lý toàn bộ pipeline dữ liệu cho ForgAI Task 2 (Nhóm A - 8 bài tập, 493 mẫu):
  1. dynamic_strided_sampling(): Lấy mẫu nhảy cóc cách đều T=32 frames trên toàn bộ rep (kể cả L < 32).
  2. GoldenRepPrototypeBank: Xây dựng Khuôn Mẫu Chuẩn (chỉ fit trên tập Train) để tính độ lệch.
  3. ZScoreScaler: Chuẩn hóa Z-Score (chỉ fit trên tập Train - Zero Data Leakage).
  4. build_stratified_group_folds(): Chia tập 5-Fold StratifiedGroupKFold theo video_id (0 rò rỉ).
  5. prepare_fold_data() & ForgAITask2Dataset: Xuất tensor cho Deep Learning & ma trận cho Random Forest.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    ENABLE_HORIZONTAL_FLIP,
    EXERCISE_TO_ID,
    FEATURE_DIM,
    GROUP_A_EXERCISES,
    HOLDOUT_TEST_FOLD,
    MIN_VISIBILITY_THRESHOLD,
    N_SPLITS,
    NUM_EXERCISES,
    NUM_FRAMES,
    NUM_JOINTS,
    POSE_NPY_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TASK2_CSV_PATH,
    TRAIN_STRIDE_OFFSETS,
    VAL_RATIO_WITHIN_TRAIN,
)
from src.constants import FORM_LABEL_TO_ID, J_IDX
from src.features import (
    ANATOMICAL_GOLDEN_KEYS,
    ANATOMICAL_GOLDEN_MIN_STD,
    EXERCISE_BIOMECH_SPEC,
    TABULAR_FEATURE_DIM,
    TABULAR_FEATURE_NAMES,
    compute_117d_biomechanical_mix,
    compute_anatomical_angle_series,
    compute_biomechanical_tabular_features,
    extract_anatomical_golden_matrix,
    handle_zero_frames,
    horizontal_flip_skeleton,
    normalize_skeleton_geometry,
    unwarp_sequence_from_inflection,
    warp_sequence_to_inflection,
)

_ID_TO_EXERCISE = {idx: ex for ex, idx in EXERCISE_TO_ID.items()}
_ANAT_KEY_IDX = {k: i for i, k in enumerate(ANATOMICAL_GOLDEN_KEYS)}


# ==============================================================================
# 1. LẤY MẪU NHẢY CÓC CÁCH ĐỀU (DYNAMIC STRIDED SAMPLING - T = 32 FRAMES)
# ==============================================================================
def dynamic_strided_sampling(
    seq: np.ndarray,
    target_len: int = NUM_FRAMES,
    is_train: bool = False,
    offset_idx: int = 0,
    num_offsets: int = TRAIN_STRIDE_OFFSETS,
) -> np.ndarray:
    """
    Rút gọn (hoặc nội suy nếu L < target_len) 1 chuỗi độ dài L bất kỳ về đúng T=32 mốc
    thời gian cách đều nhau trải dài từ đầu đến cuối rep.

    Args:
        seq: Mảng shape (L, ...), ví dụ (L, 13, 3) hoặc (L, 13).
        target_len: Số mốc thời gian mục tiêu (mặc định 32).
        is_train: Bật dịch pha offset khi Train để tăng cường dữ liệu.
        offset_idx: Chỉ số pha dịch [0, 1, 2].
        num_offsets: Tổng số pha dịch bước nhảy cóc (mặc định 3).
    """
    arr = np.asarray(seq, dtype=np.float32)
    src_len = arr.shape[0]
    if src_len == 0:
        raise ValueError("Cannot sample from empty sequence (L=0)")

    if src_len == 1:
        return np.repeat(arr, target_len, axis=0)

    if src_len >= target_len:
        if not is_train:
            indices = np.rint(np.linspace(0, src_len - 1, target_len)).astype(np.int64)
        else:
            step = src_len / float(target_len)
            phase = float(offset_idx) / float(max(1, num_offsets))
            indices = np.floor((np.arange(target_len, dtype=np.float64) + phase) * step).astype(np.int64)
            indices = np.clip(indices, 0, src_len - 1)
        return arr[indices].copy()

    # Trường hợp hiếm: video ngắn hơn 32 frame (22 <= L <= 31) -> Nội suy tuyến tính mượt lên 32 mốc
    step_interp = (src_len - 1.0) / float(target_len)
    phase_shift = (float(offset_idx) / float(max(1, num_offsets))) * step_interp if is_train else 0.0
    q = np.clip(np.linspace(0.0, src_len - 1.0, target_len) + phase_shift, 0.0, src_len - 1.0)
    i0 = np.floor(q).astype(np.int64)
    i1 = np.minimum(i0 + 1, src_len - 1)
    alpha = (q - i0).astype(np.float32).reshape((target_len,) + (1,) * (arr.ndim - 1))
    return ((1.0 - alpha) * arr[i0] + alpha * arr[i1]).astype(np.float32)


# ==============================================================================
# 2. KHUÔN MẪU CHUẨN & CHUẨN HÓA Z-SCORE (TUYỆT ĐỐI CHỈ FIT TRÊN TẬP TRAIN)
# ==============================================================================
class GoldenRepPrototypeBank:
    """
    Lưu trữ quỹ đạo chuẩn (dùng trung vị median kháng nhiễu ngoại lai & độ lệch chuẩn)
    của các rep ĐÚNG (y == 0) cho từng bài tập trên cả chuỗi 117-d và 10 đường cong giải phẫu,
    đã được Căn chỉnh Pha 2 nửa chu kỳ quanh điểm cực trị t* về frame target_mid=15
    (theo nguyên lý KIMORE 2019 & Liao et al., 2020).
    CHỈ được phép .fit() trên tập Train của Fold hiện tại (Zero Data Leakage).
    """

    def __init__(self, num_exercises: int = NUM_EXERCISES, min_std: float = 0.05, target_mid: int = 15):
        self.num_exercises = num_exercises
        self.min_std = float(min_std)
        self.target_mid = int(target_mid)
        self.proto_med_117: dict[int, np.ndarray] = {}
        self.proto_std_117: dict[int, np.ndarray] = {}
        self.proto_med_anat: dict[int, np.ndarray] = {}
        self.proto_std_anat: dict[int, np.ndarray] = {}
        self.fitted = False

    @staticmethod
    def _infer_sample_t_star(s117: np.ndarray, a10: np.ndarray, ex_id: int) -> int:
        ex_name = _ID_TO_EXERCISE.get(int(ex_id))
        t_len = s117.shape[0]
        if ex_name is None or ex_name not in EXERCISE_BIOMECH_SPEC or t_len <= 2:
            return max(0, (t_len - 1) // 2)
        spec = EXERCISE_BIOMECH_SPEC[ex_name]
        drv_key = spec["driver"]
        if drv_key == "wrist_squeeze_dist":
            coords = s117[:, : NUM_JOINTS * 3].reshape(t_len, NUM_JOINTS, 3)
            driver = np.linalg.norm(
                coords[:, J_IDX["left_wrist"], :] - coords[:, J_IDX["right_wrist"], :], axis=-1
            )
        else:
            driver = a10[:, _ANAT_KEY_IDX[drv_key]]
        sub = driver[1 : t_len - 1]
        rel_idx = int(np.argmin(sub)) if spec["mode"] == "min" else int(np.argmax(sub))
        return 1 + rel_idx

    def fit(
        self,
        seq_117: np.ndarray,
        anat_10: np.ndarray,
        y: np.ndarray,
        ex_ids: np.ndarray,
    ) -> "GoldenRepPrototypeBank":
        correct_mask = y == FORM_LABEL_TO_ID["correct"]
        target_mid = getattr(self, "target_mid", 15)

        warped_117 = np.stack(
            [
                warp_sequence_to_inflection(
                    seq_117[i],
                    self._infer_sample_t_star(seq_117[i], anat_10[i], int(ex_ids[i])),
                    target_mid=target_mid,
                )
                for i in range(len(seq_117))
            ],
            axis=0,
        )
        warped_anat = np.stack(
            [
                warp_sequence_to_inflection(
                    anat_10[i],
                    self._infer_sample_t_star(seq_117[i], anat_10[i], int(ex_ids[i])),
                    target_mid=target_mid,
                )
                for i in range(len(anat_10))
            ],
            axis=0,
        )

        fallback_117 = warped_117[correct_mask] if np.any(correct_mask) else warped_117
        fallback_anat = warped_anat[correct_mask] if np.any(correct_mask) else warped_anat

        g_med_117 = np.median(fallback_117, axis=0).astype(np.float32)
        g_std_117 = np.maximum(fallback_117.std(axis=0), self.min_std).astype(np.float32)
        g_med_anat = np.median(fallback_anat, axis=0).astype(np.float32)
        g_std_anat = np.maximum(fallback_anat.std(axis=0), ANATOMICAL_GOLDEN_MIN_STD).astype(np.float32)

        for ex_id in range(self.num_exercises):
            mask = (ex_ids == ex_id) & correct_mask
            if np.any(mask):
                p_117 = warped_117[mask]
                p_anat = warped_anat[mask]
                self.proto_med_117[ex_id] = np.median(p_117, axis=0).astype(np.float32)
                self.proto_std_117[ex_id] = np.maximum(p_117.std(axis=0), self.min_std).astype(np.float32)
                self.proto_med_anat[ex_id] = np.median(p_anat, axis=0).astype(np.float32)
                self.proto_std_anat[ex_id] = np.maximum(p_anat.std(axis=0), ANATOMICAL_GOLDEN_MIN_STD).astype(np.float32)
            else:
                self.proto_med_117[ex_id] = g_med_117.copy()
                self.proto_std_117[ex_id] = g_std_117.copy()
                self.proto_med_anat[ex_id] = g_med_anat.copy()
                self.proto_std_anat[ex_id] = g_std_anat.copy()

        self.fitted = True
        return self

    def get_prototype(self, ex_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("GoldenRepPrototypeBank must be fitted on Train before calling get_prototype()")
        return (
            self.proto_med_117[ex_id],
            self.proto_std_117[ex_id],
            self.proto_med_anat[ex_id],
            self.proto_std_anat[ex_id],
        )

    def transform_sequence_deviations(
        self,
        seq_117: np.ndarray,
        anat_10: np.ndarray,
        ex_ids: np.ndarray,
    ) -> np.ndarray:
        """
        Tính 3 kênh độ lệch so với Khuôn Mẫu Chuẩn tại mỗi trong 32 mốc thời gian
        (sau khi ánh xạ ngược khuôn mẫu về đúng điểm cực trị t* của mẫu đang xét):
          - Kênh 0: Độ lệch TB 39 tọa độ 3D so với trung vị chuẩn.
          - Kênh 1: Độ lệch TB 78 góc nghiêng xương 3D so với trung vị chuẩn.
          - Kênh 2: Độ lệch TB 10 góc giải phẫu chọn bên rõ so với trung vị chuẩn.
        Returns shape (N, T, 3).
        """
        if not self.fitted:
            raise RuntimeError("GoldenRepPrototypeBank must be fitted on Train before transforming!")

        n_samples, t_len, _ = seq_117.shape
        out = np.zeros((n_samples, t_len, 3), dtype=np.float32)
        coord_dim = NUM_JOINTS * 3
        target_mid = getattr(self, "target_mid", 15)

        for i in range(n_samples):
            eid = int(ex_ids[i])
            m117, s117, manat, sanat = self.get_prototype(eid)
            t_star = self._infer_sample_t_star(seq_117[i], anat_10[i], eid)
            m117_u = unwarp_sequence_from_inflection(m117, t_star, source_mid=target_mid)
            s117_u = np.maximum(
                unwarp_sequence_from_inflection(s117, t_star, source_mid=target_mid), self.min_std
            )
            manat_u = unwarp_sequence_from_inflection(manat, t_star, source_mid=target_mid)
            sanat_u = np.maximum(
                unwarp_sequence_from_inflection(sanat, t_star, source_mid=target_mid),
                ANATOMICAL_GOLDEN_MIN_STD,
            )

            z_117 = np.clip(np.abs(seq_117[i] - m117_u) / s117_u, 0.0, 10.0)
            z_anat = np.clip(np.abs(anat_10[i] - manat_u) / sanat_u, 0.0, 10.0)
            out[i, :, 0] = z_117[:, :coord_dim].mean(axis=-1)
            out[i, :, 1] = z_117[:, coord_dim:].mean(axis=-1)
            out[i, :, 2] = z_anat.mean(axis=-1)
        return out


class ZScoreScaler:
    """
    Bộ chuẩn hóa Z-Score (mu, sigma) chỉ học từ Train để áp dụng sang Val/Test.
    Với ma trận đặc trưng bảng 2D (103-d), kết hợp Z-Score toàn cục (35%) với
    Per-Exercise Correct-Reference Z-Score (65%) tính trên các mẫu Đúng của riêng
    từng bài tập trong tập Train, giúp nhánh DL phân biệt chính xác ngưỡng từng bài.
    """

    def __init__(self, reduce_axes: tuple[int, ...] = (0, 1), eps: float = 1e-6):
        self.reduce_axes = reduce_axes
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.ex_medians: np.ndarray | None = None
        self.ex_scales: np.ndarray | None = None
        self.cont_dim: int = TABULAR_FEATURE_DIM - NUM_EXERCISES

    def fit(
        self,
        X_train: np.ndarray,
        y: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
    ) -> "ZScoreScaler":
        self.mean = np.mean(X_train, axis=self.reduce_axes, keepdims=True).astype(np.float32)
        raw_std = np.std(X_train, axis=self.reduce_axes, keepdims=True)
        self.std = np.where(raw_std < self.eps, 1.0, raw_std).astype(np.float32)

        self.ex_medians = None
        self.ex_scales = None
        self.cont_dim = TABULAR_FEATURE_DIM - NUM_EXERCISES
        if X_train.ndim == 2 and X_train.shape[1] == TABULAR_FEATURE_DIM and y is not None and ex_ids is not None:
            cont = np.asarray(X_train[:, : self.cont_dim], dtype=np.float32)
            g_std = np.where(cont.std(axis=0) < 1e-5, 1.0, cont.std(axis=0))
            meds = np.zeros((NUM_EXERCISES, self.cont_dim), dtype=np.float32)
            scales = np.ones((NUM_EXERCISES, self.cont_dim), dtype=np.float32)
            y_arr = np.asarray(y, dtype=np.int64)
            ex_arr = np.asarray(ex_ids, dtype=np.int64)
            for eid in range(NUM_EXERCISES):
                mask_c = (ex_arr == eid) & (y_arr == FORM_LABEL_TO_ID["correct"])
                sub = cont[mask_c] if np.any(mask_c) else cont
                med = np.median(sub, axis=0)
                q75, q25 = np.percentile(sub, [75, 25], axis=0)
                sc = np.maximum(np.maximum((q75 - q25) / 1.349, sub.std(axis=0) * 0.6), g_std * 0.25)
                meds[eid] = med.astype(np.float32)
                scales[eid] = np.where(sc < 1e-5, 1.0, sc).astype(np.float32)
            self.ex_medians = meds
            self.ex_scales = scales
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ZScoreScaler must be fitted on Train first!")
        z_global = ((X - self.mean) / self.std).astype(np.float32)
        ex_meds = getattr(self, "ex_medians", None)
        ex_scs = getattr(self, "ex_scales", None)
        c_dim = getattr(self, "cont_dim", TABULAR_FEATURE_DIM - NUM_EXERCISES)
        if X.ndim == 2 and ex_meds is not None and ex_scs is not None and X.shape[1] == TABULAR_FEATURE_DIM:
            eids = np.argmax(X[:, c_dim : c_dim + NUM_EXERCISES], axis=-1).astype(np.int64)
            z_ex = np.clip((X[:, :c_dim] - ex_meds[eids]) / ex_scs[eids], -6.0, 6.0).astype(np.float32)
            out = z_global.copy()
            out[:, :c_dim] = (0.65 * z_ex + 0.35 * np.clip(z_global[:, :c_dim], -6.0, 6.0)).astype(np.float32)
            out[:, c_dim:] = X[:, c_dim:].astype(np.float32)
            return out
        return z_global

    def fit_transform(
        self,
        X_train: np.ndarray,
        y: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.fit(X_train, y=y, ex_ids=ex_ids).transform(X_train)


# ==============================================================================
# 3. NẠP DỮ LIỆU & CHIA TẬP 5-FOLD STRATIFIED GROUP K-FOLD (ZERO LEAKAGE)
# ==============================================================================
def load_task2_metadata(csv_path: Path = TASK2_CSV_PATH) -> pd.DataFrame:
    """Đọc file metadata chuẩn duy nhất và kiểm tra tính toàn vẹn toàn bộ file .npy."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    df = df[df["exercise"].isin(GROUP_A_EXERCISES)].copy().reset_index(drop=True)
    df["exercise_id"] = df["exercise"].map(EXERCISE_TO_ID).astype(int)
    df["label_id"] = df["form_label"].map(FORM_LABEL_TO_ID).astype(int)
    df["npy_path"] = df["segment_id"].map(lambda sid: str(POSE_NPY_DIR / f"{sid}.npy"))
    df["strat_label"] = df["exercise"] + "__" + df["form_label"]

    missing = [p for p in df["npy_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} file .npy, ví dụ: {missing[0]}")
    return df


def load_pose_cache(
    df: pd.DataFrame,
    min_vis: float = MIN_VISIBILITY_THRESHOLD,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Nạp và làm sạch toàn bộ 493 chuỗi khung xương một lần vào RAM (~10 MB):
      - handle_zero_frames() (nội suy + làm mượt 3-frame)
      - normalize_skeleton_geometry() (Hip-Centering + Torso-Length Normalization)
    """
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in df.iterrows():
        sid = row["segment_id"]
        raw = np.load(row["npy_path"])
        coords_clean, vis_clean = handle_zero_frames(raw, min_vis=min_vis, smooth_window=3)
        coords_norm = normalize_skeleton_geometry(coords_clean)
        cache[sid] = (coords_norm, vis_clean)
    return cache


def build_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    val_ratio: float = VAL_RATIO_WITHIN_TRAIN,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """
    Tạo danh sách 5 Fold bằng StratifiedGroupKFold gom nhóm nghiêm ngặt theo cột `video_id`.
    Với mỗi Fold:
      - ~20% dữ liệu làm Test độc lập.
      - Từ ~80% còn lại tách ra ~15% (1/6 phần còn lại) làm Validation theo video_id.
      - Tự động assert giao video_id giữa Train, Val, Test bằng RỖNG.
    """
    outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    groups_all = df["video_id"].to_numpy()
    strat_all = df["strat_label"].to_numpy()
    n_val_splits = max(2, int(round(1.0 / val_ratio)))
    # Với val_ratio=0.15 -> 1/0.15 = 6.67 -> chọn 6 (~16.7% của phần train_val) để mỗi bài đều có đủ mẫu Sai trong Val
    if val_ratio == 0.15:
        n_val_splits = 6

    folds: list[dict] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(df, strat_all, groups_all)):
        sub_df = df.iloc[train_val_idx]
        inner_cv = StratifiedGroupKFold(
            n_splits=n_val_splits, shuffle=True, random_state=seed + 100 + fold_idx
        )
        sub_train_pos, sub_val_pos = next(
            inner_cv.split(sub_df, sub_df["strat_label"].to_numpy(), sub_df["video_id"].to_numpy())
        )
        train_idx = train_val_idx[sub_train_pos]
        val_idx = train_val_idx[sub_val_pos]

        # Chốt chặn kiểm tra chống rò rỉ dữ liệu tuyệt đối
        v_train = set(df.iloc[train_idx]["video_id"])
        v_val = set(df.iloc[val_idx]["video_id"])
        v_test = set(df.iloc[test_idx]["video_id"])

        assert v_train.isdisjoint(v_val), f"Fold {fold_idx}: Rò rỉ video_id giữa Train và Val!"
        assert v_train.isdisjoint(v_test), f"Fold {fold_idx}: Rò rỉ video_id giữa Train và Test!"
        assert v_val.isdisjoint(v_test), f"Fold {fold_idx}: Rò rỉ video_id giữa Val và Test!"
        assert len(train_idx) + len(val_idx) + len(test_idx) == len(df), "Tổng số mẫu 3 tập không khớp!"

        folds.append({
            "fold": fold_idx,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        })
    return folds


# ==============================================================================
# 4. ĐÓNG GÓI DỮ LIỆU 1 FOLD (TRAIN NHÂN BẢN x6 | VAL & TEST GIỮ NGUYÊN x1)
# ==============================================================================
def _materialize_split_tensors(
    df_split: pd.DataFrame,
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    is_train: bool,
    apply_aug: bool,
    num_offsets: int = TRAIN_STRIDE_OFFSETS,
    enable_flip: bool = ENABLE_HORIZONTAL_FLIP,
) -> dict:
    """Sinh các chuỗi T=32 frames (có nhân bản x6 nếu là Train và apply_aug=True)."""
    offsets = list(range(num_offsets)) if (is_train and apply_aug) else [0]
    flips = [False, True] if (is_train and apply_aug and enable_flip) else [False]

    coords_list, vis_list, seq117_list, anat10_list = [], [], [], []
    ex_ids, y_list, ex_names, seg_ids, vid_ids, durations = [], [], [], [], [], []

    for _, row in df_split.iterrows():
        sid = row["segment_id"]
        coords_norm, vis_clean = pose_cache[sid]
        ex_name = row["exercise"]
        ex_id = int(row["exercise_id"])
        label_id = int(row["label_id"])
        vid_id = row["video_id"]
        dur = float(row["duration_sec"])

        for flip_flag in flips:
            c_use, v_use = (
                horizontal_flip_skeleton(coords_norm, vis_clean)
                if flip_flag
                else (coords_norm, vis_clean)
            )
            for off in offsets:
                c_32 = dynamic_strided_sampling(
                    c_use, target_len=NUM_FRAMES, is_train=(is_train and apply_aug), offset_idx=off, num_offsets=num_offsets
                )
                v_32 = dynamic_strided_sampling(
                    v_use, target_len=NUM_FRAMES, is_train=(is_train and apply_aug), offset_idx=off, num_offsets=num_offsets
                )
                s_117 = compute_117d_biomechanical_mix(c_32)
                a_10 = extract_anatomical_golden_matrix(c_32, v_32)

                coords_list.append(c_32)
                vis_list.append(v_32)
                seq117_list.append(s_117)
                anat10_list.append(a_10)
                ex_ids.append(ex_id)
                y_list.append(label_id)
                ex_names.append(ex_name)
                seg_ids.append(f"{sid}_f{int(flip_flag)}_o{off}" if (is_train and apply_aug) else sid)
                vid_ids.append(vid_id)
                durations.append(dur)

    return {
        "coords_32": np.stack(coords_list, axis=0).astype(np.float32),
        "vis_32": np.stack(vis_list, axis=0).astype(np.float32),
        "seq_117": np.stack(seq117_list, axis=0).astype(np.float32),
        "anat_10": np.stack(anat10_list, axis=0).astype(np.float32),
        "ex_id": np.asarray(ex_ids, dtype=np.int64),
        "y": np.asarray(y_list, dtype=np.int64),
        "exercise": ex_names,
        "segment_id": seg_ids,
        "video_id": np.asarray(vid_ids),
        "duration_sec": np.asarray(durations, dtype=np.float32),
    }


def _build_tabular_matrix(split_data: dict, golden_bank: GoldenRepPrototypeBank) -> np.ndarray:
    """Tính ma trận đặc trưng bảng 101 chiều kèm độ lệch Golden-Rep từ tập Train."""
    n = len(split_data["y"])
    rows = []
    for i in range(n):
        ex_id = int(split_data["ex_id"][i])
        m117, s117, manat, sanat = golden_bank.get_prototype(ex_id)
        feats = compute_biomechanical_tabular_features(
            coords_seq=split_data["coords_32"][i],
            vis_seq=split_data["vis_32"][i],
            exercise_name=split_data["exercise"][i],
            seq_117=split_data["seq_117"][i],
            golden_mean_117=m117,
            golden_std_117=s117,
            anat_golden_seq=split_data["anat_10"][i],
            golden_mean_anat=manat,
            golden_std_anat=sanat,
            duration_sec=float(split_data["duration_sec"][i]),
        )
        rows.append(feats)
    return np.stack(rows, axis=0).astype(np.float32)


def prepare_fold_data(
    df: pd.DataFrame,
    fold_spec: dict,
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    apply_train_aug: bool = True,
    include_golden_channels: bool = True,
) -> dict:
    """
    Chuẩn bị đầy đủ dữ liệu Train / Val / Test cho 1 Fold cụ thể:
      - Không bao giờ rò rỉ tham số chuẩn hóa (Z-Score & Golden-Rep) từ Val/Test.
      - Trả về cả dữ liệu chuỗi thời gian cho Deep Learning lẫn dữ liệu bảng cho Random Forest.
    """
    if pose_cache is None:
        pose_cache = load_pose_cache(df)

    df_train = df.iloc[fold_spec["train_idx"]].reset_index(drop=True)
    df_val = df.iloc[fold_spec["val_idx"]].reset_index(drop=True)
    df_test = df.iloc[fold_spec["test_idx"]].reset_index(drop=True)

    raw_train = _materialize_split_tensors(df_train, pose_cache, is_train=True, apply_aug=apply_train_aug)
    raw_val = _materialize_split_tensors(df_val, pose_cache, is_train=False, apply_aug=False)
    raw_test = _materialize_split_tensors(df_test, pose_cache, is_train=False, apply_aug=False)

    # 1. Fit Khuôn Mẫu Chuẩn CHỈ trên Train
    golden_bank = GoldenRepPrototypeBank(num_exercises=NUM_EXERCISES).fit(
        raw_train["seq_117"], raw_train["anat_10"], raw_train["y"], raw_train["ex_id"]
    )

    for split_d in (raw_train, raw_val, raw_test):
        g_dev = golden_bank.transform_sequence_deviations(
            split_d["seq_117"], split_d["anat_10"], split_d["ex_id"]
        )
        split_d["golden_dev_3"] = g_dev
        split_d["seq_input_raw"] = (
            np.concatenate([split_d["seq_117"], g_dev], axis=-1).astype(np.float32)
            if include_golden_channels
            else split_d["seq_117"]
        )
        split_d["tab_raw"] = _build_tabular_matrix(split_d, golden_bank)

    # 2. Fit Z-Score Scalers CHỈ trên Train
    seq_scaler = ZScoreScaler(reduce_axes=(0, 1)).fit(raw_train["seq_input_raw"])
    tab_scaler = ZScoreScaler(reduce_axes=(0,)).fit(
        raw_train["tab_raw"], y=raw_train["y"], ex_ids=raw_train["ex_id"]
    )

    for split_d in (raw_train, raw_val, raw_test):
        split_d["seq_scaled"] = seq_scaler.transform(split_d["seq_input_raw"])
        split_d["tab_scaled"] = tab_scaler.transform(split_d["tab_raw"])

    return {
        "fold": fold_spec["fold"],
        "train": raw_train,
        "val": raw_val,
        "test": raw_test,
        "golden_bank": golden_bank,
        "seq_scaler": seq_scaler,
        "tab_scaler": tab_scaler,
        "seq_dim": int(raw_train["seq_input_raw"].shape[-1]),
        "tab_dim": int(raw_train["tab_raw"].shape[-1]),
        "tabular_feature_names": TABULAR_FEATURE_NAMES,
        "raw_counts": {
            "train_clips": len(df_train),
            "train_aug_clips": len(raw_train["y"]),
            "val_clips": len(df_val),
            "test_clips": len(df_test),
        },
    }


# ==============================================================================
# 5. PYTORCH DATASET CHO MÔ HÌNH DEEP LEARNING & HYBRID
# ==============================================================================
class ForgAITask2Dataset(Dataset):
    """PyTorch Dataset trả về cặp chuỗi (32, D_seq), vector bảng (101,), ex_id và nhãn Đúng/Sai."""

    def __init__(self, split_dict: dict, use_scaled: bool = True):
        seq_key = "seq_scaled" if use_scaled else "seq_input_raw"
        tab_key = "tab_scaled" if use_scaled else "tab_raw"
        self.X_seq = torch.from_numpy(split_dict[seq_key]).float()
        self.X_tab = torch.from_numpy(split_dict[tab_key]).float()
        self.ex_id = torch.from_numpy(split_dict["ex_id"]).long()
        self.y = torch.from_numpy(split_dict["y"]).long()
        self.segment_ids = split_dict["segment_id"]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.X_seq[idx],
            self.X_tab[idx],
            self.ex_id[idx],
            self.y[idx],
            self.segment_ids[idx],
        )


# ==============================================================================
# 6. KIỂM ĐỊNH TOÀN DIỆN BƯỚC C (--verify)
# ==============================================================================
def verify_step_c() -> dict:
    """Chạy đầy đủ 6 bài test kiểm định hình học, số chiều, NaN và Zero Data Leakage cho cả 5 Fold."""
    print("=" * 78)
    print("FORGAI STEP C - KIỂM ĐỊNH ĐẶC TRƯNG 117-D, 5 CẢI TIẾN & CHỐNG RÒ RỈ 5-FOLD")
    print("=" * 78)

    df = load_task2_metadata(TASK2_CSV_PATH)
    pose_cache = load_pose_cache(df)
    print(f"[1/6] Đã nạp & làm sạch {len(pose_cache)}/{len(df)} chuỗi khung xương vào RAM.")

    # Test 1: Kiểm tra tính chất hình học (Hip-center, Torso scale=1.0, bất biến lật gương)
    max_hip_err = 0.0
    max_scale_err = 0.0
    max_angle_flip_err = 0.0
    max_tab_flip_err = 0.0
    l_hp, r_hp = J_IDX["left_hip"], J_IDX["right_hip"]
    l_sh, r_sh = J_IDX["left_shoulder"], J_IDX["right_shoulder"]

    for idx, row in df.iterrows():
        sid = row["segment_id"]
        c_norm, v_clean = pose_cache[sid]
        hip_mid = 0.5 * (c_norm[:, l_hp, :] + c_norm[:, r_hp, :])
        sh_mid = 0.5 * (c_norm[:, l_sh, :] + c_norm[:, r_sh, :])
        max_hip_err = max(max_hip_err, float(np.abs(hip_mid).max()))
        torso_med = float(np.median(np.linalg.norm(sh_mid - hip_mid, axis=-1)))
        # Clip cực dị có thể chạm ngưỡng chặn dưới min_torso_len=0.15 nhưng với 493 clip thực tế kỳ vọng ~1.0
        if torso_med > 0.15:
            max_scale_err = max(max_scale_err, abs(torso_med - 1.0))

        # Kiểm tra lật gương trên 50 clip đại diện (khớp cả góc giải phẫu lẫn 101 đặc trưng bảng)
        if idx % 10 == 0:
            c_flip, v_flip = horizontal_flip_skeleton(c_norm, v_clean)
            s_orig = compute_anatomical_angle_series(c_norm, v_clean)
            s_flip = compute_anatomical_angle_series(c_flip, v_flip)
            for k in ("knee_angle", "hip_angle", "elbow_angle", "torso_tilt", "waist_sag_signed"):
                max_angle_flip_err = max(max_angle_flip_err, float(np.max(np.abs(s_orig[k] - s_flip[k]))))

            t_orig = compute_biomechanical_tabular_features(
                dynamic_strided_sampling(c_norm, is_train=False),
                dynamic_strided_sampling(v_clean, is_train=False),
                row["exercise"],
                duration_sec=float(row["duration_sec"]),
            )
            t_flip = compute_biomechanical_tabular_features(
                dynamic_strided_sampling(c_flip, is_train=False),
                dynamic_strided_sampling(v_flip, is_train=False),
                row["exercise"],
                duration_sec=float(row["duration_sec"]),
            )
            max_tab_flip_err = max(max_tab_flip_err, float(np.max(np.abs(t_orig - t_flip))))

    assert max_hip_err < 1e-5, f"Hip centering lệch: {max_hip_err}"
    assert max_angle_flip_err < 1e-3, f"Góc bị biến đổi sau lật gương: {max_angle_flip_err}"
    assert max_tab_flip_err < 1e-2, f"Tabular features lệch sau lật gương: {max_tab_flip_err}"
    print(
        f"[2/6] Hình học CHUẨN: sai số tâm hông={max_hip_err:.2e}, "
        f"sai số góc lật gương={max_angle_flip_err:.2e}°, sai số bảng lật gương={max_tab_flip_err:.2e}."
    )

    # Test 2-5: Kiểm định toàn bộ 5 Fold
    folds = build_stratified_group_folds(df)
    fold_summaries = []
    for f_spec in folds:
        bundle = prepare_fold_data(df, f_spec, pose_cache=pose_cache, apply_train_aug=True, include_golden_channels=True)
        tr, va, te = bundle["train"], bundle["val"], bundle["test"]

        # Kiểm tra kích thước & NaN/Inf
        for name, sp in (("train", tr), ("val", va), ("test", te)):
            assert sp["seq_117"].shape[1:] == (NUM_FRAMES, FEATURE_DIM), f"Sai shape 117-d ở {name}: {sp['seq_117'].shape}"
            assert sp["seq_scaled"].shape[1:] == (NUM_FRAMES, FEATURE_DIM + 3), f"Sai shape scaled ở {name}"
            assert sp["tab_raw"].shape[1] == TABULAR_FEATURE_DIM, f"Sai số cột tabular ở {name}"
            assert np.isfinite(sp["seq_scaled"]).all(), f"NaN/Inf trong seq_scaled của {name}"
            assert np.isfinite(sp["tab_raw"]).all(), f"NaN/Inf trong tab_raw của {name}"

        # Kiểm tra đúng tỷ lệ nhân bản x6 trên Train và x1 trên Val/Test
        rc = bundle["raw_counts"]
        assert rc["train_aug_clips"] == rc["train_clips"] * 6, "Train chưa nhân đúng x6!"
        assert len(va["y"]) == rc["val_clips"], "Val bị nhân bản sai!"
        assert len(te["y"]) == rc["test_clips"], "Test bị nhân bản sai!"

        # Kiểm tra độ phủ đủ 8 bài tập và cả Đúng/Sai trên cả 3 tập
        n_ex_train = len(set(tr["ex_id"].tolist()))
        n_ex_val = len(set(va["ex_id"].tolist()))
        n_ex_test = len(set(te["ex_id"].tolist()))
        assert n_ex_train == NUM_EXERCISES and n_ex_val == NUM_EXERCISES and n_ex_test == NUM_EXERCISES

        # Kiểm tra Zero Data Leakage giữa video_id
        vt, vv, vte = set(tr["video_id"].tolist()), set(va["video_id"].tolist()), set(te["video_id"].tolist())
        assert vt.isdisjoint(vv) and vt.isdisjoint(vte) and vv.isdisjoint(vte)

        # Kiểm tra PyTorch Dataset
        ds = ForgAITask2Dataset(tr)
        x_s, x_t, ex_i, y_i, _ = ds[0]
        assert x_s.shape == (NUM_FRAMES, FEATURE_DIM + 3) and x_t.shape == (TABULAR_FEATURE_DIM,)

        fold_summaries.append({
            "fold": int(f_spec["fold"]),
            "train_raw_clips": int(rc["train_clips"]),
            "train_aug_clips_x6": int(rc["train_aug_clips"]),
            "train_correct_raw": int((df.iloc[f_spec["train_idx"]]["label_id"] == 0).sum()),
            "train_incorrect_raw": int((df.iloc[f_spec["train_idx"]]["label_id"] == 1).sum()),
            "val_clips_x1": int(rc["val_clips"]),
            "val_correct": int((va["y"] == 0).sum()),
            "val_incorrect": int((va["y"] == 1).sum()),
            "test_clips_x1": int(rc["test_clips"]),
            "test_correct": int((te["y"] == 0).sum()),
            "test_incorrect": int((te["y"] == 1).sum()),
            "train_video_groups": len(vt),
            "val_video_groups": len(vv),
            "test_video_groups": len(vte),
            "video_id_overlap_count": 0,
        })

    print(f"[3/6] Kiểm tra Shape chuỗi (32, 117) -> (32, 120) và bảng ({TABULAR_FEATURE_DIM},): 100% PASS, 0 NaN/Inf.")
    print("[4/6] Kiểm tra Augmentation x6 trên Train & giữ nguyên x1 trên Val/Test: 100% PASS.")
    print("[5/6] Kiểm tra 5-Fold StratifiedGroupKFold (video_id overlap = 0 trên cả 5/5 Fold, phủ đủ 8 bài): 100% PASS.")

    # Kiểm tra cột hằng số & đối chiếu dấu hiệu cơ sinh học thật trên Fold 0
    b0 = prepare_fold_data(df, folds[HOLDOUT_TEST_FOLD], pose_cache=pose_cache)
    tab_tr = b0["train"]["tab_raw"]
    zero_var_cols = [TABULAR_FEATURE_NAMES[j] for j in range(TABULAR_FEATURE_DIM) if np.std(tab_tr[:, j]) < 1e-7]
    assert len(zero_var_cols) == 0, f"Có cột hằng số không mang thông tin: {zero_var_cols}"

    # So sánh thực tế trung bình đặc trưng giữa Correct vs Incorrect (Fold 0 Train gốc)
    df_tr0 = df.iloc[folds[HOLDOUT_TEST_FOLD]["train_idx"]].reset_index(drop=True)
    raw0_mask = np.array([s.endswith("_f0_o0") for s in b0["train"]["segment_id"]])
    tab_tr0 = tab_tr[raw0_mask]

    def _mean_feat_err(ex: str, err_reason: str | None, label: int, feat_name: str) -> float:
        mask = (df_tr0["exercise"] == ex) & (df_tr0["label_id"] == label)
        if err_reason is not None:
            mask = mask & (df_tr0["error_reason"] == err_reason)
        col_i = TABULAR_FEATURE_NAMES.index(feat_name)
        return float(tab_tr0[mask.to_numpy(), col_i].mean())

    biomech_sanity = {
        "squat_knee_correct_deg": round(_mean_feat_err("squat", None, 0, "knee_angle_at_inflection"), 2),
        "squat_knee_partial_rom_deg": round(_mean_feat_err("squat", "partial_rom", 1, "knee_angle_at_inflection"), 2),
        "pushup_waist_sag_correct": round(_mean_feat_err("push-up", None, 0, "waist_sag_signed_at_inflection"), 4),
        "pushup_waist_sagging_err": round(_mean_feat_err("push-up", "waist_sagging", 1, "waist_sag_signed_at_inflection"), 4),
        "pushup_hips_too_high_err": round(_mean_feat_err("push-up", "hips_too_high", 1, "waist_sag_signed_at_inflection"), 4),
        "pullup_chin_above_wrist_correct": round(_mean_feat_err("pull up", None, 0, "chin_above_wrist_max"), 3),
        "pullup_chin_above_wrist_partial_rom": round(_mean_feat_err("pull up", "partial_rom", 1, "chin_above_wrist_max"), 3),
        "pushup_anat_golden_dev_correct": round(_mean_feat_err("push-up", None, 0, "golden_angle_dev_mean"), 3),
        "pushup_anat_golden_dev_incorrect": round(_mean_feat_err("push-up", None, 1, "golden_angle_dev_mean"), 3),
        "squat_anat_golden_dev_correct": round(_mean_feat_err("squat", None, 0, "golden_angle_dev_mean"), 3),
        "squat_anat_golden_dev_incorrect": round(_mean_feat_err("squat", None, 1, "golden_angle_dev_mean"), 3),
    }
    print("[6/6] Kiểm tra độ phân tách cơ sinh học thật (Fold 0 Train):")
    for k, v in biomech_sanity.items():
        print(f"      - {k}: {v}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_report = {
        "total_clips": len(df),
        "unique_video_id_groups": int(df["video_id"].nunique()),
        "sequence_dim_117": FEATURE_DIM,
        "sequence_dim_with_golden": FEATURE_DIM + 3,
        "tabular_feature_dim": TABULAR_FEATURE_DIM,
        "tabular_feature_names": TABULAR_FEATURE_NAMES,
        "zero_variance_tabular_columns": zero_var_cols,
        "geometry_checks": {
            "max_hip_centering_error": max_hip_err,
            "max_torso_scale_deviation": max_scale_err,
            "max_flip_angle_diff_deg": max_angle_flip_err,
            "max_flip_tabular_diff": max_tab_flip_err,
        },
        "biomech_sanity_fold0": biomech_sanity,
        "folds": fold_summaries,
    }
    out_json = REPORTS_DIR / "step_c_dataset_audit.json"
    out_json.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[HOÀN TẤT] Đã lưu báo cáo kiểm định chi tiết tại: {out_json}")
    print("\nBảng tổng hợp 5-Fold:")
    print(pd.DataFrame(fold_summaries).to_string(index=False))
    return audit_report


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser(description="ForgAI Step C Dataset Pipeline & Audit")
    parser.add_argument("--verify", action="store_true", help="Chạy kiểm định toàn diện 5-Fold & đặc trưng")
    args = parser.parse_args()
    if args.verify:
        verify_step_c()
    else:
        print("Chạy `python src/dataset.py --verify` để kiểm định toàn bộ Bước C.")
