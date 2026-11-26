# -*- coding: utf-8 -*-
"""
src/inference.py
Engine suy luận đóng gói cho Bước 7 (Deployment - ForgAI Task 2):
  - ForgAITask2Predictor: Hỗ trợ nạp 1 Fold (siêu nhanh) hoặc Ensemble cả 5 Folds.
  - Hỗ trợ 2 nguồn đầu vào chính thức:
      1. Chuỗi khung xương .npy có sẵn (predict_from_npy).
      2. File video .mp4 bất kỳ (predict_from_video, tái sử dụng src/extractor.py).
  - Trả về đầy đủ xác suất 4 nhánh (BiomechanicalRF 114-d, CNN-BiLSTM, Transformer, Hybrid),
    Form Score (0..100%), chẩn đoán mã lỗi BiomechanicalRuleEngine, danh sách khớp tô đỏ trên
    video HUD, chuỗi góc khớp chủ đạo theo từng frame và số đo độ trễ từng công đoạn (ms).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from configs.config import (
    CHECKPOINTS_DIR,
    EXERCISE_TO_ID,
    GROUP_A_EXERCISES,
    MIN_VISIBILITY_THRESHOLD,
    N_SPLITS,
    NUM_FRAMES,
)
from src.constants import ERROR_REASON_VN_DICT, J_IDX, SKELETON_EDGES_13
from src.dataset import dynamic_strided_sampling
from src.extractor import extract_pose_sequence
from src.features import (
    EXERCISE_BIOMECH_SPEC,
    TABULAR_FEATURE_DIM,
    compute_117d_biomechanical_mix,
    compute_anatomical_angle_series,
    compute_biomechanical_tabular_features,
    extract_anatomical_golden_matrix,
    handle_zero_frames,
    handle_zero_frames_2d,
    normalize_skeleton_geometry,
)
from src.models import build_model


_UPPER_ARMS = {
    J_IDX["left_shoulder"],
    J_IDX["right_shoulder"],
    J_IDX["left_elbow"],
    J_IDX["right_elbow"],
    J_IDX["left_wrist"],
    J_IDX["right_wrist"],
}
_CORE_SPINE = {
    J_IDX["left_shoulder"],
    J_IDX["right_shoulder"],
    J_IDX["left_hip"],
    J_IDX["right_hip"],
}
_LOWER_LEGS = {
    J_IDX["left_hip"],
    J_IDX["right_hip"],
    J_IDX["left_knee"],
    J_IDX["right_knee"],
    J_IDX["left_ankle"],
    J_IDX["right_ankle"],
}

ERROR_JOINT_HIGHLIGHT_MAP: dict[str, set[int]] = {
    "none": set(),
    "hips_rising_first": _CORE_SPINE,
    "waist_sagging": _CORE_SPINE | {J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "hips_too_high": _CORE_SPINE,
    "snaking_push_up": _CORE_SPINE,
    "arching_lower_back": _CORE_SPINE,
    "hyperextended_back": _CORE_SPINE,
    "round_back": _CORE_SPINE,
    "leaning_forward": _CORE_SPINE,
    "standing_too_upright": _CORE_SPINE,
    "torso_swinging": _CORE_SPINE,
    "hip_flexor_dominance": _LOWER_LEGS,
    "butt_lifting": {J_IDX["left_hip"], J_IDX["right_hip"], J_IDX["left_knee"], J_IDX["right_knee"]},
    "butt_wink": _CORE_SPINE | {J_IDX["left_knee"], J_IDX["right_knee"]},
    "hips_too_low": {J_IDX["left_hip"], J_IDX["right_hip"], J_IDX["left_knee"], J_IDX["right_knee"]},
    "knee_valgus": {J_IDX["left_hip"], J_IDX["right_hip"], J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "heels_lifting": {J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "straight_legs": {J_IDX["left_hip"], J_IDX["right_hip"], J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "improper_foot_placement": {J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "hands_on_knees": {J_IDX["left_wrist"], J_IDX["right_wrist"], J_IDX["left_knee"], J_IDX["right_knee"]},
    "elbow_excessive_flare": {J_IDX["left_shoulder"], J_IDX["right_shoulder"], J_IDX["left_elbow"], J_IDX["right_elbow"]},
    "elbow_lockout": {J_IDX["left_shoulder"], J_IDX["right_shoulder"], J_IDX["left_elbow"], J_IDX["right_elbow"], J_IDX["left_wrist"], J_IDX["right_wrist"]},
    "excessive_elbow_bend": {J_IDX["left_shoulder"], J_IDX["right_shoulder"], J_IDX["left_elbow"], J_IDX["right_elbow"], J_IDX["left_wrist"], J_IDX["right_wrist"]},
    "elbows_dropping": {J_IDX["left_shoulder"], J_IDX["right_shoulder"], J_IDX["left_elbow"], J_IDX["right_elbow"]},
    "unequal_pull": _UPPER_ARMS,
    "shoulders_shrugging": {J_IDX["nose"], J_IDX["left_shoulder"], J_IDX["right_shoulder"]},
    "shoulder_protraction": {J_IDX["left_shoulder"], J_IDX["right_shoulder"], J_IDX["left_elbow"], J_IDX["right_elbow"]},
    "neck_extension": {J_IDX["nose"], J_IDX["left_shoulder"], J_IDX["right_shoulder"]},
    "head_looking_up": {J_IDX["nose"], J_IDX["left_shoulder"], J_IDX["right_shoulder"]},
    "kipping": _CORE_SPINE | {J_IDX["left_knee"], J_IDX["right_knee"]},
    "uncontrolled_eccentric": _CORE_SPINE | {J_IDX["left_elbow"], J_IDX["right_elbow"]},
    "using_momentum": _CORE_SPINE,
}

EXERCISE_PARTIAL_ROM_JOINTS: dict[str, set[int]] = {
    "squat": _LOWER_LEGS,
    "leg extension": {J_IDX["left_hip"], J_IDX["right_hip"], J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "hip thrust": _CORE_SPINE | {J_IDX["left_knee"], J_IDX["right_knee"]},
    "leg raises": _CORE_SPINE | {J_IDX["left_knee"], J_IDX["right_knee"], J_IDX["left_ankle"], J_IDX["right_ankle"]},
    "pull up": _UPPER_ARMS,
    "push-up": _UPPER_ARMS | {J_IDX["left_hip"], J_IDX["right_hip"]},
    "t bar row": _UPPER_ARMS | {J_IDX["left_hip"], J_IDX["right_hip"]},
    "chest fly machine": _UPPER_ARMS,
}


def get_error_highlight_joints(exercise_name: str, error_code: str) -> tuple[set[int], set[tuple[int, int]]]:
    """Trả về tập chỉ số khớp và đoạn xương cần tô đỏ/cam trên HUD khi rep bị lỗi."""
    if not error_code or error_code == "none":
        return set(), set()
    if error_code == "partial_rom":
        joints = set(EXERCISE_PARTIAL_ROM_JOINTS.get(exercise_name, _CORE_SPINE))
    else:
        joints = set(ERROR_JOINT_HIGHLIGHT_MAP.get(error_code, _CORE_SPINE))
    edges = {(a, b) for (a, b) in SKELETON_EDGES_13 if a in joints and b in joints}
    return joints, edges


def _read_video_fps_and_frames(video_path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, total_frames


class _FoldRuntime:
    """Bộ trọng số và scaler đã nạp cho 1 Fold cụ thể."""

    def __init__(self, fold: int, checkpoints_dir: Path, device: torch.device):
        self.fold = int(fold)
        self.device = device
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

        self.cnn_model = self._load_dl("cnn_bilstm", self.ckpt_dir / "cnn_bilstm_best.pt")
        self.trans_model = self._load_dl("transformer", self.ckpt_dir / "transformer_best.pt")

        # Khởi động trước MKL/oneDNN & OpenMP kernel trên 1 mẫu giả để lần suy luận đầu không dính cold-start
        with torch.inference_mode():
            d_xs = torch.zeros((1, NUM_FRAMES, 120), dtype=torch.float32, device=self.device)
            d_xt = torch.zeros((1, TABULAR_FEATURE_DIM), dtype=torch.float32, device=self.device)
            d_ex = torch.zeros((1,), dtype=torch.int64, device=self.device)
            _ = self.cnn_model(d_xs, d_ex, d_xt)
            _ = self.trans_model(d_xs, d_ex, d_xt)
        d_np = np.zeros((1, TABULAR_FEATURE_DIM), dtype=np.float32)
        if hasattr(self.rf_model, "predict_unshifted_proba"):
            _ = self.rf_model.predict_unshifted_proba(d_np, tab_raw=d_np, exercises=["squat"])

    def _load_dl(self, model_name: str, pt_path: Path) -> torch.nn.Module:
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


class ForgAITask2Predictor:
    """
    Engine suy luận đóng gói cho Bước 7 (Deployment - Task 2):
      - fold=0..4 & ensemble_all_folds=False: Nạp nhanh đúng 1 Fold.
      - ensemble_all_folds=True: Nạp đủ 5 Folds để Ensemble toàn bộ hoặc chọn OOF Fold theo mẫu.
    """

    def __init__(
        self,
        fold: int | None = 2,
        ensemble_all_folds: bool = False,
        device: str | torch.device = "cpu",
        checkpoints_dir: Path = CHECKPOINTS_DIR,
    ):
        self.device = torch.device(device)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.ensemble_all_folds = bool(ensemble_all_folds)

        if self.ensemble_all_folds:
            target_folds = list(range(N_SPLITS))
        else:
            if fold is None:
                fold = 2
            target_folds = [int(fold)]

        self.runtimes: dict[int, _FoldRuntime] = {
            k: _FoldRuntime(k, self.checkpoints_dir, self.device) for k in target_folds
        }
        self.default_fold = target_folds[0]

    def _score_on_runtime(
        self,
        rt: _FoldRuntime,
        c32: np.ndarray,
        v32: np.ndarray,
        s117: np.ndarray,
        a10: np.ndarray,
        exercise_name: str,
        duration_sec: float,
        c2d32: np.ndarray | None = None,
    ) -> dict[str, Any]:
        ex_id = int(EXERCISE_TO_ID[exercise_name])
        ex_arr = np.asarray([ex_id], dtype=np.int64)
        m117, std117, manat, sanat = rt.golden_bank.get_prototype(ex_id)
        gdev = rt.golden_bank.transform_sequence_deviations(s117[None], a10[None], ex_arr)[0]
        seq_120 = np.concatenate([s117, gdev], axis=-1)[None]
        x_seq_sc = rt.seq_scaler.transform(seq_120)

        x_tab_raw = compute_biomechanical_tabular_features(
            coords_seq=c32,
            vis_seq=v32,
            exercise_name=exercise_name,
            seq_117=s117,
            golden_mean_117=m117,
            golden_std_117=std117,
            anat_golden_seq=a10,
            golden_mean_anat=manat,
            golden_std_anat=sanat,
            duration_sec=float(duration_sec),
            coords2d_seq=c2d32,
        )
        x_tab_sc = rt.tab_scaler.transform(x_tab_raw[None])

        if hasattr(rt.rf_model, "predict_unshifted_proba"):
            p_rf_u = rt.rf_model.predict_unshifted_proba(x_tab_sc, tab_raw=x_tab_raw[None], exercises=[exercise_name])
            p_rf_cal = rt.rf_model.apply_prior_shift(p_rf_u, exercises=[exercise_name])
        else:
            p_rf_cal = rt.rf_model.predict_proba(x_tab_sc, tab_raw=x_tab_raw[None], exercises=[exercise_name])
            p_rf_u = p_rf_cal

        with torch.inference_mode():
            xs_t = torch.from_numpy(x_seq_sc).to(self.device)
            xt_t = torch.from_numpy(x_tab_sc).to(self.device)
            ex_t = torch.from_numpy(ex_arr).to(self.device)
            p_c = float(torch.sigmoid(rt.cnn_model(xs_t, ex_t, xt_t)).item())
            p_t = float(torch.sigmoid(rt.trans_model(xs_t, ex_t, xt_t)).item())

        probs_dict = {
            "rf": p_rf_cal,
            "cnn_bilstm": np.asarray([[1.0 - p_c, p_c]], dtype=np.float32),
            "transformer": np.asarray([[1.0 - p_t, p_t]], dtype=np.float32),
        }
        p_hyb = float(rt.hybrid_ensemble.predict_proba(probs_dict)[0, 1])
        return {
            "fold": rt.fold,
            "p_rf": float(p_rf_cal[0, 1]),
            "p_rf_unshifted": float(p_rf_u[0, 1]),
            "p_cnn_bilstm": p_c,
            "p_transformer": p_t,
            "p_hybrid": p_hyb,
            "threshold": float(rt.hybrid_ensemble.threshold),
            "x_tab_raw": x_tab_raw,
            "rule_engine": rt.rule_engine,
        }

    def predict_from_pose(
        self,
        pose_seq: np.ndarray,
        exercise_name: str,
        duration_sec: float | None = None,
        fps: float = 30.0,
        fold_override: int | None = None,
    ) -> dict[str, Any]:
        """Suy luận hoàn chỉnh từ chuỗi khung xương (L, 13, 6)."""
        ex = exercise_name.strip().lower()
        if ex not in GROUP_A_EXERCISES:
            raise ValueError(f"Bài tập '{exercise_name}' không thuộc Nhóm A: {GROUP_A_EXERCISES}")

        arr = np.asarray(pose_seq, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (13, 6):
            raise ValueError(f"Expected pose_seq (L, 13, 6), got {arr.shape}")

        dur = float(duration_sec) if duration_sec is not None else float(len(arr) / max(fps, 1.0))

        # Giai đoạn 1: Tiền xử lý hình học & 117-d Biomechanical Mix
        t_feat0 = time.perf_counter()
        coords_clean, vis_clean = handle_zero_frames(arr, min_vis=MIN_VISIBILITY_THRESHOLD, smooth_window=3)
        coords2d_clean = handle_zero_frames_2d(arr, min_vis=MIN_VISIBILITY_THRESHOLD, smooth_window=3)
        coords_norm = normalize_skeleton_geometry(coords_clean)
        full_series = compute_anatomical_angle_series(coords_norm, vis_clean)
        driver_key = EXERCISE_BIOMECH_SPEC[ex]["driver"]
        driver_curve = full_series[driver_key].astype(np.float32)

        c32 = dynamic_strided_sampling(coords_norm, target_len=NUM_FRAMES, is_train=False)
        v32 = dynamic_strided_sampling(vis_clean, target_len=NUM_FRAMES, is_train=False)
        c2d32 = (
            dynamic_strided_sampling(coords2d_clean, target_len=NUM_FRAMES, is_train=False)
            if coords2d_clean is not None
            else None
        )
        s117 = compute_117d_biomechanical_mix(c32)
        a10 = extract_anatomical_golden_matrix(c32, v32)
        feat_ms = (time.perf_counter() - t_feat0) * 1000.0

        # Giai đoạn 2: Suy luận Mô hình (1 Fold chỉ định hoặc Ensemble 5 Folds)
        if fold_override is not None:
            k_fold = int(fold_override)
            if k_fold not in self.runtimes:
                self.runtimes[k_fold] = _FoldRuntime(k_fold, self.checkpoints_dir, self.device)
            active_runtimes = [self.runtimes[k_fold]]
        elif self.ensemble_all_folds:
            active_runtimes = [self.runtimes[k] for k in sorted(self.runtimes.keys())]
        else:
            active_runtimes = [self.runtimes[self.default_fold]]

        t_mod0 = time.perf_counter()

        per_fold_out = [
            self._score_on_runtime(rt, c32, v32, s117, a10, ex, dur, c2d32=c2d32) for rt in active_runtimes
        ]
        model_ms = (time.perf_counter() - t_mod0) * 1000.0

        p_rf = float(np.mean([o["p_rf"] for o in per_fold_out]))
        p_rf_u = float(np.mean([o["p_rf_unshifted"] for o in per_fold_out]))
        p_cnn = float(np.mean([o["p_cnn_bilstm"] for o in per_fold_out]))
        p_trans = float(np.mean([o["p_transformer"] for o in per_fold_out]))
        p_hyb = float(np.mean([o["p_hybrid"] for o in per_fold_out]))
        threshold = float(np.mean([o["threshold"] for o in per_fold_out]))
        is_incorrect = bool(p_hyb >= threshold)

        # Giai đoạn 3: Chẩn đoán cơ sinh học từ BiomechanicalRuleEngine
        t_rule0 = time.perf_counter()
        if len(per_fold_out) == 1:
            rep_eval = per_fold_out[0]
            explain = rep_eval["rule_engine"].explain_rep(
                rep_eval["x_tab_raw"],
                exercise_name=ex,
                p_incorrect=p_hyb,
                is_incorrect=is_incorrect,
                top_k=3,
            )
        else:
            # Khi chạy 5-Fold ensemble: gộp xác suất Top lỗi từ Rule Engine 5 Fold để ổn định nhất
            if not is_incorrect:
                rep_eval = per_fold_out[0]
                explain = rep_eval["rule_engine"].explain_rep(
                    rep_eval["x_tab_raw"],
                    exercise_name=ex,
                    p_incorrect=p_hyb,
                    is_incorrect=False,
                    top_k=3,
                )
            else:
                fold_explains = [
                    o["rule_engine"].explain_rep(
                        o["x_tab_raw"], exercise_name=ex, p_incorrect=p_hyb, is_incorrect=True, top_k=10
                    )
                    for o in per_fold_out
                ]
                score_acc: dict[str, float] = {}
                sample_detail: dict[str, dict] = {}
                for fe in fold_explains:
                    for cand in fe["ranked_details"]:
                        code = cand["error_reason"]
                        score_acc[code] = score_acc.get(code, 0.0) + float(cand["score"])
                        sample_detail.setdefault(code, cand)
                best_code = max(score_acc.items(), key=lambda kv: kv[1])[0]
                # Lấy bản explain chi tiết của fold trùng với best_code (hoặc fold 0)
                chosen = next((fe for fe in fold_explains if fe["predicted_error_code"] == best_code), fold_explains[0])
                explain = chosen

        rule_ms = (time.perf_counter() - t_rule0) * 1000.0
        total_ai_ms = feat_ms + model_ms + rule_ms

        err_code = str(explain["predicted_error_code"])
        hi_joints, hi_edges = get_error_highlight_joints(ex, err_code if is_incorrect else "none")
        det_rate = float((np.abs(arr[:, :, :3]).sum(axis=(1, 2)) > 1e-6).mean())

        return {
            "exercise": ex,
            "folds_used": [o["fold"] for o in per_fold_out],
            "num_frames": int(len(arr)),
            "fps": round(float(fps), 2),
            "duration_sec": round(dur, 2),
            "det_rate": round(det_rate, 4),
            "predicted_label": "INCORRECT" if is_incorrect else "CORRECT",
            "label_vi": "SAI KỸ THUẬT" if is_incorrect else "ĐÚNG KỸ THUẬT",
            "is_incorrect": is_incorrect,
            "form_score_pct": round((1.0 - p_hyb) * 100.0, 1),
            "p_rf": round(p_rf, 4),
            "p_rf_unshifted": round(p_rf_u, 4),
            "p_cnn_bilstm": round(p_cnn, 4),
            "p_transformer": round(p_trans, 4),
            "p_hybrid": round(p_hyb, 4),
            "threshold": round(threshold, 4),
            "driver_key": driver_key,
            "driver_curve": [round(float(v), 2) for v in driver_curve],
            "highlight_joints": sorted(hi_joints),
            "highlight_edges": sorted(hi_edges),
            "timings_ms": {
                "feature_extract_ms": round(feat_ms, 3),
                "model_inference_ms": round(model_ms, 3),
                "rule_engine_ms": round(rule_ms, 3),
                "total_ai_ms": round(total_ai_ms, 3),
                "per_frame_ai_ms": round(total_ai_ms / max(1, len(arr)), 4),
            },
            **explain,
        }

    def predict_from_npy(
        self,
        npy_path: str | Path,
        exercise_name: str,
        duration_sec: float | None = None,
        fps: float = 30.0,
        fold_override: int | None = None,
    ) -> dict[str, Any]:
        """Suy luận trực tiếp từ file khung xương .npy."""
        pose_seq = np.load(str(npy_path))
        res = self.predict_from_pose(
            pose_seq=pose_seq,
            exercise_name=exercise_name,
            duration_sec=duration_sec,
            fps=fps,
            fold_override=fold_override,
        )
        res["input_source"] = str(npy_path)
        return res

    def predict_from_video(
        self,
        video_path: str | Path,
        exercise_name: str,
        start_frame: int = 0,
        end_frame: int | None = None,
        conf_threshold: float = 0.25,
        fold_override: int | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        """Trích xuất MediaPipe Pose 13 khớp từ video .mp4 và chạy suy luận đầy đủ."""
        v_path = Path(video_path)
        fps, total_frames = _read_video_fps_and_frames(v_path)

        t_pose0 = time.perf_counter()
        pose_seq, det_frames, mean_vis = extract_pose_sequence(
            v_path,
            start_frame=start_frame,
            end_frame=end_frame,
            conf_threshold=conf_threshold,
        )
        pose_ms = (time.perf_counter() - t_pose0) * 1000.0
        dur = float(len(pose_seq) / max(fps, 1.0))

        res = self.predict_from_pose(
            pose_seq=pose_seq,
            exercise_name=exercise_name,
            duration_sec=dur,
            fps=fps,
            fold_override=fold_override,
        )
        res["input_source"] = str(v_path)
        res["pose_det_frames"] = int(det_frames)
        res["pose_mean_vis"] = round(float(mean_vis), 3)
        res["timings_ms"]["mediapipe_extract_ms"] = round(pose_ms, 2)
        res["timings_ms"]["mediapipe_per_frame_ms"] = round(pose_ms / max(1, len(pose_seq)), 2)
        res["timings_ms"]["end_to_end_total_ms"] = round(pose_ms + res["timings_ms"]["total_ai_ms"], 2)
        res["timings_ms"]["end_to_end_per_frame_ms"] = round(
            (pose_ms + res["timings_ms"]["total_ai_ms"]) / max(1, len(pose_seq)), 2
        )
        return res, pose_seq

    def benchmark_latency(
        self,
        pose_seq: np.ndarray,
        exercise_name: str,
        duration_sec: float = 2.5,
        n_runs: int = 100,
        fold_override: int | None = None,
    ) -> dict[str, Any]:
        """Đo kiểm tốc độ trung bình & p95 qua n_runs lần chạy liên tiếp (loại bỏ lượt warmup đầu)."""
        _ = self.predict_from_pose(pose_seq, exercise_name, duration_sec=duration_sec, fold_override=fold_override)
        feat_list, mod_list, rule_list, tot_list = [], [], [], []
        for _ in range(n_runs):
            out = self.predict_from_pose(
                pose_seq, exercise_name, duration_sec=duration_sec, fold_override=fold_override
            )
            tm = out["timings_ms"]
            feat_list.append(tm["feature_extract_ms"])
            mod_list.append(tm["model_inference_ms"])
            rule_list.append(tm["rule_engine_ms"])
            tot_list.append(tm["total_ai_ms"])

        mean_tot = float(np.mean(tot_list))
        per_frame = mean_tot / max(1, len(pose_seq))
        return {
            "n_runs": int(n_runs),
            "mode": "5-Fold Ensemble" if (self.ensemble_all_folds and fold_override is None) else f"Single-Fold ({fold_override if fold_override is not None else self.default_fold})",
            "exercise": exercise_name,
            "num_frames_rep": int(len(pose_seq)),
            "feature_extract_mean_ms": round(float(np.mean(feat_list)), 3),
            "model_inference_mean_ms": round(float(np.mean(mod_list)), 3),
            "rule_engine_mean_ms": round(float(np.mean(rule_list)), 3),
            "total_ai_mean_ms": round(mean_tot, 3),
            "total_ai_p95_ms": round(float(np.percentile(tot_list, 95)), 3),
            "per_frame_ai_ms": round(per_frame, 4),
            "core_ai_fps_equivalent": round(1000.0 / max(per_frame, 1e-6), 1),
        }
