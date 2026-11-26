# -*- coding: utf-8 -*-
"""
src/models.py
Định nghĩa toàn bộ 4 mô hình phân loại kỹ thuật Đúng / Sai (Task 2) và hàm mất mát chống lệch lớp:
  1. FiLMLayer & SpatialDropout1d: Các khối điều biến theo bài tập & chống học vẹt chuỗi khung xương.
  2. FocalBCEWithLogitsLoss: Hàm loss nhị phân cân bằng lớp + Focal + Label Smoothing (0.03).
  3. BiomechanicalRandomForest (Model 1 - Baseline): Tổ hợp RandomForest + ExtraTrees trên 111 đặc trưng bảng (+ 3 điểm Luật = 114-d).
  4. ExerciseConditionedCNNBiLSTM (Model 2 - Deep Learning 1): Multi-Scale 1D-CNN + FiLM + Bi-LSTM 2 tầng
     + Temporal Attention + nhánh Gated Tabular (167,120 tham số).
  5. ExerciseConditionedTransformer (Model 3 - Deep Learning 2): Pre-LN Transformer + FiLM + [TAB] Token
     (96,578 tham số).
  6. HybridFormEnsemble (Model 4): Tổ hợp xác suất 3 mô hình & dò ngưỡng tối ưu Macro-F1 CHỈ trên Validation.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EXERCISE_EMBED_DIM,
    EXERCISE_TO_ID,
    FEATURE_DIM,
    HIDDEN_DIM,
    HOLDOUT_TEST_FOLD,
    MIN_RF_ENSEMBLE_WEIGHT,
    NUM_EXERCISES,
    NUM_FRAMES,
    RANDOM_SEED,
)
from src.features import TABULAR_FEATURE_DIM, TABULAR_FEATURE_NAMES

RF_CONT_DIM = TABULAR_FEATURE_DIM - NUM_EXERCISES
RF_RULE_FEATURE_NAMES = (
    "rule_top1_score",
    "rule_top2_score",
    "rule_top1_rule_score",
)
RF_TABULAR_FEATURE_NAMES = (
    list(TABULAR_FEATURE_NAMES[:RF_CONT_DIM])
    + list(RF_RULE_FEATURE_NAMES)
    + list(TABULAR_FEATURE_NAMES[RF_CONT_DIM:])
)


# ==============================================================================
# 1. CÁC KHỐI CHỐNG OVERFITTING, ĐIỀU BIẾN BÀI TẬP & HÀM MẤT MÁT FOCAL BCE
# ==============================================================================
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM):
      x_out = (1 + gamma(cond)) * x + beta(cond)
    Khởi tạo zero-init cho (gamma, beta) để ở epoch 0 tín hiệu đi qua nguyên vẹn,
    sau đó mạng tự học khuếch đại/triệt tiêu kênh khớp phù hợp từng bài tập.
    """

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, feature_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        params = self.proj(cond)
        gamma, beta = params.chunk(2, dim=-1)
        if x.ndim == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1.0 + gamma) * x + beta


class SpatialDropout1d(nn.Module):
    """
    Khóa ngẫu nhiên nguyên cả kênh đặc trưng xuyên suốt toàn bộ T=32 khung hình (B, C, T)
    trong khi Train, ép mô hình không được phụ thuộc cứng vào 1 khớp đơn lẻ dễ rung nhiễu.
    """

    def __init__(self, p: float = 0.30):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = torch.empty((x.shape[0], x.shape[1], 1), device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask.div_(keep_prob)


def compute_pos_weight(y_train: np.ndarray, max_weight: float = 4.0) -> float:
    """Tính trọng số cho lớp Dương (incorrect = 1) CHỈ trên tập Train: N_correct / N_incorrect."""
    y_arr = np.asarray(y_train, dtype=np.int64)
    n_pos = max(1, int((y_arr == 1).sum()))
    n_neg = max(1, int((y_arr == 0).sum()))
    return float(min(max_weight, max(1.0, n_neg / float(n_pos))))


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Hàm mất mát nhị phân Focal + cân bằng lớp + Label Smoothing:
      - pos_weight: Phạt nặng hơn khi bỏ sót mẫu Sai (tỷ lệ ~2:1 Đúng/Sai).
      - gamma: Giảm gradient của mẫu dễ, dồn lực học vào ca khó ở ranh giới Đúng/Sai.
      - label_smoothing: Giữ nhãn mềm [0.015, 0.985] tránh nổ logit khi Train x6.
    """

    def __init__(
        self,
        pos_weight: float | None = None,
        gamma: float = 1.5,
        label_smoothing: float = 0.03,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight) if pos_weight is not None else None
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.view(-1)
        y = targets.float().view(-1)
        y_smooth = y * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(logits_flat, y_smooth, reduction="none")
        probs = torch.sigmoid(logits_flat)
        pt = torch.where(y > 0.5, probs, 1.0 - probs).clamp(1e-6, 1.0 - 1e-6)
        loss = ((1.0 - pt) ** self.gamma) * bce

        if self.pos_weight is not None:
            w = torch.where(y > 0.5, self.pos_weight, 1.0)
            loss = loss * w
        return loss.mean()


def count_parameters(model: nn.Module) -> int:
    """Đếm tổng số tham số đang được huấn luyện của mô hình PyTorch."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def predict_torch_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Chạy suy luận theo batch trả về xác suất 2 lớp (probs [N, 2], y_true [N])."""
    model.eval()
    probs_all, y_all = [], []
    for batch in loader:
        x_seq = batch[0].to(device)
        x_tab = batch[1].to(device)
        ex_id = batch[2].to(device)
        y = batch[3]

        logits = model(x_seq, ex_id, x_tab).view(-1)
        p1 = torch.sigmoid(logits).cpu().numpy()
        probs = np.stack([1.0 - p1, p1], axis=-1).astype(np.float32)
        probs_all.append(probs)
        y_all.append(y.numpy())
    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)


def extract_rule_summary_features(
    rule_engine,
    tab_raw: np.ndarray,
    exercises: list[str] | np.ndarray,
) -> np.ndarray:
    """
    Trích xuất 3 chỉ số tóm tắt từ BiomechanicalRuleEngine (chỉ fit trên Train):
      - rule_top1_score: Xác suất lỗi cao nhất kết hợp Tầng 1 + Tầng 2.
      - rule_top2_score: Xác suất lỗi cao thứ hai.
      - rule_top1_rule_score: Điểm vi phạm thuần vật lý của lỗi đứng đầu.
    """
    diags = rule_engine.explain_batch(tab_raw, exercises, top_k=3)
    s1 = np.array(
        [d["top3_candidates"][0]["score"] if len(d["top3_candidates"]) > 0 else 0.0 for d in diags],
        dtype=np.float32,
    )
    s2 = np.array(
        [d["top3_candidates"][1]["score"] if len(d["top3_candidates"]) > 1 else 0.0 for d in diags],
        dtype=np.float32,
    )
    r1 = np.array(
        [d["top3_candidates"][0]["rule_score"] if len(d["top3_candidates"]) > 0 else 0.0 for d in diags],
        dtype=np.float32,
    )
    return np.stack([s1, s2, r1], axis=-1).astype(np.float32)


# ==============================================================================
# 2. MÔ HÌNH 1 (BASELINE / TABULAR E.3): BIOMECHANICAL RANDOM FOREST (103-D)
# ==============================================================================
class BiomechanicalRandomForest:
    """
    Nhánh mô hình Tabular cải tiến (Bước 6.2 / E.3-LowRisk) dựa trên các công trình:
      - Liao et al. (IEEE TNSRE 2020) & Yu et al. (ICCV 2021): Chuẩn hóa tại chỗ
        các đặc trưng liên tục (RF_CONT_DIM) theo phân phối mẫu ĐÚNG (y == 0) của riêng từng bài tập
        (Per-Exercise Correct-Reference Z-Score), tránh lẫn thang đo giữa 8 bài.
      - Capecci et al. (KIMORE, IEEE TNSRE 2019): Ghép thêm 3 chỉ số tổng hợp từ
        BiomechanicalRuleEngine + 8 One-Hot bài tập -> RF_TABULAR_FEATURE_NAMES (114 chiều).
      - Chống overfit Augmentation x6: RandomForest & ExtraTrees học trên toàn bộ Train;
        HistGradientBoostingClassifier (cây nông depth=4, L2=3.0) CHỈ học trên mẫu gốc
        không nhân bản (_f0_o0).
      - Niculescu-Mizil & Caruana (ICML 2005): Hiệu chỉnh Log-Odds theo tỷ lệ lớp tập
        Train (prior_shift_alpha=0.25) cho suy luận độc lập, không dò tham số trên Val.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        rf_max_depth: int = 10,
        et_max_depth: int = 12,
        hgb_max_iter: int = 100,
        hgb_max_depth: int = 4,
        hgb_min_samples_leaf: int = 10,
        hgb_l2_regularization: float = 3.0,
        model_weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
        prior_shift_alpha: float = 0.25,
        clip_z: float = 6.0,
        random_state: int = RANDOM_SEED,
    ):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
        self.et = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=et_max_depth,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state + 1,
            n_jobs=-1,
        )
        self.hgb = HistGradientBoostingClassifier(
            max_iter=hgb_max_iter,
            max_depth=hgb_max_depth,
            min_samples_leaf=hgb_min_samples_leaf,
            learning_rate=0.05,
            l2_regularization=hgb_l2_regularization,
            random_state=random_state + 2,
        )
        self.model_weights = tuple(float(w) for w in model_weights)
        self.prior_shift_alpha = float(prior_shift_alpha)
        self.clip_z = float(clip_z)

        self.cor_med: dict[int, np.ndarray] = {}
        self.cor_scale: dict[int, np.ndarray] = {}
        self.rule_engine = None
        self.tab_scaler = None
        self.id_to_exercise = {idx: ex for ex, idx in EXERCISE_TO_ID.items()}
        self.train_pos_weight: float = 1.0
        self.ex_pos_weights: dict[int, float] = {}
        self.use_e3_transform: bool = False
        self.feature_names: list[str] = list(TABULAR_FEATURE_NAMES)
        self.fitted = False

    def _infer_ex_ids(self, tab_raw: np.ndarray, ex_ids: np.ndarray | None, exercises: list[str] | np.ndarray | None) -> np.ndarray:
        if ex_ids is not None:
            return np.asarray(ex_ids, dtype=np.int64)
        if exercises is not None:
            return np.asarray([EXERCISE_TO_ID[str(e)] for e in exercises], dtype=np.int64)
        return np.argmax(tab_raw[:, RF_CONT_DIM : RF_CONT_DIM + NUM_EXERCISES], axis=-1).astype(np.int64)

    def _fit_per_exercise_correct_stats(self, tab_raw: np.ndarray, y: np.ndarray, ex_ids: np.ndarray) -> None:
        cont = np.asarray(tab_raw[:, :RF_CONT_DIM], dtype=np.float32)
        g_std = np.where(cont.std(axis=0) < 1e-5, 1.0, cont.std(axis=0))
        self.cor_med = {}
        self.cor_scale = {}
        for ex_id in range(NUM_EXERCISES):
            mask_c = (ex_ids == ex_id) & (y == 0)
            sub = cont[mask_c] if np.any(mask_c) else cont
            med = np.median(sub, axis=0)
            q75, q25 = np.percentile(sub, [75, 25], axis=0)
            scale = np.maximum(np.maximum((q75 - q25) / 1.349, sub.std(axis=0) * 0.6), g_std * 0.25)
            self.cor_med[ex_id] = med.astype(np.float32)
            self.cor_scale[ex_id] = np.where(scale < 1e-5, 1.0, scale).astype(np.float32)

    def _build_104d_features(
        self,
        tab_raw: np.ndarray,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        arr_raw = np.asarray(tab_raw, dtype=np.float32)
        eids = self._infer_ex_ids(arr_raw, ex_ids, exercises)
        ex_list = (
            [str(e) for e in exercises]
            if exercises is not None
            else [self.id_to_exercise[int(e)] for e in eids]
        )

        cont = arr_raw[:, :RF_CONT_DIM]
        z_ex = np.zeros_like(cont, dtype=np.float32)
        for i in range(len(cont)):
            e = int(eids[i])
            z_ex[i] = np.clip((cont[i] - self.cor_med[e]) / self.cor_scale[e], -self.clip_z, self.clip_z)

        if self.rule_engine is not None:
            rule_feats = extract_rule_summary_features(self.rule_engine, arr_raw, ex_list)
        else:
            rule_feats = np.zeros((len(arr_raw), len(RF_RULE_FEATURE_NAMES)), dtype=np.float32)

        ex_onehot = arr_raw[:, RF_CONT_DIM : RF_CONT_DIM + NUM_EXERCISES]
        return np.concatenate([z_ex, rule_feats, ex_onehot], axis=-1).astype(np.float32)

    def _resolve_features(
        self,
        X_tab: np.ndarray,
        tab_raw: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.use_e3_transform:
            return np.asarray(X_tab, dtype=np.float32)
        if tab_raw is not None:
            return self._build_104d_features(tab_raw, ex_ids=ex_ids, exercises=exercises)
        arr = np.asarray(X_tab, dtype=np.float32)
        if arr.shape[1] == len(RF_TABULAR_FEATURE_NAMES):
            return arr
        if self.tab_scaler is not None and arr.shape[1] == TABULAR_FEATURE_DIM:
            recovered_raw = (arr * self.tab_scaler.std + self.tab_scaler.mean).astype(np.float32)
            return self._build_104d_features(recovered_raw, ex_ids=ex_ids, exercises=exercises)
        return self._build_104d_features(arr, ex_ids=ex_ids, exercises=exercises)

    def fit(
        self,
        X_tab_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        tab_raw: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
        rule_engine=None,
        raw_mask: np.ndarray | None = None,
        tab_scaler=None,
    ) -> "BiomechanicalRandomForest":
        y_arr = np.asarray(y_train, dtype=np.int64)
        self.tab_scaler = tab_scaler
        self.rule_engine = rule_engine

        if tab_raw is not None:
            self.use_e3_transform = True
            self.feature_names = list(RF_TABULAR_FEATURE_NAMES)
            raw_arr = np.asarray(tab_raw, dtype=np.float32)
            eids = self._infer_ex_ids(raw_arr, ex_ids, exercises)
            self._fit_per_exercise_correct_stats(raw_arr, y_arr, eids)
            X_fit = self._build_104d_features(raw_arr, ex_ids=eids, exercises=exercises)
        else:
            self.use_e3_transform = False
            self.feature_names = list(TABULAR_FEATURE_NAMES)
            X_fit = np.asarray(X_tab_train, dtype=np.float32)

        self.rf.n_jobs = -1
        self.et.n_jobs = -1
        self.rf.fit(X_fit, y_arr, sample_weight=sample_weight)
        self.et.fit(X_fit, y_arr, sample_weight=sample_weight)
        # Sau khi fit xong, chuyển n_jobs=1 để tránh chi phí tạo threadpool joblib khi suy luận 1 rep realtime
        self.rf.n_jobs = 1
        self.et.n_jobs = 1

        # Huấn luyện HistGradientBoosting CHỈ trên tập mẫu gốc không nhân bản (raw_mask)
        if raw_mask is not None and np.any(raw_mask):
            hgb_mask = np.asarray(raw_mask, dtype=bool)
        else:
            hgb_mask = np.ones(len(y_arr), dtype=bool)

        y_hgb = y_arr[hgb_mask]
        pos_w = float((y_hgb == 0).sum() / max(1, (y_hgb == 1).sum()))
        self.train_pos_weight = pos_w
        self.ex_pos_weights = {}
        if tab_raw is not None:
            eids_all = eids
        elif ex_ids is not None or exercises is not None:
            eids_all = self._infer_ex_ids(X_fit, ex_ids, exercises)
        elif X_fit.shape[1] >= RF_CONT_DIM + NUM_EXERCISES:
            eids_all = np.argmax(X_fit[:, RF_CONT_DIM : RF_CONT_DIM + NUM_EXERCISES], axis=-1).astype(np.int64)
        else:
            eids_all = None

        if eids_all is not None:
            eids_hgb = eids_all[hgb_mask]
            for eid in range(NUM_EXERCISES):
                m_e = eids_hgb == eid
                if np.any(m_e):
                    ye = y_hgb[m_e]
                    ex_pw = float((ye == 0).sum() / max(1, (ye == 1).sum()))
                else:
                    ex_pw = pos_w
                self.ex_pos_weights[eid] = float(np.clip(0.75 * pos_w + 0.25 * ex_pw, 1.10, 2.20))

        sw_hgb = np.where(y_hgb == 1, pos_w, 1.0).astype(np.float64)
        self.hgb.fit(X_fit[hgb_mask], y_hgb, sample_weight=sw_hgb)

        self.fitted = True
        return self

    def predict_unshifted_proba(
        self,
        X_tab: np.ndarray,
        tab_raw: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Trả về xác suất thô (chưa dịch Log-Odds) dùng làm đầu vào cho HybridFormEnsemble."""
        if not self.fitted:
            raise RuntimeError("BiomechanicalRandomForest must be fitted before predict_proba()!")
        X_use = self._resolve_features(X_tab, tab_raw=tab_raw, ex_ids=ex_ids, exercises=exercises)
        w_rf, w_et, w_hgb = self.model_weights
        p1 = (
            w_rf * self.rf.predict_proba(X_use)[:, 1]
            + w_et * self.et.predict_proba(X_use)[:, 1]
            + w_hgb * self.hgb.predict_proba(X_use)[:, 1]
        )
        p1 = np.clip(p1, 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=-1).astype(np.float32)

    def apply_prior_shift(
        self,
        probs: np.ndarray,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Áp dụng phép dịch Log-Odds theo tỷ lệ lớp tập Train (pha trộn theo bài tập) lên xác suất thô."""
        if self.prior_shift_alpha <= 0.0 or self.train_pos_weight <= 1.0:
            return probs
        eids = ex_ids
        if eids is None and exercises is not None:
            eids = np.asarray([EXERCISE_TO_ID[str(e)] for e in exercises], dtype=np.int64)
        ex_pw_map = getattr(self, "ex_pos_weights", None)
        if eids is not None and ex_pw_map:
            pw = np.asarray([ex_pw_map.get(int(e), self.train_pos_weight) for e in eids], dtype=np.float64)
            pw = np.maximum(pw, 1.0)
        else:
            pw = float(self.train_pos_weight)
        p1 = np.clip(probs[:, 1].astype(np.float64), 1e-4, 1.0 - 1e-4)
        logit = np.log(p1 / (1.0 - p1)) + self.prior_shift_alpha * np.log(pw)
        p1_cal = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0)
        return np.stack([1.0 - p1_cal, p1_cal], axis=-1).astype(np.float32)

    def predict_proba(
        self,
        X_tab: np.ndarray,
        tab_raw: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Trả về xác suất đã hiệu chỉnh Log-Odds theo tỷ lệ lớp tập Train (không chạm Val)."""
        probs = self.predict_unshifted_proba(X_tab, tab_raw=tab_raw, ex_ids=ex_ids, exercises=exercises)
        eids = ex_ids
        if eids is None and exercises is None and tab_raw is not None:
            eids = self._infer_ex_ids(np.asarray(tab_raw, dtype=np.float32), None, None)
        return self.apply_prior_shift(probs, ex_ids=eids, exercises=exercises)

    def predict(
        self,
        X_tab: np.ndarray,
        threshold: float = 0.50,
        tab_raw: np.ndarray | None = None,
        ex_ids: np.ndarray | None = None,
        exercises: list[str] | np.ndarray | None = None,
    ) -> np.ndarray:
        probs = self.predict_proba(X_tab, tab_raw=tab_raw, ex_ids=ex_ids, exercises=exercises)[:, 1]
        return (probs >= threshold).astype(np.int64)

    def get_feature_importances(self) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("BiomechanicalRandomForest must be fitted before get_feature_importances()!")
        w_rf, w_et, _ = self.model_weights
        w_sum = max(1e-6, w_rf + w_et)
        return ((w_rf * self.rf.feature_importances_ + w_et * self.et.feature_importances_) / w_sum).astype(np.float64)

    def get_top_features(
        self,
        feature_names: list[str] | tuple[str, ...] | None = None,
        top_k: int = 15,
    ) -> list[tuple[str, float]]:
        if not self.fitted:
            raise RuntimeError("BiomechanicalRandomForest must be fitted before get_top_features()!")
        names = list(feature_names) if feature_names is not None else self.feature_names
        importances = self.get_feature_importances()
        top_idx = np.argsort(importances)[::-1][:top_k]
        return [(str(names[i]), float(importances[i])) for i in top_idx]


# ==============================================================================
# 3. MÔ HÌNH 2 (DEEP LEARNING 1): EXERCISE-CONDITIONED MULTI-SCALE CNN + BI-LSTM
# ==============================================================================
class ExerciseConditionedCNNBiLSTM(nn.Module):
    """
    Kiến trúc Deep Learning lai chuỗi-bảng chuyên biệt cho Task 2:
      - Tầng 1 (FiLM): Khuếch đại/triệt tiêu 120 kênh chuỗi theo mã bài tập ex_id.
      - Tầng 2 (Multi-Scale 1D-CNN): Hai nhánh Conv1d song song (kernel 3 & 5) + SpatialDropout1d(0.30).
      - Tầng 3 (Bi-LSTM 2 tầng): hidden_size=48 mỗi chiều (96-d) đọc trọn vẹn chu kỳ xuôi/ngược.
      - Tầng 4 (Attention + MaxPool + Gated Tabular): Hội tụ tại frame quyết định kết hợp bảng 111-d.
    """

    def __init__(
        self,
        seq_dim: int = FEATURE_DIM + 3,
        tab_dim: int = TABULAR_FEATURE_DIM,
        num_exercises: int = NUM_EXERCISES,
        ex_embed_dim: int = EXERCISE_EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT_RATE,
        spatial_dropout: float = 0.30,
    ):
        super().__init__()
        self.ex_embed = nn.Embedding(num_exercises, ex_embed_dim)
        self.seq_film = FiLMLayer(ex_embed_dim, seq_dim)

        in_channels = seq_dim + ex_embed_dim
        branch_ch = hidden_dim // 2  # 32 + 32 = 64 kênh sau khi nối
        self.conv_k3 = nn.Conv1d(in_channels, branch_ch, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(in_channels, branch_ch, kernel_size=5, padding=2)
        self.conv_post = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            SpatialDropout1d(spatial_dropout),
        )

        lstm_hidden = (hidden_dim * 3) // 4  # 48 mỗi chiều -> 96-d hai chiều
        lstm_out_dim = lstm_hidden * 2
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.attn_net = nn.Sequential(
            nn.Linear(lstm_out_dim + ex_embed_dim, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1),
        )

        tab_proj_dim = 32
        self.tab_proj = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, tab_proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        seq_summary_dim = lstm_out_dim * 2 + ex_embed_dim
        self.tab_gate = nn.Linear(seq_summary_dim, tab_proj_dim)

        fused_dim = seq_summary_dim + tab_proj_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward_with_attention(
        self,
        x_seq: torch.Tensor,
        ex_id: torch.Tensor,
        x_tab: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = x_seq.shape
        ex_vec = self.ex_embed(ex_id)                         # (B, E)
        x_mod = self.seq_film(x_seq, ex_vec)                  # (B, T, D_seq)
        ex_seq = ex_vec.unsqueeze(1).expand(bsz, t_len, -1)   # (B, T, E)

        conv_in = torch.cat([x_mod, ex_seq], dim=-1).transpose(1, 2)  # (B, D_seq+E, T)
        c3 = self.conv_k3(conv_in)
        c5 = self.conv_k5(conv_in)
        h = self.conv_post(torch.cat([c3, c5], dim=1)).transpose(1, 2)  # (B, T, H)

        lstm_out, _ = self.bilstm(h)                          # (B, T, 2*L)
        attn_scores = self.attn_net(torch.cat([lstm_out, ex_seq], dim=-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)      # (B, T)

        attn_pool = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        max_pool = lstm_out.max(dim=1).values
        seq_summary = torch.cat([attn_pool, max_pool, ex_vec], dim=-1)

        if x_tab is not None:
            tab_feat = self.tab_proj(x_tab)
            gate = torch.sigmoid(self.tab_gate(seq_summary))
            tab_gated = gate * tab_feat
        else:
            tab_gated = torch.zeros((bsz, 32), device=x_seq.device, dtype=x_seq.dtype)

        logits = self.classifier(torch.cat([seq_summary, tab_gated], dim=-1))
        return logits, attn_weights

    def forward(
        self,
        x_seq: torch.Tensor,
        ex_id: torch.Tensor,
        x_tab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_attention(x_seq, ex_id, x_tab)
        return logits


# ==============================================================================
# 4. MÔ HÌNH 3 (DEEP LEARNING 2): EXERCISE-CONDITIONED PRE-LN TRANSFORMER
# ==============================================================================
class ExerciseConditionedTransformer(nn.Module):
    """
    Kế thừa SkelGym Temporal Transformer + Cải tiến [TAB] Summary Token & FiLM:
      - Chiếu chuỗi (B, 32, 120) -> d_model=64 + Positional Encoding + FiLM điều biến theo bài tập.
      - Chiếu vector bảng 111-d thành 1 Token tổng hợp ([TAB] Token) ghép vào đầu chuỗi -> (B, 33, 64),
        cho phép Self-Attention đối chiếu trực tiếp từng frame với các chỉ số đáy & thời lượng rep.
      - Pre-LN TransformerEncoder 2 tầng (norm_first=True) hội tụ ổn định, không cần warmup phức tạp.
    """

    def __init__(
        self,
        seq_dim: int = FEATURE_DIM + 3,
        tab_dim: int = TABULAR_FEATURE_DIM,
        num_frames: int = NUM_FRAMES,
        num_exercises: int = NUM_EXERCISES,
        ex_embed_dim: int = EXERCISE_EMBED_DIM,
        d_model: int = HIDDEN_DIM,
        nhead: int = 4,
        dim_feedforward: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
        cls_dropout: float = DROPOUT_RATE,
    ):
        super().__init__()
        self.d_model = d_model
        self.ex_embed = nn.Embedding(num_exercises, ex_embed_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(seq_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout * 0.5),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.seq_film = FiLMLayer(ex_embed_dim, d_model)

        self.tab_to_token = nn.Sequential(
            nn.Linear(tab_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fallback_tab_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.fallback_tab_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.out_norm = nn.LayerNorm(d_model)

        self.attn_net = nn.Sequential(
            nn.Linear(d_model + ex_embed_dim, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        head_in = d_model * 2 + ex_embed_dim  # [TAB] + AttnPool(frames) + ex_vec = 144
        self.classifier = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 48),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(48, 1),
        )

    def forward_with_attention(
        self,
        x_seq: torch.Tensor,
        ex_id: torch.Tensor,
        x_tab: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, t_len, _ = x_seq.shape
        ex_vec = self.ex_embed(ex_id)                              # (B, E)

        x_tok = self.input_proj(x_seq) + self.pos_embed[:, :t_len, :]
        x_tok = self.seq_film(x_tok, ex_vec)                       # (B, T, d_model)

        if x_tab is not None:
            tab_tok = self.tab_to_token(x_tab).unsqueeze(1)        # (B, 1, d_model)
        else:
            tab_tok = self.fallback_tab_token.expand(bsz, 1, -1)

        tokens = torch.cat([tab_tok, x_tok], dim=1)                # (B, 1+T, d_model)
        enc = self.out_norm(self.encoder(tokens))

        tab_out = enc[:, 0, :]
        frame_tokens = enc[:, 1:, :]
        ex_seq = ex_vec.unsqueeze(1).expand(bsz, t_len, -1)
        attn_scores = self.attn_net(torch.cat([frame_tokens, ex_seq], dim=-1)).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pool = torch.bmm(attn_weights.unsqueeze(1), frame_tokens).squeeze(1)

        feats = torch.cat([tab_out, attn_pool, ex_vec], dim=-1)
        logits = self.classifier(feats)
        return logits, attn_weights

    def forward(
        self,
        x_seq: torch.Tensor,
        ex_id: torch.Tensor,
        x_tab: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.forward_with_attention(x_seq, ex_id, x_tab)
        return logits


# ==============================================================================
# 5. MÔ HÌNH 4: HYBRID FORM ENSEMBLE (TỔ HỢP & NGƯỠNG TỐI ƯU CHỈ TRÊN VALIDATION)
# ==============================================================================
DEFAULT_HYBRID_WEIGHTS: dict[str, float] = {
    "rf": 0.50,
    "cnn_bilstm": 0.25,
    "transformer": 0.25,
}


class HybridFormEnsemble:
    """
    Kết hợp xác suất dự đoán từ 3 họ mô hình độc lập (Tabular 114-d, CNN-BiLSTM, Transformer).
    Hỗ trợ hai chế độ:
      - Prior-Anchored Fixed Simplex (`fit_fixed_simplex`, mặc định Bước E.3+):
        Dùng trọng số tiên nghiệm cân bằng (0.50 / 0.25 / 0.25, ngưỡng 0.50), triệt tiêu
        rủi ro overfit tập Validation nhỏ (~66 mẫu) theo Niculescu-Mizil & Caruana (ICML 2005).
      - Validation Grid Search (`fit_weights_and_threshold`): Giữ tương thích ngược nếu cần thử nghiệm.
    """

    def __init__(
        self,
        model_names: tuple[str, ...] = ("rf", "cnn_bilstm", "transformer"),
        default_weights: dict[str, float] | None = None,
        threshold: float = 0.50,
    ):
        self.model_names = tuple(model_names)
        base_w = default_weights or DEFAULT_HYBRID_WEIGHTS
        raw_w = {m: float(base_w.get(m, 1.0 / len(self.model_names))) for m in self.model_names}
        w_sum = sum(raw_w.values()) or 1.0
        self.weights: dict[str, float] = {m: round(v / w_sum, 4) for m, v in raw_w.items()}
        self.threshold: float = float(threshold)
        self.val_best_macro_f1: float = 0.0

    def fit_fixed_simplex(
        self,
        val_probs_dict: dict[str, np.ndarray],
        y_val: np.ndarray,
        fixed_weights: dict[str, float] | None = None,
        threshold: float = 0.50,
    ) -> "HybridFormEnsemble":
        """Áp dụng trọng số neo cố định (0 params dò trên Val) & ghi nhận Val Macro-F1 để đối chứng."""
        target_w = fixed_weights or DEFAULT_HYBRID_WEIGHTS
        active = {m: float(target_w[m]) for m in self.model_names if m in val_probs_dict and m in target_w}
        w_sum = sum(active.values())
        if w_sum <= 0.0:
            names = [m for m in self.model_names if m in val_probs_dict]
            active = {m: 1.0 / max(1, len(names)) for m in names}
            w_sum = sum(active.values()) or 1.0
        self.weights = {m: round(v / w_sum, 4) for m, v in active.items()}
        self.threshold = round(float(threshold), 4)
        val_blend = self.predict_proba(val_probs_dict)[:, 1]
        val_preds = (val_blend >= self.threshold).astype(np.int64)
        self.val_best_macro_f1 = round(float(f1_score(y_val, val_preds, average="macro", zero_division=0)), 6)
        return self

    def fit_weights_and_threshold(
        self,
        val_probs_dict: dict[str, np.ndarray],
        y_val: np.ndarray,
        grid_step: float = 0.05,
        min_rf_weight: float = MIN_RF_ENSEMBLE_WEIGHT,
    ) -> "HybridFormEnsemble":
        names = [m for m in self.model_names if m in val_probs_dict]
        if not names:
            raise ValueError("val_probs_dict is empty!")

        probs_p1 = np.stack([np.asarray(val_probs_dict[m])[:, 1] for m in names], axis=0)
        m_cnt = len(names)
        steps = np.arange(0.0, 1.0 + 1e-8, grid_step)

        weight_candidates: list[np.ndarray] = []
        if m_cnt == 1:
            weight_candidates = [np.array([1.0])]
        elif m_cnt == 2:
            for w0 in steps:
                weight_candidates.append(np.array([w0, max(0.0, 1.0 - w0)]))
        else:
            for w0 in steps:
                for w1 in steps:
                    w2 = 1.0 - w0 - w1
                    if w2 >= -1e-8:
                        weight_candidates.append(np.array([w0, w1, max(0.0, w2)]))

        rf_idx = names.index("rf") if ("rf" in names and m_cnt > 1 and min_rf_weight > 0.0) else None
        if rf_idx is not None:
            constrained = [w for w in weight_candidates if w[rf_idx] >= min_rf_weight - 1e-8]
            if constrained:
                weight_candidates = constrained

        thresholds = np.linspace(0.35, 0.65, 13)
        has_two_classes = len(np.unique(y_val)) >= 2
        best_score = (-1e9, -1e9, -1e9)
        best_w = np.full(m_cnt, 1.0 / m_cnt)
        best_thr = 0.50
        best_mf1 = 0.0

        for w in weight_candidates:
            blend_p1 = np.dot(w, probs_p1)
            v_auc = float(roc_auc_score(y_val, blend_p1)) if has_two_classes else 0.5
            prior_pen = 0.02 * float((w[rf_idx] - 0.5) ** 2) if rf_idx is not None else 0.0
            for thr in thresholds:
                preds = (blend_p1 >= thr).astype(np.int64)
                mf1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
                bacc = float(balanced_accuracy_score(y_val, preds))
                comb = 0.6 * mf1 + 0.4 * v_auc - prior_pen
                score_key = (round(comb, 6), round(bacc, 6), -abs(float(thr) - 0.50))
                if score_key > best_score:
                    best_score = score_key
                    best_w = w
                    best_thr = float(thr)
                    best_mf1 = mf1

        self.weights = {m: round(float(w_val), 4) for m, w_val in zip(names, best_w)}
        self.threshold = round(best_thr, 4)
        self.val_best_macro_f1 = round(float(best_mf1), 6)
        return self

    def fit_on_val(
        self,
        val_probs_dict: dict[str, np.ndarray],
        y_val: np.ndarray,
        grid_step: float = 0.05,
        min_rf_weight: float = MIN_RF_ENSEMBLE_WEIGHT,
    ) -> "HybridFormEnsemble":
        return self.fit_weights_and_threshold(
            val_probs_dict, y_val, grid_step=grid_step, min_rf_weight=min_rf_weight
        )

    def predict_proba(self, probs_dict: dict[str, np.ndarray]) -> np.ndarray:
        first = next(iter(probs_dict.values()))
        p1 = np.zeros(len(first), dtype=np.float64)
        w_sum = 0.0
        for name, w in self.weights.items():
            if name in probs_dict and w > 0.0:
                p1 += w * np.asarray(probs_dict[name], dtype=np.float64)[:, 1]
                w_sum += w
        if w_sum > 0.0:
            p1 /= w_sum
        p1 = np.clip(p1, 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=-1).astype(np.float32)

    def predict(self, probs_dict: dict[str, np.ndarray]) -> np.ndarray:
        p1 = self.predict_proba(probs_dict)[:, 1]
        return (p1 >= self.threshold).astype(np.int64)


# ==============================================================================
# 6. FACTORY & KIỂM ĐỊNH TOÀN DIỆN BƯỚC D PHẦN MÔ HÌNH (--verify)
# ==============================================================================
def build_model(
    model_name: str,
    seq_dim: int = FEATURE_DIM + 3,
    tab_dim: int = TABULAR_FEATURE_DIM,
    **kwargs,
):
    """Khởi tạo nhanh mô hình theo tên chuẩn ('rf', 'cnn_bilstm', 'transformer')."""
    key = model_name.strip().lower()
    if key in ("rf", "random_forest", "baseline"):
        return BiomechanicalRandomForest(**kwargs)
    if key in ("cnn_bilstm", "cnn", "bilstm"):
        return ExerciseConditionedCNNBiLSTM(seq_dim=seq_dim, tab_dim=tab_dim, **kwargs)
    if key in ("transformer", "temporal_transformer"):
        return ExerciseConditionedTransformer(seq_dim=seq_dim, tab_dim=tab_dim, **kwargs)
    raise ValueError(f"Unknown model_name: {model_name}")


def verify_models() -> None:
    """Kiểm định tham số, chiều tensor, gradient, tốc độ 1 epoch CPU và tổ hợp trên Fold 0."""
    from src.dataset import (
        ForgAITask2Dataset,
        build_stratified_group_folds,
        load_pose_cache,
        load_task2_metadata,
        prepare_fold_data,
    )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 78)
    print("FORGAI STEP D - KIỂM ĐỊNH KIẾN TRÚC 4 MÔ HÌNH (src/models.py)")
    print("=" * 78)

    df = load_task2_metadata()
    pose_cache = load_pose_cache(df)
    folds = build_stratified_group_folds(df)
    b0 = prepare_fold_data(df, folds[HOLDOUT_TEST_FOLD], pose_cache=pose_cache)
    tr, va, te = b0["train"], b0["val"], b0["test"]
    print(f"[1/4] Nạp Fold {HOLDOUT_TEST_FOLD}: Train(x6)={len(tr['y'])}, Val(x1)={len(va['y'])}, Test(x1)={len(te['y'])}")

    # 1. Kiểm định Model 1: BiomechanicalRandomForest (114-d E.3+)
    from src.rules import BiomechanicalRuleEngine

    df_lookup = df.set_index("segment_id")
    raw0_mask = np.asarray([str(sid).endswith("_f0_o0") for sid in tr["segment_id"]], dtype=bool)
    base_sids = [str(sid)[:-6] for sid, keep in zip(tr["segment_id"], raw0_mask) if keep]
    df_tr_base = df_lookup.loc[base_sids].reset_index()
    rule_eng = BiomechanicalRuleEngine(rule_weight=0.55, proto_weight=0.45).fit(
        df_tr_base, tr["tab_raw"][raw0_mask]
    )

    t0 = time.perf_counter()
    rf_model = BiomechanicalRandomForest()
    rf_model.fit(
        tr["tab_scaled"],
        tr["y"],
        tab_raw=tr["tab_raw"],
        ex_ids=tr["ex_id"],
        exercises=tr["exercise"],
        rule_engine=rule_eng,
        raw_mask=raw0_mask,
        tab_scaler=b0["tab_scaler"],
    )
    rf_fit_sec = time.perf_counter() - t0
    rf_val_p = rf_model.predict_proba(va["tab_scaled"], tab_raw=va["tab_raw"], ex_ids=va["ex_id"], exercises=va["exercise"])
    rf_val_u = rf_model.predict_unshifted_proba(va["tab_scaled"], tab_raw=va["tab_raw"], ex_ids=va["ex_id"], exercises=va["exercise"])
    rf_test_p = rf_model.predict_proba(te["tab_scaled"], tab_raw=te["tab_raw"], ex_ids=te["ex_id"], exercises=te["exercise"])
    rf_test_u = rf_model.predict_unshifted_proba(te["tab_scaled"], tab_raw=te["tab_raw"], ex_ids=te["ex_id"], exercises=te["exercise"])
    rf_val_f1 = f1_score(va["y"], (rf_val_p[:, 1] >= 0.5).astype(int), average="macro")
    rf_test_f1 = f1_score(te["y"], (rf_test_p[:, 1] >= 0.5).astype(int), average="macro")
    rf_test_acc = accuracy_score(te["y"], (rf_test_p[:, 1] >= 0.5).astype(int))
    print(
        f"[2/4] Model 1 BiomechanicalRandomForest (114-d): Fit={rf_fit_sec:.2f}s | "
        f"Val Macro-F1={rf_val_f1:.3f} | Test Acc={rf_test_acc:.3f}, Macro-F1={rf_test_f1:.3f}"
    )
    top_feats = rf_model.get_top_features(top_k=5)
    print("      Top 5 đặc trưng quan trọng:", ", ".join(f"{n}({v:.3f})" for n, v in top_feats))

    # 2. Kiểm định Model 2 & Model 3 PyTorch (Forward + Gradient + 1-epoch CPU speed)
    tr_loader = DataLoader(ForgAITask2Dataset(tr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(ForgAITask2Dataset(va), batch_size=BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(ForgAITask2Dataset(te), batch_size=BATCH_SIZE, shuffle=False)
    pos_w = compute_pos_weight(tr["y"])
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_w, gamma=1.5, label_smoothing=0.03)

    val_probs = {"rf": rf_val_u}
    test_probs = {"rf": rf_test_u}
    print(f"[3/4] Kiểm định PyTorch Deep Learning (pos_weight={pos_w:.2f}, batch_size={BATCH_SIZE}):")

    for name in ("cnn_bilstm", "transformer"):
        model = build_model(name, seq_dim=b0["seq_dim"], tab_dim=b0["tab_dim"])
        n_params = count_parameters(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        model.train()
        t_ep0 = time.perf_counter()
        ep_losses = []
        for x_s, x_t, ex_i, y_b, _ in tr_loader:
            opt.zero_grad()
            logits, attn = model.forward_with_attention(x_s, ex_i, x_t)
            assert logits.shape == (len(y_b), 1), f"Sai shape logits {name}: {logits.shape}"
            assert attn.shape == (len(y_b), NUM_FRAMES), f"Sai shape attention {name}: {attn.shape}"
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for p_name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"NaN/Inf gradient ở {name}.{p_name}"
            opt.step()
            ep_losses.append(float(loss.item()))
        ep_sec = time.perf_counter() - t_ep0

        vp, _ = predict_torch_probs(model, va_loader)
        tp, _ = predict_torch_probs(model, te_loader)
        assert vp.shape == (len(va["y"]), 2) and np.isfinite(vp).all()
        val_probs[name] = vp
        test_probs[name] = tp
        print(
            f"      - {model.__class__.__name__:32s} | Params={n_params:7,d} | "
            f"1-Epoch CPU={ep_sec:.3f}s | Loss={np.mean(ep_losses):.4f} -> PASS"
        )

    # 3. Kiểm định HybridFormEnsemble (Prior-Anchored Fixed Simplex 0.50 / 0.25 / 0.25)
    ens = HybridFormEnsemble()
    ens.fit_fixed_simplex(val_probs, va["y"])
    te_pred = ens.predict(test_probs)
    te_proba = ens.predict_proba(test_probs)
    assert te_pred.shape == (len(te["y"]),) and te_proba.shape == (len(te["y"]), 2)
    print(
        f"[4/4] HybridFormEnsemble (Fixed Simplex): weights={ens.weights}, "
        f"tau*={ens.threshold}, Val Macro-F1={ens.val_best_macro_f1:.3f} -> PASS"
    )


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser(description="ForgAI Step D Model Architectures Verification")
    parser.add_argument("--verify", action="store_true", help="Chạy kiểm định 4 mô hình trên Fold 0")
    args = parser.parse_args()
    if args.verify:
        verify_models()
    else:
        print("Chạy `python src/models.py --verify` để kiểm định 4 mô hình Bước D.")
