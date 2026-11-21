# -*- coding: utf-8 -*-
"""
experiments/train_task2.py
Huấn luyện & Lưu trữ mô hình Task 2 theo 5-Fold Stratified Group Cross-Validation (Zero Data Leakage).

Huấn luyện đầy đủ 4 họ mô hình phân loại trong src/models.py + bộ giải thích cơ sinh học src/rules.py:
  1. BiomechanicalRandomForest (RF + ExtraTrees trên 101 đặc trưng bảng).
  2. ExerciseConditionedCNNBiLSTM (Multi-Scale 1D-CNN + FiLM + Bi-LSTM + Attention + Gated Tabular).
  3. ExerciseConditionedTransformer (Pre-LN Transformer + FiLM + [TAB] Summary Token + Attention).
  4. HybridFormEnsemble (Hòa trộn xác suất & dò ngưỡng tối ưu Macro-F1 CHỈ trên Validation).
  5. BiomechanicalRuleEngine (Lai 2 tầng Luật vật lý + Chữ ký lỗi, fit CHỈ trên Train gốc _f0_o0).

Cách dùng:
  python experiments/train_task2.py                        # Chạy đủ 5 Fold (epochs=80, patience=18)
  python experiments/train_task2.py --folds 0 --epochs 3   # Smoke test nhanh trên Fold 0
  python experiments/train_task2.py --skip-existing        # Bỏ qua các Fold đã có checkpoint hoàn chỉnh
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MIN_RF_ENSEMBLE_WEIGHT,
    N_SPLITS,
    RANDOM_SEED,
    REPORTS_DIR,
    TRAIN_NOISE_STD,
    WEIGHT_DECAY,
)
from src.dataset import (
    ForgAITask2Dataset,
    build_stratified_group_folds,
    load_pose_2d_cache,
    load_pose_cache,
    load_task2_metadata,
    prepare_fold_data,
)
from src.features import TABULAR_FEATURE_NAMES
from src.models import (
    DEFAULT_HYBRID_WEIGHTS,
    FocalBCEWithLogitsLoss,
    HybridFormEnsemble,
    build_model,
    compute_pos_weight,
    count_parameters,
)
from src.rules import BiomechanicalRuleEngine


def set_global_seed(seed: int) -> None:
    """Cố định seed tái lập cho Python, NumPy và PyTorch CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _safe_roc_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, p_pos))


def _compute_split_metrics(y_true: np.ndarray, p_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (p_pos >= threshold).astype(np.int64)
    return {
        "acc": float(accuracy_score(y_true, preds)),
        "bacc": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_incorrect": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, p_pos),
    }


@torch.no_grad()
def evaluate_torch_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Đánh giá mất mát, xác suất (N, 2), nhãn thật (N,) và trọng số Attention (N, 32) trong 1 lượt."""
    model.eval()
    losses, probs_list, y_list, attn_list = [], [], [], []
    for x_seq, x_tab, ex_id, y_b, _ in loader:
        x_seq = x_seq.to(device)
        x_tab = x_tab.to(device)
        ex_id = ex_id.to(device)
        y_dev = y_b.to(device)

        logits, attn = model.forward_with_attention(x_seq, ex_id, x_tab)
        loss = criterion(logits, y_dev)
        p1 = torch.sigmoid(logits.view(-1)).cpu().numpy()
        probs = np.stack([1.0 - p1, p1], axis=-1).astype(np.float32)

        losses.append(float(loss.item()) * len(y_b))
        probs_list.append(probs)
        y_list.append(y_b.numpy())
        attn_list.append(attn.cpu().numpy().astype(np.float32))

    total_n = max(1, sum(len(y) for y in y_list))
    mean_loss = float(sum(losses) / total_n)
    return (
        mean_loss,
        np.concatenate(probs_list, axis=0),
        np.concatenate(y_list, axis=0),
        np.concatenate(attn_list, axis=0),
    )


def train_torch_model_on_fold(
    model_name: str,
    fold_idx: int,
    seq_dim: int,
    tab_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train: np.ndarray,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    label_smoothing: float,
    train_noise_std: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """
    Huấn luyện 1 mô hình PyTorch trên Fold hiện tại:
      - Chọn checkpoint tốt nhất theo điểm tổng hợp cân bằng (0.45*Val MF1 + 0.45*Val AUC - 0.10*min(Val Loss, 2.0)).
      - Chống overfit: nhiễu Gaussian nhẹ chỉ lúc Train, Dropout, Label Smoothing, Weight Decay & CosineAnnealingLR.
      - Nạp lại state_dict tốt nhất trước khi trả về.
    """
    set_global_seed(seed)
    model = build_model(model_name, seq_dim=seq_dim, tab_dim=tab_dim, dropout=dropout).to(device)
    n_params = count_parameters(model)

    pos_w = compute_pos_weight(y_train)
    criterion = FocalBCEWithLogitsLoss(pos_weight=pos_w, gamma=1.5, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-5)

    best_state = copy.deepcopy(model.state_dict())
    best_score = (-1e9, -1e9, -1e9)
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    epochs_no_improve = 0

    history: dict[str, list[float]] = {
        "epoch": [],
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_bacc": [],
        "val_macro_f1": [],
        "val_roc_auc": [],
        "val_comp": [],
    }

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for x_seq, x_tab, ex_id, y_b, _ in train_loader:
            x_seq = x_seq.to(device)
            x_tab = x_tab.to(device)
            ex_id = ex_id.to(device)
            y_dev = y_b.to(device)

            if train_noise_std > 0.0:
                x_seq = x_seq + train_noise_std * torch.randn_like(x_seq)
                x_tab = x_tab + train_noise_std * torch.randn_like(x_tab)

            optimizer.zero_grad()
            logits = model(x_seq, ex_id, x_tab)
            loss = criterion(logits, y_dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(y_b)
            n_seen += len(y_b)

        scheduler.step()
        train_loss = running_loss / max(1, n_seen)
        val_loss, val_probs, val_y, _ = evaluate_torch_loader(model, val_loader, criterion, device)
        val_m = _compute_split_metrics(val_y, val_probs[:, 1], threshold=0.5)
        val_comp = 0.45 * val_m["macro_f1"] + 0.45 * val_m["roc_auc"] - 0.10 * min(val_loss, 2.0)

        history["epoch"].append(epoch)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_m["acc"])
        history["val_bacc"].append(val_m["bacc"])
        history["val_macro_f1"].append(val_m["macro_f1"])
        history["val_roc_auc"].append(val_m["roc_auc"])
        history["val_comp"].append(float(val_comp))

        score_key = (round(val_comp, 6), round(val_m["roc_auc"], 6), -round(val_loss, 6))
        marker = ""
        if score_key > best_score:
            best_score = score_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = dict(val_m)
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or marker or epochs_no_improve >= patience or epoch == epochs:
            print(
                f"      [Fold {fold_idx} | {model_name:11s}] Ep {epoch:02d}/{epochs:02d} | "
                f"TrLoss={train_loss:.4f} VaLoss={val_loss:.4f} | "
                f"VaAcc={val_m['acc']:.3f} VaMF1={val_m['macro_f1']:.3f} VaAUC={val_m['roc_auc']:.3f} Comp={val_comp:.3f}{marker}"
            )

        if epochs_no_improve >= patience:
            print(f"      -> Early Stopping {model_name} ở epoch {epoch} (tốt nhất: epoch {best_epoch}).")
            break

    elapsed_sec = time.perf_counter() - t_start
    model.load_state_dict(best_state)

    summary = {
        "model_name": model_name,
        "class_name": model.__class__.__name__,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["epoch"]),
        "elapsed_sec": round(elapsed_sec, 2),
        "best_val_metrics": best_val_metrics,
        "history": history,
    }
    return model, summary


def _extract_raw_train_base(df_lookup: pd.DataFrame, train_split: dict, expected_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Lọc đúng bản gốc không augment (_f0_o0) và căn khớp 1-1 với metadata cho Rule Engine."""
    raw0_mask = np.asarray([str(sid).endswith("_f0_o0") for sid in train_split["segment_id"]], dtype=bool)
    base_seg_ids = [str(sid)[:-6] for sid, keep in zip(train_split["segment_id"], raw0_mask) if keep]
    df_train_base = df_lookup.loc[base_seg_ids].reset_index()
    if len(df_train_base) != expected_count:
        raise AssertionError(
            f"Lệch số lượng mẫu Train gốc khi lọc _f0_o0: {len(df_train_base)} vs {expected_count}"
        )
    return df_train_base, raw0_mask


def _load_reused_dl_fold(
    model_name: str,
    fold_idx: int,
    fold_ckpt_dir: Path,
    seq_dim: int,
    tab_dim: int,
    val_loader: DataLoader,
    te_loader: DataLoader,
    eval_criterion: nn.Module,
    device: torch.device,
    prev_fold_history: dict | None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Nạp lại checkpoint DL Bước E.2 (.pt) & giữ nguyên lịch sử epoch để tránh phải train lại."""
    pt_path = fold_ckpt_dir / f"{model_name}_best.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint DL để dùng lại: {pt_path}. Hãy bỏ --reuse-dl để huấn luyện mới."
        )
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    if int(ckpt["seq_dim"]) != int(seq_dim) or int(ckpt["tab_dim"]) != int(tab_dim):
        raise ValueError(
            f"Lệch chiều tensor ở {pt_path}: ckpt=({ckpt['seq_dim']}, {ckpt['tab_dim']}) vs fold=({seq_dim}, {tab_dim})"
        )
    dropout = float(ckpt.get("hparams", {}).get("dropout", DROPOUT_RATE))
    model = build_model(model_name, seq_dim=seq_dim, tab_dim=tab_dim, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, vp, vy, _ = evaluate_torch_loader(model, val_loader, eval_criterion, device)
    _, tp, ty, t_attn = evaluate_torch_loader(model, te_loader, eval_criterion, device)
    te_m = _compute_split_metrics(ty, tp[:, 1], threshold=0.5)
    va_m = ckpt.get("best_val_metrics") or _compute_split_metrics(vy, vp[:, 1], threshold=0.5)

    prev_summary = (prev_fold_history or {}).get(model_name, {})
    summary = {
        "model_name": model_name,
        "class_name": model.__class__.__name__,
        "n_params": int(prev_summary.get("n_params", count_parameters(model))),
        "best_epoch": int(ckpt.get("best_epoch", prev_summary.get("best_epoch", 0))),
        "epochs_ran": int(prev_summary.get("epochs_ran", len(prev_summary.get("history", {}).get("epoch", [])))),
        "elapsed_sec": float(prev_summary.get("elapsed_sec", 0.0)),
        "best_val_metrics": va_m,
        "test_metrics": te_m,
        "history": prev_summary.get("history", {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}),
    }
    return summary, vp, tp, t_attn


def run_single_fold(
    fold_spec: dict,
    df: pd.DataFrame,
    df_lookup: pd.DataFrame,
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]],
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    label_smoothing: float,
    train_noise_std: float,
    min_rf_weight: float,
    seed: int,
    device: torch.device,
    reuse_dl: bool = False,
    pose_2d_cache: dict[str, np.ndarray | None] | None = None,
) -> dict:
    """Thực thi quy trình huấn luyện E.3-LowRisk, giải thích lỗi và lưu checkpoint cho 1 Fold."""
    fold_idx = int(fold_spec["fold"])
    fold_ckpt_dir = CHECKPOINTS_DIR / f"fold_{fold_idx}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    prev_fold_history = None
    prev_oof_pkl = fold_ckpt_dir / "fold_oof_results.pkl"
    if reuse_dl and prev_oof_pkl.exists():
        with open(prev_oof_pkl, "rb") as f_prev:
            prev_fold_history = pickle.load(f_prev).get("history")

    t_fold0 = time.perf_counter()
    print("=" * 84)
    print(f"FOLD {fold_idx} / {N_SPLITS - 1} - CHUẨN BỊ DỮ LIỆU ZERO DATA LEAKAGE (E.3-LowRisk)")
    print("=" * 84)

    fold_data = prepare_fold_data(
        df,
        fold_spec,
        pose_cache=pose_cache,
        apply_train_aug=True,
        include_golden_channels=True,
        pose_2d_cache=pose_2d_cache,
    )
    tr, va, te = fold_data["train"], fold_data["val"], fold_data["test"]
    seq_dim = fold_data["seq_dim"]
    tab_dim = fold_data["tab_dim"]
    print(
        f"  [Data] Train gốc={fold_data['raw_counts']['train_clips']} -> Aug x6={len(tr['y'])} | "
        f"Val={len(va['y'])} | Test={len(te['y'])} | seq_dim={seq_dim}, tab_dim={tab_dim}"
    )

    # 1. BiomechanicalRuleEngine (fit STRICTLY trên Train gốc _f0_o0 trước khi fit RF 103-d)
    df_train_base, raw0_mask = _extract_raw_train_base(
        df_lookup, tr, expected_count=fold_data["raw_counts"]["train_clips"]
    )
    rule_engine = BiomechanicalRuleEngine(rule_weight=0.55, proto_weight=0.45)
    rule_engine.fit(df_train_base, tr["tab_raw"][raw0_mask])

    # 2. BiomechanicalRandomForest E.3-LowRisk (Per-Exercise Correct-Ref 103-d + HGB trên _f0_o0 + Train Prior Shift)
    t_rf0 = time.perf_counter()
    rf_model = build_model("rf", random_state=seed + fold_idx)
    rf_model.fit(
        tr["tab_scaled"],
        tr["y"],
        tab_raw=tr["tab_raw"],
        ex_ids=tr["ex_id"],
        exercises=tr["exercise"],
        rule_engine=rule_engine,
        raw_mask=raw0_mask,
        tab_scaler=fold_data["tab_scaler"],
    )
    rf_sec = time.perf_counter() - t_rf0
    rf_val_p = rf_model.predict_proba(va["tab_scaled"], tab_raw=va["tab_raw"], ex_ids=va["ex_id"], exercises=va["exercise"])
    rf_val_u = rf_model.predict_unshifted_proba(va["tab_scaled"], tab_raw=va["tab_raw"], ex_ids=va["ex_id"], exercises=va["exercise"])
    rf_test_p = rf_model.predict_proba(te["tab_scaled"], tab_raw=te["tab_raw"], ex_ids=te["ex_id"], exercises=te["exercise"])
    rf_test_u = rf_model.predict_unshifted_proba(te["tab_scaled"], tab_raw=te["tab_raw"], ex_ids=te["ex_id"], exercises=te["exercise"])
    rf_imp = rf_model.get_feature_importances()
    rf_val_m = _compute_split_metrics(va["y"], rf_val_p[:, 1], threshold=0.5)
    rf_test_m = _compute_split_metrics(te["y"], rf_test_p[:, 1], threshold=0.5)
    print(
        f"  [Model 1] BiomechanicalRF 103-d ({rf_sec:.2f}s) | "
        f"Val MF1={rf_val_m['macro_f1']:.3f} | Test Acc={rf_test_m['acc']:.3f}, "
        f"MF1={rf_test_m['macro_f1']:.3f}, AUC={rf_test_m['roc_auc']:.3f}"
    )

    # 3. DataLoaders PyTorch
    dl_gen = torch.Generator().manual_seed(seed + fold_idx)
    tr_loader = DataLoader(ForgAITask2Dataset(tr), batch_size=batch_size, shuffle=True, generator=dl_gen)
    va_loader = DataLoader(ForgAITask2Dataset(va), batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(ForgAITask2Dataset(te), batch_size=batch_size, shuffle=False)
    pos_w = compute_pos_weight(tr["y"])
    eval_criterion = FocalBCEWithLogitsLoss(pos_weight=pos_w, gamma=1.5, label_smoothing=label_smoothing)

    # 4. Nạp lại (--reuse-dl) hoặc Huấn luyện mới Deep Learning Models
    val_probs = {"rf": rf_val_p}
    test_probs = {"rf": rf_test_p}
    dl_summaries = {}
    test_attns = {}

    for m_idx, model_name in enumerate(("cnn_bilstm", "transformer")):
        if reuse_dl:
            summary, vp, tp, t_attn = _load_reused_dl_fold(
                model_name=model_name,
                fold_idx=fold_idx,
                fold_ckpt_dir=fold_ckpt_dir,
                seq_dim=seq_dim,
                tab_dim=tab_dim,
                val_loader=va_loader,
                te_loader=te_loader,
                eval_criterion=eval_criterion,
                device=device,
                prev_fold_history=prev_fold_history,
            )
            te_m = summary["test_metrics"]
            print(
                f"  [Model {m_idx + 2}] Tái sử dụng {model_name}_best.pt (Best Ep {summary['best_epoch']}) | "
                f"Test Acc={te_m['acc']:.3f}, MF1={te_m['macro_f1']:.3f}, AUC={te_m['roc_auc']:.3f}"
            )
        else:
            print(f"  [Model {m_idx + 2}] Huấn luyện {model_name} (pos_weight={pos_w:.2f}, lr={lr}, wd={weight_decay}, drop={dropout})...")
            best_model, summary = train_torch_model_on_fold(
                model_name=model_name,
                fold_idx=fold_idx,
                seq_dim=seq_dim,
                tab_dim=tab_dim,
                train_loader=tr_loader,
                val_loader=va_loader,
                y_train=tr["y"],
                epochs=epochs,
                patience=patience,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                label_smoothing=label_smoothing,
                train_noise_std=train_noise_std,
                seed=seed + fold_idx * 100 + m_idx,
                device=device,
            )
            _, vp, _, _ = evaluate_torch_loader(best_model, va_loader, eval_criterion, device)
            _, tp, _, t_attn = evaluate_torch_loader(best_model, te_loader, eval_criterion, device)
            te_m = _compute_split_metrics(te["y"], tp[:, 1], threshold=0.5)
            summary["test_metrics"] = te_m
            print(
                f"      => Best Ep {summary['best_epoch']} ({summary['elapsed_sec']:.1f}s) | "
                f"Test Acc={te_m['acc']:.3f}, MF1={te_m['macro_f1']:.3f}, AUC={te_m['roc_auc']:.3f}"
            )

            fold_ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": model_name,
                    "fold": fold_idx,
                    "seq_dim": seq_dim,
                    "tab_dim": tab_dim,
                    "best_epoch": summary["best_epoch"],
                    "best_val_metrics": summary["best_val_metrics"],
                    "test_metrics": te_m,
                    "hparams": {
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "dropout": dropout,
                        "label_smoothing": label_smoothing,
                        "train_noise_std": train_noise_std,
                    },
                    "model_state_dict": best_model.state_dict(),
                },
                fold_ckpt_dir / f"{model_name}_best.pt",
            )

        val_probs[model_name] = vp
        test_probs[model_name] = tp
        test_attns[model_name] = t_attn
        dl_summaries[model_name] = summary

    # 5. HybridFormEnsemble (Prior-Anchored Fixed Simplex 0.60 / 0.25 / 0.15, thr=0.50 -> 0 tham số dò trên Val)
    hybrid = HybridFormEnsemble(model_names=("rf", "cnn_bilstm", "transformer"))
    hybrid.fit_fixed_simplex(val_probs, va["y"], fixed_weights=DEFAULT_HYBRID_WEIGHTS, threshold=0.50)
    hyb_test_p = hybrid.predict_proba(test_probs)
    hyb_test_pred = hybrid.predict(test_probs)
    hyb_test_m = _compute_split_metrics(te["y"], hyb_test_p[:, 1], threshold=hybrid.threshold)
    print(
        f"  [Model 4] HybridFormEnsemble (Fixed Simplex): w={hybrid.weights}, tau={hybrid.threshold} | "
        f"Val MF1={hybrid.val_best_macro_f1:.3f} -> Test Acc={hyb_test_m['acc']:.3f}, "
        f"MF1={hyb_test_m['macro_f1']:.3f}, AUC={hyb_test_m['roc_auc']:.3f}"
    )

    diag_outputs = rule_engine.explain_batch(
        te["tab_raw"], te["exercise"], probs_incorrect=hyb_test_p[:, 1], top_k=3
    )
    e2e_outputs = [
        rule_engine.explain_rep(
            te["tab_raw"][i],
            str(te["exercise"][i]),
            p_incorrect=float(hyb_test_p[i, 1]),
            is_incorrect=bool(hyb_test_pred[i] == 1),
            top_k=3,
        )
        for i in range(len(te["y"]))
    ]

    # Đối chiếu metadata thật (source, error_reason) cho tập Test
    te_seg_ids = [str(s) for s in te["segment_id"]]
    df_test_meta = df_lookup.loc[te_seg_ids]
    y_err_true = df_test_meta["error_reason"].astype(str).tolist()
    te_sources = df_test_meta["source"].astype(str).tolist()

    inc_indices = [i for i, y_v in enumerate(te["y"]) if int(y_v) == 1]
    if inc_indices:
        top1_rec = sum(diag_outputs[i]["predicted_error_code"] == y_err_true[i] for i in inc_indices) / len(inc_indices)
        top3_rec = sum(y_err_true[i] in diag_outputs[i]["top_k_errors"] for i in inc_indices) / len(inc_indices)
        print(
            f"  [Explainer] Rule Engine trên {len(inc_indices)} mẫu Test Sai: "
            f"Top-1 Recall={top1_rec * 100:.1f}%, Top-3 Recall={top3_rec * 100:.1f}%"
        )

    # Tạo bảng dự đoán Out-of-Fold cho Fold hiện tại
    oof_rows = []
    for i, sid in enumerate(te_seg_ids):
        top_k_codes = diag_outputs[i]["top_k_errors"]
        top3_cands = diag_outputs[i]["top3_candidates"]
        oof_rows.append(
            {
                "segment_id": sid,
                "video_id": str(te["video_id"][i]),
                "exercise": str(te["exercise"][i]),
                "source": te_sources[i],
                "fold": fold_idx,
                "duration_sec": float(te["duration_sec"][i]),
                "y_true": int(te["y"][i]),
                "error_reason": y_err_true[i],
                "prob_rf": float(rf_test_p[i, 1]),
                "prob_rf_unshifted": float(rf_test_u[i, 1]),
                "prob_cnn_bilstm": float(test_probs["cnn_bilstm"][i, 1]),
                "prob_transformer": float(test_probs["transformer"][i, 1]),
                "prob_hybrid": float(hyb_test_p[i, 1]),
                "hybrid_threshold": float(hybrid.threshold),
                "pred_rf": int(rf_test_p[i, 1] >= 0.5),
                "pred_cnn_bilstm": int(test_probs["cnn_bilstm"][i, 1] >= 0.5),
                "pred_transformer": int(test_probs["transformer"][i, 1] >= 0.5),
                "pred_hybrid": int(hyb_test_pred[i]),
                "rule_top1_code": top_k_codes[0] if len(top_k_codes) > 0 else "none",
                "rule_top2_code": top_k_codes[1] if len(top_k_codes) > 1 else "none",
                "rule_top3_code": top_k_codes[2] if len(top_k_codes) > 2 else "none",
                "rule_top1_score": float(top3_cands[0]["score"]) if top3_cands else 0.0,
                "rule_measured_evidence": diag_outputs[i]["measured_evidence"],
                "rule_coaching_advice": diag_outputs[i]["coaching_advice"],
                "e2e_error_code": e2e_outputs[i]["predicted_error_code"],
                "feedback_vi": e2e_outputs[i]["feedback_vi"],
            }
        )
    oof_df = pd.DataFrame(oof_rows)

    fold_elapsed = round(time.perf_counter() - t_fold0, 2)
    fold_history = {
        "fold": fold_idx,
        "raw_counts": fold_data["raw_counts"],
        "fold_elapsed_sec": fold_elapsed,
        "rf": {"fit_sec": round(rf_sec, 2), "val_metrics": rf_val_m, "test_metrics": rf_test_m},
        "cnn_bilstm": dl_summaries["cnn_bilstm"],
        "transformer": dl_summaries["transformer"],
        "hybrid": {
            "weights": hybrid.weights,
            "threshold": hybrid.threshold,
            "val_best_macro_f1": hybrid.val_best_macro_f1,
            "test_metrics": hyb_test_m,
        },
    }

    # Lưu bundle phục vụ Bước F (Webcam Demo) và khôi phục Fold
    bundle = {
        "fold": fold_idx,
        "seq_dim": seq_dim,
        "tab_dim": tab_dim,
        "rf_model": rf_model,
        "hybrid_ensemble": hybrid,
        "rule_engine": rule_engine,
        "golden_bank": fold_data["golden_bank"],
        "seq_scaler": fold_data["seq_scaler"],
        "tab_scaler": fold_data["tab_scaler"],
        "test_video_ids": sorted(set(oof_df["video_id"].tolist())),
    }
    with open(fold_ckpt_dir / "tabular_and_ensemble.pkl", "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    fold_result = {
        "fold": fold_idx,
        "oof_df": oof_df,
        "history": fold_history,
        "rf_importances": rf_imp,
        "rf_feature_names": list(rf_model.feature_names),
        "test_segment_id": np.asarray(te_seg_ids),
        "test_exercise": np.asarray(te["exercise"]),
        "test_y": np.asarray(te["y"], dtype=np.int64),
        "test_error_reason": np.asarray(y_err_true),
        "test_cnn_attn": test_attns["cnn_bilstm"],
        "test_trans_attn": test_attns["transformer"],
        "test_anat_10": te["anat_10"].astype(np.float32),
    }
    with open(fold_ckpt_dir / "fold_oof_results.pkl", "wb") as f:
        pickle.dump(fold_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  -> Đã lưu checkpoint & OOF Fold {fold_idx} tại {fold_ckpt_dir} ({fold_elapsed:.1f}s).")
    return fold_result


def _fold_checkpoint_complete(fold_idx: int) -> bool:
    fdir = CHECKPOINTS_DIR / f"fold_{fold_idx}"
    required = (
        fdir / "cnn_bilstm_best.pt",
        fdir / "transformer_best.pt",
        fdir / "tabular_and_ensemble.pkl",
        fdir / "fold_oof_results.pkl",
    )
    return all(p.exists() for p in required)


def aggregate_and_save_global_reports(expected_total_clips: int) -> bool:
    """Gộp kết quả đủ 5 Fold (0..4) thành bộ báo cáo OOF toàn cục tại outputs/reports/."""
    fold_results = []
    for k in range(N_SPLITS):
        pkl_path = CHECKPOINTS_DIR / f"fold_{k}" / "fold_oof_results.pkl"
        if not pkl_path.exists():
            return False
        with open(pkl_path, "rb") as f:
            fold_results.append(pickle.load(f))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.concat([r["oof_df"] for r in fold_results], ignore_index=True)

    # Xác minh không rò rỉ dữ liệu & đủ toàn bộ 493 clip
    assert len(oof_df) == expected_total_clips, f"OOF thiếu mẫu: {len(oof_df)} vs {expected_total_clips}"
    assert oof_df["segment_id"].nunique() == expected_total_clips, "Phát hiện trùng segment_id trong OOF!"
    assert (oof_df.groupby("video_id")["fold"].nunique() == 1).all(), "Rò rỉ video_id giữa các Test Fold!"

    oof_csv_path = REPORTS_DIR / "cv_oof_test_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False, encoding="utf-8-sig")

    histories = {"folds": [r["history"] for r in fold_results]}
    hist_path = REPORTS_DIR / "cv_training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    imp_rows = []
    for r in fold_results:
        feat_names = r.get("rf_feature_names", TABULAR_FEATURE_NAMES)
        for feat_name, imp_val in zip(feat_names, r["rf_importances"]):
            imp_rows.append({"fold": int(r["fold"]), "feature": feat_name, "importance": float(imp_val)})
    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(REPORTS_DIR / "cv_rf_feature_importances.csv", index=False, encoding="utf-8-sig")

    np.savez_compressed(
        REPORTS_DIR / "cv_attention_and_angles.npz",
        segment_id=np.concatenate([r["test_segment_id"] for r in fold_results], axis=0),
        exercise=np.concatenate([r["test_exercise"] for r in fold_results], axis=0),
        y_true=np.concatenate([r["test_y"] for r in fold_results], axis=0),
        error_reason=np.concatenate([r["test_error_reason"] for r in fold_results], axis=0),
        cnn_attn=np.concatenate([r["test_cnn_attn"] for r in fold_results], axis=0),
        trans_attn=np.concatenate([r["test_trans_attn"] for r in fold_results], axis=0),
        anat_angles_10=np.concatenate([r["test_anat_10"] for r in fold_results], axis=0),
    )

    print("\n" + "=" * 84)
    print(f"HOÀN TẤT GỘP KẾT QUẢ 5-FOLD OOF ({len(oof_df)} MẪU, {oof_df['video_id'].nunique()} NHÓM VIDEO_ID)")
    print(f"  - OOF Predictions : {oof_csv_path}")
    print(f"  - Training History: {hist_path}")
    print(f"  - Attention/Angles: {REPORTS_DIR / 'cv_attention_and_angles.npz'}")
    print("=" * 84)
    return True


def parse_folds_arg(folds_str: str) -> list[int]:
    s = folds_str.strip().lower()
    if s == "all":
        return list(range(N_SPLITS))
    folds = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    for k in folds:
        if not 0 <= k < N_SPLITS:
            raise ValueError(f"Fold phải nằm trong [0, {N_SPLITS - 1}], nhận được: {k}")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description="ForgAI Step E - 5-Fold CV Task 2 Training")
    parser.add_argument("--folds", type=str, default="all", help="Các Fold cần chạy ('all' hoặc '0,1')")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số epoch tối đa mỗi mô hình Deep Learning")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Ngưỡng Early Stopping")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Kích thước mini-batch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate AdamW ban đầu")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="Weight decay AdamW")
    parser.add_argument("--dropout", type=float, default=DROPOUT_RATE, help="Tỷ lệ Dropout cho mô hình DL")
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING, help="Label smoothing trong FocalBCE")
    parser.add_argument("--noise-std", type=float, default=TRAIN_NOISE_STD, help="Độ lệch chuẩn nhiễu Gaussian chỉ lúc Train")
    parser.add_argument("--min-rf-weight", type=float, default=MIN_RF_ENSEMBLE_WEIGHT, help="Sàn trọng số tối thiểu cho RF trong Hybrid")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Seed cố định")
    parser.add_argument("--skip-existing", action="store_true", help="Bỏ qua Fold đã đủ checkpoint")
    parser.add_argument(
        "--reuse-dl",
        action="store_true",
        help="Dùng lại checkpoint DL (cnn_bilstm_best.pt, transformer_best.pt) để cập nhật nhanh E.3-LowRisk (~30s)",
    )
    args = parser.parse_args()

    selected_folds = parse_folds_arg(args.folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 84)
    print(f"FORGAI STEP 6.2 (E.3-LowRisk) - HUẤN LUYỆN 5-FOLD STRATIFIED GROUP CV (reuse_dl={args.reuse_dl})")
    print(
        f"Device={device} | Folds={selected_folds} | Epochs={args.epochs} | Patience={args.patience} | "
        f"lr={args.lr}, wd={args.weight_decay}, drop={args.dropout}, noise={args.noise_std}"
    )
    print("=" * 84)

    df = load_task2_metadata()
    df_lookup = df.set_index("segment_id")
    pose_cache = load_pose_cache(df)
    pose_2d_cache = load_pose_2d_cache(df)
    folds = build_stratified_group_folds(df)
    print(f"[Init] Đã nạp {len(df)} mẫu ({df['video_id'].nunique()} nhóm video_id) & cache khung xương 3D+2D.")

    t_all0 = time.perf_counter()
    for fold_idx in selected_folds:
        if args.skip_existing and (not args.reuse_dl) and _fold_checkpoint_complete(fold_idx):
            print(f"[Skip] Fold {fold_idx} đã có đầy đủ checkpoint tại {CHECKPOINTS_DIR / f'fold_{fold_idx}'}.")
            continue
        run_single_fold(
            fold_spec=folds[fold_idx],
            df=df,
            df_lookup=df_lookup,
            pose_cache=pose_cache,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            label_smoothing=args.label_smoothing,
            train_noise_std=args.noise_std,
            min_rf_weight=args.min_rf_weight,
            seed=args.seed,
            device=device,
            reuse_dl=args.reuse_dl,
            pose_2d_cache=pose_2d_cache,
        )

    merged_all = aggregate_and_save_global_reports(expected_total_clips=len(df))
    if not merged_all:
        print(
            "[Note] Chưa đủ cả 5 Fold (0..4) hoàn chỉnh trên đĩa nên chưa xuất bộ báo cáo 5-Fold toàn cục."
        )
    print(f"[Done] Tổng thời gian chạy: {(time.perf_counter() - t_all0) / 60.0:.2f} phút.")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    main()
