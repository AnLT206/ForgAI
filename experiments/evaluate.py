# -*- coding: utf-8 -*-
"""
experiments/evaluate.py
Đánh giá toàn diện & Trực quan hóa kết quả 5-Fold Cross-Validation cho Task 2 (Bước E).

Đọc độc lập các tệp đầu ra của experiments/train_task2.py tại outputs/reports/:
  - cv_oof_test_predictions.csv
  - cv_training_history.json
  - cv_attention_and_angles.npz
  - cv_rf_feature_importances.csv

Chức năng chính:
  1. Hậu kiểm Zero Data Leakage (493 clip, 356 nhóm video_id, 0 giao nhau giữa các Test Fold).
  2. Tính bảng chỉ số 5-Fold (Mean ± Std) & Pooled OOF (N = 493) cho cả 4 mô hình.
  3. Phân rã hiệu năng theo 8 bài tập Nhóm A, theo 3 nguồn (EFA, GYMED, YouTube) và McNemar test.
  4. Đánh giá chẩn đoán 32 mã lỗi của BiomechanicalRuleEngine (Top-1 / Top-3 Error Recall).
  5. Xuất 5 biểu đồ báo cáo 300-DPI (eval_fig1*..eval_fig5*) + task2_eval_summary.json + task2_eval_report.md.

Cách dùng:
  python experiments/evaluate.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import FIGURES_DIR, GROUP_A_EXERCISES, N_SPLITS, REPORTS_DIR
from src.constants import ERROR_REASON_VN_DICT
from src.features import ANATOMICAL_GOLDEN_KEYS, EXERCISE_BIOMECH_SPEC
from src.rules import EXERCISE_ERROR_RULES

plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MODEL_SPECS = [
    ("rf", "BiomechanicalRF (106-d E.3+)", "prob_rf", "pred_rf", "#3B528B"),
    ("cnn_bilstm", "CNN-BiLSTM + FiLM", "prob_cnn_bilstm", "pred_cnn_bilstm", "#21918C"),
    ("transformer", "Transformer + FiLM", "prob_transformer", "pred_transformer", "#5EC962"),
    ("hybrid", "Hybrid Ensemble (0.50/0.25/0.25)", "prob_hybrid", "pred_hybrid", "#D95F02"),
]

COLOR_CORRECT = "#1B9E77"
COLOR_INCORRECT = "#D95F02"


def _safe_auc(y_true: np.ndarray, p_pos: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p_pos))


def compute_binary_metrics(y_true: np.ndarray, p_pos: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    """Tính trọn bộ chỉ số phân loại nhị phân Đúng (0) / Sai (1)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    auc_val = _safe_auc(y_true, p_pos)
    return {
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float(accuracy_score(y_true, y_pred)),
        "specificity_correct": float(spec),
        "precision_incorrect": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_incorrect": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_incorrect": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_correct": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": auc_val,
    }


def audit_zero_leakage(oof_df: pd.DataFrame) -> dict:
    """Xác minh toàn vẹn OOF N=493 không trùng segment_id và không rò rỉ video_id giữa các Test Fold."""
    n_rows = int(len(oof_df))
    n_seg = int(oof_df["segment_id"].nunique())
    n_vid = int(oof_df["video_id"].nunique())
    folds_seen = sorted(int(f) for f in oof_df["fold"].unique())
    vids_per_multiple_folds = int((oof_df.groupby("video_id")["fold"].nunique() > 1).sum())

    assert n_rows == n_seg, f"Trùng lặp segment_id trong OOF: {n_rows} dòng vs {n_seg} unique!"
    assert folds_seen == list(range(N_SPLITS)), f"Thiếu Fold trong OOF: {folds_seen}"
    assert vids_per_multiple_folds == 0, f"Phát hiện {vids_per_multiple_folds} video_id rò rỉ qua >1 Fold!"

    return {
        "total_oof_clips": n_rows,
        "unique_segments": n_seg,
        "unique_video_id_groups": n_vid,
        "folds_evaluated": folds_seen,
        "cross_fold_video_id_leaks": vids_per_multiple_folds,
        "n_correct": int((oof_df["y_true"] == 0).sum()),
        "n_incorrect": int((oof_df["y_true"] == 1).sum()),
    }


def compute_cv_and_pooled_tables(oof_df: pd.DataFrame) -> tuple[dict, dict]:
    """Tính Mean ± Std qua 5 Fold và Pooled OOF cho 4 mô hình."""
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_incorrect",
        "recall_incorrect",
        "f1_incorrect",
        "specificity_correct",
        "roc_auc",
    ]
    cv_summary = {}
    pooled_summary = {}

    y_all = oof_df["y_true"].to_numpy(dtype=np.int64)
    for key, label, prob_col, pred_col, _ in MODEL_SPECS:
        fold_rows = []
        for f_idx in range(N_SPLITS):
            sub = oof_df[oof_df["fold"] == f_idx]
            m = compute_binary_metrics(
                sub["y_true"].to_numpy(dtype=np.int64),
                sub[prob_col].to_numpy(dtype=np.float64),
                sub[pred_col].to_numpy(dtype=np.int64),
            )
            fold_rows.append(m)

        agg = {"label": label, "per_fold": fold_rows}
        for mk in metric_keys:
            vals = [float(r[mk]) for r in fold_rows if r[mk] is not None]
            agg[f"{mk}_mean"] = float(np.mean(vals)) if vals else None
            agg[f"{mk}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cv_summary[key] = agg

        pooled_m = compute_binary_metrics(
            y_all,
            oof_df[prob_col].to_numpy(dtype=np.float64),
            oof_df[pred_col].to_numpy(dtype=np.int64),
        )
        pooled_m["label"] = label
        pooled_summary[key] = pooled_m

    return cv_summary, pooled_summary


def compute_per_exercise_and_source(oof_df: pd.DataFrame) -> tuple[list[dict], list[dict], dict]:
    """Phân rã theo 8 bài tập, 3 nguồn dữ liệu và kiểm định cặp McNemar (RF vs Hybrid)."""
    ex_rows = []
    for ex in GROUP_A_EXERCISES:
        sub = oof_df[oof_df["exercise"] == ex]
        y_ex = sub["y_true"].to_numpy(dtype=np.int64)
        row = {
            "exercise": ex,
            "n_total": int(len(sub)),
            "n_correct": int((y_ex == 0).sum()),
            "n_incorrect": int((y_ex == 1).sum()),
        }
        for key, _, prob_col, pred_col, _ in MODEL_SPECS:
            m = compute_binary_metrics(
                y_ex,
                sub[prob_col].to_numpy(dtype=np.float64),
                sub[pred_col].to_numpy(dtype=np.int64),
            )
            row[f"{key}_acc"] = m["accuracy"]
            row[f"{key}_macro_f1"] = m["macro_f1"]
            row[f"{key}_recall_inc"] = m["recall_incorrect"]
            row[f"{key}_auc"] = m["roc_auc"]
        ex_rows.append(row)

    src_rows = []
    for src in sorted(oof_df["source"].unique().tolist()):
        sub = oof_df[oof_df["source"] == src]
        y_s = sub["y_true"].to_numpy(dtype=np.int64)
        rf_m = compute_binary_metrics(y_s, sub["prob_rf"].to_numpy(), sub["pred_rf"].to_numpy())
        hyb_m = compute_binary_metrics(y_s, sub["prob_hybrid"].to_numpy(), sub["pred_hybrid"].to_numpy())
        src_rows.append(
            {
                "source": src,
                "n_total": int(len(sub)),
                "n_correct": int((y_s == 0).sum()),
                "n_incorrect": int((y_s == 1).sum()),
                "rf_acc": rf_m["accuracy"],
                "rf_spec": rf_m["specificity_correct"],
                "rf_macro_f1": rf_m["macro_f1"],
                "hybrid_acc": hyb_m["accuracy"],
                "hybrid_spec": hyb_m["specificity_correct"],
                "hybrid_bacc": hyb_m["balanced_accuracy"],
                "hybrid_macro_f1": hyb_m["macro_f1"],
                "hybrid_recall_inc": hyb_m["recall_incorrect"],
            }
        )

    y_all = oof_df["y_true"].to_numpy(dtype=np.int64)
    rf_ok = oof_df["pred_rf"].to_numpy(dtype=np.int64) == y_all
    hyb_ok = oof_df["pred_hybrid"].to_numpy(dtype=np.int64) == y_all
    both_ok = int((rf_ok & hyb_ok).sum())
    hyb_fixed = int((~rf_ok & hyb_ok).sum())
    rf_fixed = int((rf_ok & ~hyb_ok).sum())
    both_wrong = int((~rf_ok & ~hyb_ok).sum())
    n_disc = hyb_fixed + rf_fixed
    p_val = float(binomtest(min(hyb_fixed, rf_fixed), n_disc, 0.5).pvalue) if n_disc > 0 else 1.0
    mcnemar = {
        "both_correct": both_ok,
        "hybrid_fixed_rf": hyb_fixed,
        "rf_fixed_hybrid": rf_fixed,
        "both_wrong": both_wrong,
        "discordant_pairs": n_disc,
        "exact_mcnemar_p_value": p_val,
    }
    return ex_rows, src_rows, mcnemar


def compute_rule_engine_metrics(oof_df: pd.DataFrame) -> dict:
    """Đánh giá độ chính xác chẩn đoán mã lỗi của BiomechanicalRuleEngine trên các mẫu Sai."""
    inc_df = oof_df[(oof_df["y_true"] == 1) & (oof_df["error_reason"] != "none")].copy()
    n_inc = len(inc_df)
    top1_hits = (inc_df["rule_top1_code"] == inc_df["error_reason"]).to_numpy()
    top2_hits = np.asarray(
        [yt in (t1, t2) for yt, t1, t2 in zip(inc_df["error_reason"], inc_df["rule_top1_code"], inc_df["rule_top2_code"])],
        dtype=bool,
    )
    top3_hits = np.asarray(
        [
            yt in (t1, t2, t3)
            for yt, t1, t2, t3 in zip(
                inc_df["error_reason"],
                inc_df["rule_top1_code"],
                inc_df["rule_top2_code"],
                inc_df["rule_top3_code"],
            )
        ],
        dtype=bool,
    )

    per_ex = []
    for ex in GROUP_A_EXERCISES:
        sub = inc_df[inc_df["exercise"] == ex]
        n_cand = max(1, len(EXERCISE_ERROR_RULES.get(ex, {})))
        if len(sub) == 0:
            continue
        t1 = float((sub["rule_top1_code"] == sub["error_reason"]).mean())
        t3 = float(
            np.mean(
                [
                    yt in (c1, c2, c3)
                    for yt, c1, c2, c3 in zip(
                        sub["error_reason"], sub["rule_top1_code"], sub["rule_top2_code"], sub["rule_top3_code"]
                    )
                ]
            )
        )
        per_ex.append(
            {
                "exercise": ex,
                "n_incorrect": int(len(sub)),
                "num_rule_codes": n_cand,
                "random_baseline": round(1.0 / n_cand, 4),
                "top1_recall": t1,
                "top3_recall": t3,
            }
        )

    # Đánh giá đầu cuối (chẩn đoán khi Hybrid nhận diện là Sai)
    cor_df = oof_df[oof_df["y_true"] == 0]
    e2e_cor_clean = float((cor_df["e2e_error_code"] == "none").mean()) if len(cor_df) else 0.0
    e2e_inc_exact = float((inc_df["e2e_error_code"] == inc_df["error_reason"]).mean()) if n_inc else 0.0

    return {
        "n_incorrect_evaluated": int(n_inc),
        "top1_recall": float(top1_hits.mean()) if n_inc else 0.0,
        "top2_recall": float(top2_hits.mean()) if n_inc else 0.0,
        "top3_recall": float(top3_hits.mean()) if n_inc else 0.0,
        "e2e_correct_no_false_alarm_rate": e2e_cor_clean,
        "e2e_incorrect_exact_detect_and_diagnose_rate": e2e_inc_exact,
        "per_exercise": per_ex,
    }


def plot_eval_fig1_training_curves(history_json: dict, out_path: Path) -> None:
    """Biểu đồ 1: Đường cong Train Loss, Val Loss và Val Macro-F1 qua các Fold của 2 mô hình Deep Learning."""
    folds = history_json["folds"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=300)

    specs = [
        ("cnn_bilstm", "ExerciseConditionedCNNBiLSTM", 0),
        ("transformer", "ExerciseConditionedTransformer", 1),
    ]
    for model_key, title_name, row_idx in specs:
        ax_loss = axes[row_idx, 0]
        ax_f1 = axes[row_idx, 1]

        for f in folds:
            k = f["fold"]
            h = f[model_key]["history"]
            ep = h["epoch"]
            best_ep = f[model_key]["best_epoch"]
            best_idx = ep.index(best_ep) if best_ep in ep else int(np.argmax(h["val_macro_f1"]))

            ax_loss.plot(ep, h["train_loss"], alpha=0.45, lw=1.3, linestyle="--", label=f"F{k} Train" if row_idx == 0 else None)
            line_val, = ax_loss.plot(ep, h["val_loss"], alpha=0.85, lw=1.8, label=f"Fold {k} Val")
            ax_f1.plot(ep, h["val_macro_f1"], color=line_val.get_color(), alpha=0.85, lw=1.8, label=f"Fold {k} (best ep {best_ep})")
            ax_f1.scatter([ep[best_idx]], [h["val_macro_f1"][best_idx]], color=line_val.get_color(), edgecolor="black", s=45, zorder=5)

        ax_loss.set_title(f"{title_name}: Train Loss (nét đứt) vs Val Loss (nét liền)", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Focal BCE Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend(fontsize=8, ncol=2)

        ax_f1.set_title(f"{title_name}: Val Macro-F1 & Điểm chọn Checkpoint (*)", fontweight="bold")
        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Validation Macro-F1")
        ax_f1.set_ylim(0.35, 1.02)
        ax_f1.grid(alpha=0.25)
        ax_f1.legend(fontsize=8, loc="lower right")

    fig.suptitle("Quá trình hội tụ & Early Stopping qua 5 Fold Cross-Validation", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eval_fig2_model_comparison_cv(
    cv_summary: dict,
    out_path: Path,
    audit: dict | None = None,
) -> None:
    """Biểu đồ 2: So sánh 4 mô hình (Mean ± Std trên 5 Fold) qua 4 chỉ số cốt lõi."""
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("roc_auc", "ROC-AUC"),
    ]
    x = np.arange(len(metrics))
    width = 0.19

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    for idx, (m_key, m_label, _, _, color) in enumerate(MODEL_SPECS):
        means = [cv_summary[m_key][f"{mk}_mean"] * 100.0 for mk, _ in metrics]
        stds = [cv_summary[m_key][f"{mk}_std"] * 100.0 for mk, _ in metrics]
        offset = (idx - 1.5) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=m_label,
            color=color,
            alpha=0.92,
            edgecolor="black",
            linewidth=0.6,
        )
        for b, m_val, s_val in zip(bars, means, stds):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                b.get_height() + s_val + 1.0,
                f"{m_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in metrics], fontsize=11, fontweight="bold")
    ax.set_ylabel("Phần trăm (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(45, 118)
    ax.grid(axis="y", alpha=0.28)
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=9.5)
    n_oof = audit["total_oof_clips"] if audit else "?"
    n_vid = audit["unique_video_id_groups"] if audit else "?"
    ax.set_title(
        f"So sánh hiệu năng 4 mô hình Task 2 (Trung bình ± Độ lệch chuẩn trên 5-Fold CV, N = {n_oof} / {n_vid} nhóm video_id)",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _draw_cm(ax: plt.Axes, cm: np.ndarray, title: str) -> None:
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
    labels = ["Correct (0)", "Incorrect (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Nhãn dự đoán", fontsize=10)
    ax.set_ylabel("Nhãn thật", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    row_sums = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            pct = 100.0 * cm[i, j] / max(1, int(row_sums[i, 0]))
            txt_color = "white" if cm[i, j] > cm.max() * 0.55 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, fontsize=11, fontweight="bold")


def plot_eval_fig3_confusion_and_per_exercise(
    oof_df: pd.DataFrame,
    ex_rows: list[dict],
    out_path: Path,
) -> None:
    """Biểu đồ 3: Ma trận nhầm lẫn Pooled (RF vs Hybrid) & Macro-F1 chi tiết theo 8 bài tập."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), dpi=300, gridspec_kw={"width_ratios": [1, 1, 1.45]})

    y_true = oof_df["y_true"].to_numpy(dtype=np.int64)
    cm_rf = confusion_matrix(y_true, oof_df["pred_rf"].to_numpy(dtype=np.int64), labels=[0, 1])
    cm_hyb = confusion_matrix(y_true, oof_df["pred_hybrid"].to_numpy(dtype=np.int64), labels=[0, 1])

    n_oof = len(oof_df)
    _draw_cm(axes[0], cm_rf, f"Ma trận nhầm lẫn Pooled (N={n_oof})\nBiomechanicalRandomForest")
    _draw_cm(axes[1], cm_hyb, f"Ma trận nhầm lẫn Pooled (N={n_oof})\nHybridFormEnsemble")

    ex_sorted = sorted(ex_rows, key=lambda r: r["hybrid_macro_f1"])
    y_pos = np.arange(len(ex_sorted))
    h = 0.36
    rf_f1 = [r["rf_macro_f1"] * 100.0 for r in ex_sorted]
    hyb_f1 = [r["hybrid_macro_f1"] * 100.0 for r in ex_sorted]

    ax3 = axes[2]
    ax3.barh(y_pos - h / 2, rf_f1, h, label="BiomechanicalRF", color="#3B528B", alpha=0.88)
    bars_h = ax3.barh(y_pos + h / 2, hyb_f1, h, label="Hybrid Ensemble", color="#D95F02", alpha=0.92)
    for b, r in zip(bars_h, ex_sorted):
        ax3.text(
            b.get_width() + 1.0,
            b.get_y() + b.get_height() / 2.0,
            f"{b.get_width():.1f}% (n={r['n_total']})",
            va="center",
            fontsize=8.3,
            fontweight="bold",
        )

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([r["exercise"] for r in ex_sorted], fontsize=9.5, fontweight="bold")
    ax3.set_xlabel("Macro-F1 Out-of-Fold (%)", fontsize=10, fontweight="bold")
    ax3.set_xlim(25, 124)
    ax3.grid(axis="x", alpha=0.25)
    ax3.legend(loc="lower right", fontsize=8.8, framealpha=0.92)
    ax3.set_title("Hiệu năng theo từng bài tập (8 bài Nhóm A)", fontsize=11, fontweight="bold", pad=10)

    fig.suptitle(f"Phân tích Lỗi Phân Loại Toàn Cục (N = {n_oof} Rep Độc Lập)", fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_representative_pair(oof_df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Chọn cặp mẫu Test (1 Correct dự đoán đúng, 1 Incorrect dự đoán đúng) để minh họa Attention XAI."""
    for target_ex in ("squat", "push-up", "hip thrust", "pull up"):
        cor_cand = oof_df[
            (oof_df["exercise"] == target_ex) & (oof_df["y_true"] == 0) & (oof_df["pred_hybrid"] == 0)
        ].sort_values("prob_hybrid", ascending=True)
        inc_cand = oof_df[
            (oof_df["exercise"] == target_ex) & (oof_df["y_true"] == 1) & (oof_df["pred_hybrid"] == 1)
        ].sort_values("prob_hybrid", ascending=False)
        if not cor_cand.empty and not inc_cand.empty:
            return (
                target_ex,
                str(cor_cand.iloc[0]["segment_id"]),
                str(inc_cand.iloc[0]["segment_id"]),
            )
    return None


def plot_eval_fig4_attention_vs_biomechanics(
    oof_df: pd.DataFrame,
    npz_data: np.lib.npyio.NpzFile,
    out_path: Path,
) -> None:
    """Biểu đồ 4: Minh họa Explainable AI — Góc khớp chủ đạo kết hợp trọng số Temporal Attention (32 frame)."""
    pair = _pick_representative_pair(oof_df)
    if pair is None:
        return
    ex_name, cor_sid, inc_sid = pair
    driver_key = EXERCISE_BIOMECH_SPEC[ex_name]["driver"]
    if driver_key not in ANATOMICAL_GOLDEN_KEYS:
        driver_key = "knee_angle" if "knee_angle" in ANATOMICAL_GOLDEN_KEYS else ANATOMICAL_GOLDEN_KEYS[0]
    drv_idx = ANATOMICAL_GOLDEN_KEYS.index(driver_key)
    torso_idx = ANATOMICAL_GOLDEN_KEYS.index("torso_tilt")

    sids = npz_data["segment_id"].tolist()
    sid_to_pos = {str(s): i for i, s in enumerate(sids)}
    oof_lookup = oof_df.set_index("segment_id")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=300, sharey=False)
    t_axis = np.arange(npz_data["anat_angles_10"].shape[1])

    panels = [
        (axes[0], cor_sid, "ĐÚNG KỸ THUẬT (Correct)", COLOR_CORRECT),
        (axes[1], inc_sid, "SAI KỸ THUẬT (Incorrect)", COLOR_INCORRECT),
    ]
    for ax, sid, tag_label, main_col in panels:
        pos = sid_to_pos[sid]
        meta = oof_lookup.loc[sid]
        drv_curve = npz_data["anat_angles_10"][pos, :, drv_idx]
        torso_curve = npz_data["anat_angles_10"][pos, :, torso_idx]
        cnn_a = npz_data["cnn_attn"][pos]
        trn_a = npz_data["trans_attn"][pos]
        mean_attn = 0.5 * (cnn_a + trn_a)

        ax_attn = ax.twinx()
        ax_attn.bar(t_axis, mean_attn * 100.0, width=0.72, color="#FF9800", alpha=0.25, label="Attention TB (%)")
        ax_attn.plot(t_axis, cnn_a * 100.0, color="#E65100", lw=1.2, linestyle=":", label="CNN-BiLSTM Attn (%)")
        ax_attn.set_ylabel("Trọng số Temporal Attention (%)", color="#B26A00", fontsize=9.5)
        max_attn_pct = float(max(np.max(mean_attn), np.max(cnn_a)) * 100.0)
        ax_attn.set_ylim(0, max(12.0, max_attn_pct * 1.45))

        ax.plot(t_axis, drv_curve, color=main_col, lw=2.4, marker="o", markersize=3.5, label=f"{driver_key} (°)")
        ax.plot(t_axis, torso_curve, color="#3B528B", lw=1.8, linestyle="--", label="torso_tilt (°)")
        y_top = float(max(np.max(drv_curve), np.max(torso_curve)))
        ax.set_ylim(0, max(90.0, y_top * 1.25))

        peak_t = int(np.argmax(mean_attn))
        text_x = peak_t - 8 if peak_t > 20 else peak_t + 2
        ax.axvline(peak_t, color="#C62828", linestyle="-.", lw=1.1, alpha=0.8)
        ax.annotate(
            f"Focus đỉnh @ f={peak_t}",
            xy=(peak_t, drv_curve[peak_t]),
            xytext=(text_x, y_top * 1.08),
            arrowprops={"arrowstyle": "->", "color": "#C62828", "lw": 1.0},
            fontsize=8.5,
            fontweight="bold",
            color="#C62828",
        )

        err_info = f" | Lỗi: {meta['error_reason']}" if int(meta["y_true"]) == 1 else ""
        ax.set_title(
            f"[{ex_name.upper()}] {tag_label}\n{sid} (P_hyb={float(meta['prob_hybrid'])*100:.1f}%{err_info})",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.set_xlabel("Mốc thời gian nhảy cóc chuẩn hóa (T = 0..31)", fontsize=9.5)
        ax.set_ylabel("Góc cơ sinh học (độ)", fontsize=9.5, fontweight="bold")
        ax.grid(alpha=0.25)

        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax_attn.get_legend_handles_labels()
        ax_attn.legend(handles1 + handles2, labels1 + labels2, loc="upper left", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "Explainable AI: Đối chiếu Quỹ đạo Góc khớp Thực tế và Dải Tập Trung Thời Gian (Temporal Attention)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_eval_fig5_explainability_and_features(
    imp_df: pd.DataFrame,
    rule_metrics: dict,
    out_path: Path,
) -> None:
    """Biểu đồ 5: Top-15 Đặc trưng quan trọng (5-Fold Mean ± Std) & Top-1/Top-3 Recall của Rule Engine."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)

    # Panel 1: Top-15 RF Feature Importances
    stats = (
        imp_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .head(15)
        .iloc[::-1]
    )
    y_pos = np.arange(len(stats))
    colors = [
        "#7570B3"
        if name.startswith("rule_")
        else ("#D95F02" if "golden" in name else ("#1B9E77" if ("speed" in name or "bouncing" in name or "duration" in name) else "#3B528B"))
        for name in stats.index
    ]
    ax1.barh(y_pos, stats["mean"].to_numpy(), xerr=stats["std"].to_numpy(), capsize=3, color=colors, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(stats.index.tolist(), fontsize=9)
    ax1.set_xlabel("Mức độ quan trọng trung bình qua 5 Fold (RF + ExtraTrees)", fontsize=9.5, fontweight="bold")
    ax1.set_title("Top-15 Đặc trưng Cơ sinh học Quyết định nhất (106-d E.3+)\n(Tím: Luật | Cam: Golden-Rep | Xanh lá: Nhịp | Xanh dương: Góc chuẩn bài)", fontsize=10.0, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    # Panel 2: Rule Engine Top-1 vs Top-3 Recall theo từng bài tập
    per_ex = rule_metrics["per_exercise"]
    ex_names = [r["exercise"] for r in per_ex]
    x = np.arange(len(ex_names))
    w = 0.34
    t1_vals = [r["top1_recall"] * 100.0 for r in per_ex]
    t3_vals = [r["top3_recall"] * 100.0 for r in per_ex]
    rand_vals = [r["random_baseline"] * 100.0 for r in per_ex]

    b1 = ax2.bar(x - w / 2, t1_vals, w, label=f"Top-1 Recall (TB={rule_metrics['top1_recall']*100:.1f}%)", color="#3B528B", alpha=0.9)
    b3 = ax2.bar(x + w / 2, t3_vals, w, label=f"Top-3 Recall (TB={rule_metrics['top3_recall']*100:.1f}%)", color="#1B9E77", alpha=0.9)
    ax2.plot(x, rand_vals, color="#C62828", marker="D", lw=1.4, linestyle="--", label="Đoán ngẫu nhiên (1 / số mã lỗi)")

    for bar in b1:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.0f}%", ha="center", fontsize=7.8)
    for bar in b3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.0f}%", ha="center", fontsize=7.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['exercise']}\n(n={r['n_incorrect']})" for r in per_ex], rotation=22, ha="right", fontsize=8.5)
    ax2.set_ylabel("Độ chính xác chẩn đoán mã lỗi (%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 128)
    ax2.set_title(
        f"Năng lực chẩn đoán 32 Mã lỗi của BiomechanicalRuleEngine\n(Đánh giá 5-Fold OOF trên toàn bộ {rule_metrics['n_incorrect_evaluated']} mẫu Sai)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper right", ncol=2, fontsize=8.0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    audit: dict,
    cv_summary: dict,
    pooled_summary: dict,
    ex_rows: list[dict],
    src_rows: list[dict],
    mcnemar: dict,
    rule_metrics: dict,
    imp_df: pd.DataFrame,
    oof_df: pd.DataFrame,
    out_md_path: Path,
) -> None:
    """Xuất báo cáo Markdown hoàn chỉnh từ dữ liệu đo thực tế."""
    top10_feats = (
        imp_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    oof_err = oof_df.copy()
    oof_err["is_err"] = (oof_err["pred_hybrid"] != oof_err["y_true"]).astype(int)
    oof_err["conf"] = np.where(oof_err["pred_hybrid"] == 1, oof_err["prob_hybrid"], 1.0 - oof_err["prob_hybrid"])
    hard_cases = oof_err[oof_err["is_err"] == 1].sort_values("conf", ascending=False).head(8)

    lines = [
        "# Báo cáo Đánh giá Task 2 (5-Fold Stratified Group Cross-Validation)",
        "",
        "## 1. Hậu kiểm Zero Data Leakage",
        f"- Tổng số mẫu Out-of-Fold: **{audit['total_oof_clips']}** (`{audit['n_correct']} correct` / `{audit['n_incorrect']} incorrect`)",
        f"- Số `segment_id` duy nhất: **{audit['unique_segments']}**",
        f"- Số nhóm `video_id` độc lập: **{audit['unique_video_id_groups']}**",
        f"- Số `video_id` rò rỉ giữa các Fold: **{audit['cross_fold_video_id_leaks']}**",
        "",
        f"## 2. So sánh 4 Mô hình qua 5 Fold (Mean ± Std) & Pooled OOF (N = {audit['total_oof_clips']})",
        "| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC | Pooled Macro-F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for key, label, *_ in MODEL_SPECS:
        c = cv_summary[key]
        p = pooled_summary[key]
        lines.append(
            f"| {label} | "
            f"{c['accuracy_mean']*100:.2f} ± {c['accuracy_std']*100:.2f}% | "
            f"{c['balanced_accuracy_mean']*100:.2f} ± {c['balanced_accuracy_std']*100:.2f}% | "
            f"{c['macro_f1_mean']*100:.2f} ± {c['macro_f1_std']*100:.2f}% | "
            f"{c['recall_incorrect_mean']*100:.2f} ± {c['recall_incorrect_std']*100:.2f}% | "
            f"{c['roc_auc_mean']:.3f} ± {c['roc_auc_std']:.3f} | "
            f"**{p['macro_f1']*100:.2f}%** |"
        )

    lines.extend([
        "",
        "### Đối chiếu cặp RF vs Hybrid Ensemble (McNemar Exact)",
        f"- Cả 2 cùng đúng: **{mcnemar['both_correct']}** | Hybrid sửa đúng ca RF sai: **{mcnemar['hybrid_fixed_rf']}** | "
        f"RF đúng ca Hybrid sai: **{mcnemar['rf_fixed_hybrid']}** | Cả 2 cùng sai: **{mcnemar['both_wrong']}** "
        f"(p-value = `{mcnemar['exact_mcnemar_p_value']:.4f}`).",
        "",
        "## 3. Phân rã theo 8 Bài tập Nhóm A (Pooled OOF)",
        "| Bài tập | Tổng (Đúng/Sai) | RF Acc | RF Macro-F1 | CNN-BiLSTM MF1 | Transformer MF1 | Hybrid Acc | Hybrid Macro-F1 | Hybrid Recall (Sai) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in ex_rows:
        lines.append(
            f"| {r['exercise']} | {r['n_total']} ({r['n_correct']}/{r['n_incorrect']}) | "
            f"{r['rf_acc']*100:.1f}% | {r['rf_macro_f1']*100:.1f}% | "
            f"{r['cnn_bilstm_macro_f1']*100:.1f}% | {r['transformer_macro_f1']*100:.1f}% | "
            f"{r['hybrid_acc']*100:.1f}% | **{r['hybrid_macro_f1']*100:.1f}%** | {r['hybrid_recall_inc']*100:.1f}% |"
        )

    lines.extend([
        "",
        "## 4. Phân rã theo Nguồn Dữ liệu (EFA / GYMED / YouTube)",
        "| Nguồn | Số mẫu (Đúng/Sai) | RF Acc | RF Spec (Đúng) | RF Macro-F1 | Hybrid Acc | Hybrid Spec (Đúng) | Hybrid Macro-F1 | Hybrid Recall (Sai) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for s in src_rows:
        lines.append(
            f"| {s['source']} | {s['n_total']} ({s['n_correct']}/{s['n_incorrect']}) | "
            f"{s['rf_acc']*100:.1f}% | {s['rf_spec']*100:.1f}% | {s['rf_macro_f1']*100:.1f}% | "
            f"{s['hybrid_acc']*100:.1f}% | {s['hybrid_spec']*100:.1f}% | {s['hybrid_macro_f1']*100:.1f}% | {s['hybrid_recall_inc']*100:.1f}% |"
        )

    lines.extend([
        "",
        f"## 5. Đánh giá BiomechanicalRuleEngine trên {rule_metrics['n_incorrect_evaluated']} Mẫu Sai",
        f"- **Chẩn đoán thuần (is_incorrect=True):** Top-1 Recall = **{rule_metrics['top1_recall']*100:.1f}%** | "
        f"Top-2 Recall = **{rule_metrics['top2_recall']*100:.1f}%** | Top-3 Recall = **{rule_metrics['top3_recall']*100:.1f}%**",
        f"- **Đầu cuối (qua bộ lọc Hybrid):** Tỷ lệ rep Đúng không bị báo lỗi nhầm = **{rule_metrics['e2e_correct_no_false_alarm_rate']*100:.1f}%** | "
        f"Tỷ lệ rep Sai vừa bắt đúng Sai vừa gọi đúng lỗi Top-1 = **{rule_metrics['e2e_incorrect_exact_detect_and_diagnose_rate']*100:.1f}%**",
        "",
        "## 6. Top-10 Đặc trưng Cơ sinh học Quan trọng nhất (RF + ExtraTrees)",
    ])
    for fname, val in top10_feats.items():
        lines.append(f"- `{fname}`: `{val:.4f}`")

    lines.extend([
        "",
        "## 7. Các Ca Sai Tự tin Cao nhất của Hybrid (Error Analysis)",
        "| segment_id | Bài tập | Nguồn | Nhãn thật | error_reason | P(incorrect) | Dự đoán Hybrid |",
        "| :--- | :--- | :---: | :---: | :--- | :---: | :---: |",
    ])
    for _, row in hard_cases.iterrows():
        err_code = str(row["error_reason"])
        err_txt = f"{err_code} ({ERROR_REASON_VN_DICT.get(err_code, err_code)})" if err_code != "none" else "none"
        lines.append(
            f"| {row['segment_id']} | {row['exercise']} | {row['source']} | "
            f"{'incorrect' if row['y_true'] == 1 else 'correct'} | {err_txt} | "
            f"{row['prob_hybrid']*100:.1f}% | {'incorrect' if row['pred_hybrid'] == 1 else 'correct'} |"
        )

    out_md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    oof_csv = REPORTS_DIR / "cv_oof_test_predictions.csv"
    hist_json_path = REPORTS_DIR / "cv_training_history.json"
    npz_path = REPORTS_DIR / "cv_attention_and_angles.npz"
    imp_csv = REPORTS_DIR / "cv_rf_feature_importances.csv"

    for req in (oof_csv, hist_json_path, npz_path, imp_csv):
        if not req.exists():
            raise FileNotFoundError(
                f"Không tìm thấy tệp báo cáo huấn luyện: {req}. Hãy chạy `python experiments/train_task2.py` trước!"
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.read_csv(oof_csv)
    with open(hist_json_path, "r", encoding="utf-8") as f:
        history_json = json.load(f)
    imp_df = pd.read_csv(imp_csv)
    npz_data = np.load(npz_path, allow_pickle=False)

    print("=" * 84)
    print("FORGAI STEP E - ĐÁNH GIÁ TOÀN DIỆN & TRỰC QUAN HÓA KẾT QUẢ 5-FOLD CV")
    print("=" * 84)

    audit = audit_zero_leakage(oof_df)
    print(
        f"[1/4] Hậu kiểm Zero Data Leakage: {audit['total_oof_clips']} clips | "
        f"{audit['unique_video_id_groups']} nhóm video_id | 0 giao nhau các Fold -> PASS"
    )

    cv_summary, pooled_summary = compute_cv_and_pooled_tables(oof_df)
    ex_rows, src_rows, mcnemar = compute_per_exercise_and_source(oof_df)
    rule_metrics = compute_rule_engine_metrics(oof_df)

    print("[2/4] Hiệu năng 5-Fold CV (Mean ± Std) & Pooled OOF:")
    for key, label, *_ in MODEL_SPECS:
        c = cv_summary[key]
        p = pooled_summary[key]
        print(
            f"      - {label:28s}: CV Acc={c['accuracy_mean']*100:.2f}±{c['accuracy_std']*100:.2f}% | "
            f"CV MF1={c['macro_f1_mean']*100:.2f}±{c['macro_f1_std']*100:.2f}% | "
            f"CV AUC={c['roc_auc_mean']:.3f} | Pooled MF1={p['macro_f1']*100:.2f}%"
        )
    print(
        f"[3/4] Rule Engine ({rule_metrics['n_incorrect_evaluated']} mẫu Sai): "
        f"Top-1 Recall={rule_metrics['top1_recall']*100:.1f}%, "
        f"Top-3 Recall={rule_metrics['top3_recall']*100:.1f}%"
    )

    # Xuất 5 biểu đồ 300-DPI
    fig_paths = {
        "fig1": FIGURES_DIR / "eval_fig1_training_curves.png",
        "fig2": FIGURES_DIR / "eval_fig2_model_comparison_cv.png",
        "fig3": FIGURES_DIR / "eval_fig3_confusion_and_per_exercise.png",
        "fig4": FIGURES_DIR / "eval_fig4_attention_vs_biomechanics.png",
        "fig5": FIGURES_DIR / "eval_fig5_explainability_and_features.png",
    }
    plot_eval_fig1_training_curves(history_json, fig_paths["fig1"])
    plot_eval_fig2_model_comparison_cv(cv_summary, fig_paths["fig2"], audit=audit)
    plot_eval_fig3_confusion_and_per_exercise(oof_df, ex_rows, fig_paths["fig3"])
    plot_eval_fig4_attention_vs_biomechanics(oof_df, npz_data, fig_paths["fig4"])
    plot_eval_fig5_explainability_and_features(imp_df, rule_metrics, fig_paths["fig5"])

    summary_payload = {
        "zero_leakage_audit": audit,
        "cv_summary": cv_summary,
        "pooled_oof_summary": pooled_summary,
        "per_exercise": ex_rows,
        "per_source": src_rows,
        "mcnemar_rf_vs_hybrid": mcnemar,
        "rule_engine_metrics": rule_metrics,
        "figures": {k: str(v) for k, v in fig_paths.items()},
    }
    json_out = REPORTS_DIR / "task2_eval_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    md_out = REPORTS_DIR / "task2_eval_report.md"
    write_markdown_report(
        audit=audit,
        cv_summary=cv_summary,
        pooled_summary=pooled_summary,
        ex_rows=ex_rows,
        src_rows=src_rows,
        mcnemar=mcnemar,
        rule_metrics=rule_metrics,
        imp_df=imp_df,
        oof_df=oof_df,
        out_md_path=md_out,
    )
    print(f"[4/4] Đã xuất 5 biểu đồ tại {FIGURES_DIR} và báo cáo tổng hợp:")
    print(f"      - {json_out}")
    print(f"      - {md_out}")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    main()
