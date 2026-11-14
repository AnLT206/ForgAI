# -*- coding: utf-8 -*-
"""
tools/visualize_task2.py
Trực quan hóa toàn diện bộ dữ liệu Nhóm A (8 bài tập) cho Bước 3 AI SIC:
  1. fig1_before_after_supplementation.png : Đối chiếu TRƯỚC vs SAU khi bổ sung YouTube & Augmentation x6.
  2. fig2_source_and_group_distribution.png : Phân bố 3 nguồn (EFA, GYMED, YouTube) & Nhóm video_id chống Data Leakage.
  3. fig3_error_reasons_breakdown.png       : Phân bố chi tiết 32 mã lỗi cơ sinh học theo từng bài tập (lưới 2x4).
  4. fig4_rep_duration_and_strided_sampling.png : Phân bố độ dài rep & Minh họa lấy mẫu nhảy cóc T=32 trên quỹ đạo thật.
  5. fig5_skeleton_13joints_correct_vs_incorrect.png : Vẽ khung xương 13 khớp & đồ thị góc khớp Đúng vs Sai từ file .npy.
"""

import sys
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    TASK2_CSV_PATH,
    POSE_NPY_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    GROUP_A_EXERCISES,
    NUM_FRAMES,
    TRAIN_STRIDE_OFFSETS,
    ENABLE_HORIZONTAL_FLIP,
)
from src.constants import SKELETON_EDGES_13, J_IDX

if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Thiết lập font hiển thị tiếng Việt sắc nét trên Windows
plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLOR_CORRECT = "#1B9E77"
COLOR_INCORRECT = "#D95F02"
COLOR_ALERT = "#C62828"
COLOR_EFA = "#3B528B"
COLOR_GYMED = "#21918C"
COLOR_YT = "#FDE725"


def _calc_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Tính góc tại khớp b tạo bởi hai đoạn (a-b) và (c-b) theo độ [0..180]."""
    v1 = a - b
    v2 = c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-8:
        return np.nan
    cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _best_side_angle_series(pose_3d: np.ndarray, vis: np.ndarray, left_triplet: tuple, right_triplet: tuple) -> np.ndarray:
    """Tính chuỗi góc theo thời gian, tự động chọn bên rõ hơn theo visibility ở từng rep."""
    vis_l = float(vis[:, list(left_triplet)].mean())
    vis_r = float(vis[:, list(right_triplet)].mean())
    use_left = vis_l >= vis_r
    triplet = left_triplet if use_left else right_triplet

    angles = []
    for t in range(len(pose_3d)):
        if np.abs(pose_3d[t]).sum() < 1e-6:
            angles.append(np.nan)
        else:
            angles.append(_calc_angle_3d(pose_3d[t, triplet[0]], pose_3d[t, triplet[1]], pose_3d[t, triplet[2]]))
    s = pd.Series(angles).interpolate(limit_direction="both").to_numpy()
    return s


def plot_fig1_before_after(df: pd.DataFrame, out_path: Path):
    """Hình 1: Đối chiếu TRƯỚC bổ sung YouTube vs SAU bổ sung YouTube vs SAU Augmentation x6."""
    df_before = df[df["source"].isin(["EFA", "GYMED"])]
    exs = GROUP_A_EXERCISES
    y_pos = np.arange(len(exs))

    b_cor = [int(((df_before["exercise"] == e) & (df_before["form_label"] == "correct")).sum()) for e in exs]
    b_inc = [int(((df_before["exercise"] == e) & (df_before["form_label"] == "incorrect")).sum()) for e in exs]

    a_cor = [int(((df["exercise"] == e) & (df["form_label"] == "correct")).sum()) for e in exs]
    a_inc = [int(((df["exercise"] == e) & (df["form_label"] == "incorrect")).sum()) for e in exs]

    aug_mult = TRAIN_STRIDE_OFFSETS * (2 if ENABLE_HORIZONTAL_FLIP else 1)
    # Số chuỗi Train sau x6 (tính theo tỷ lệ 80% dữ liệu dùng để Train trong 5-Fold CV)
    t_cor = [int(round(c * 0.8 * aug_mult)) for c in a_cor]
    t_inc = [int(round(inc * 0.8 * aug_mult)) for inc in a_inc]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True)
    fig.suptitle(
        "ĐỐI CHIẾU DỮ LIỆU NHÓM A (8 BÀI TẬP): TRƯỚC BỔ SUNG vs SAU BỔ SUNG vs SAU TĂNG CƯỜNG (x6)",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Panel 1: Trước khi bổ sung YouTube
    ax = axes[0]
    ax.barh(y_pos, b_cor, color=COLOR_CORRECT, alpha=0.88, label="Đúng (correct)")
    ax.barh(y_pos, b_inc, left=b_cor, color=COLOR_INCORRECT, alpha=0.92, label="Sai (incorrect)")
    for i in range(len(exs)):
        tot = b_cor[i] + b_inc[i]
        if b_inc[i] <= 1:
            ax.text(tot + 1.2, i, f"{b_cor[i]}Đ / {b_inc[i]}S (Thiếu nặng!)", va="center", fontsize=9.5, fontweight="bold", color=COLOR_ALERT)
        else:
            ax.text(tot + 1.2, i, f"{b_cor[i]}Đ / {b_inc[i]}S", va="center", fontsize=9.5, color="#222222")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e.title() for e in exs], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xlabel("Số lượng mẫu rep (Segments)", fontsize=10.5)
    ax.set_title(f"1. TRƯỚC BỔ SUNG (EFA + GYMED)\nTổng: {sum(b_cor)+sum(b_inc)} ({sum(b_cor)} Đúng / {sum(b_inc)} Sai)\n4 bài có 0 mẫu Sai!", fontsize=11.5, fontweight="bold", color=COLOR_ALERT)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9.5, frameon=True)

    # Panel 2: Sau khi bổ sung & làm sạch YouTube
    ax = axes[1]
    ax.barh(y_pos, a_cor, color=COLOR_CORRECT, alpha=0.92, label="Đúng (correct)")
    ax.barh(y_pos, a_inc, left=a_cor, color=COLOR_INCORRECT, alpha=0.92, label="Sai (incorrect)")
    for i in range(len(exs)):
        tot = a_cor[i] + a_inc[i]
        ax.text(a_cor[i] / 2, i, str(a_cor[i]), va="center", ha="center", color="white", fontweight="bold", fontsize=10)
        ax.text(a_cor[i] + a_inc[i] / 2, i, str(a_inc[i]), va="center", ha="center", color="white", fontweight="bold", fontsize=10)
        ax.text(tot + 1.2, i, f"Tổng: {tot}", va="center", fontsize=9.5, fontweight="bold", color="#111111")
    n_yt = int((df["source"] == "YouTube").sum())
    inc_ratio = sum(a_inc) / max(1, sum(b_inc))
    n_test_est = int(round(len(df) * 0.2))
    max_after = max(c + inc for c, inc in zip(a_cor, a_inc))
    ax.set_xlim(0, max(82, max_after + 18))
    ax.set_xlabel("Số lượng mẫu rep (Segments)", fontsize=10.5)
    ax.set_title(f"2. SAU BỔ SUNG YOUTUBE (+{n_yt} clip sạch)\nTổng: {len(df)} ({sum(a_cor)} Đúng / {sum(a_inc)} Sai)\nMẫu Sai tăng x{inc_ratio:.2f} lần ({sum(b_inc)} -> {sum(a_inc)})!", fontsize=11.5, fontweight="bold", color="#0B5345")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9.5, frameon=True)

    # Panel 3: Sau Augmentation x6 trên tập Train
    ax = axes[2]
    ax.barh(y_pos, t_cor, color="#2E86AB", alpha=0.90, label="Chuỗi Đúng (Train x6)")
    ax.barh(y_pos, t_inc, left=t_cor, color="#A23B72", alpha=0.90, label="Chuỗi Sai (Train x6)")
    for i in range(len(exs)):
        tot = t_cor[i] + t_inc[i]
        ax.text(tot + 6, i, f"~{tot} chuỗi ({t_cor[i]}/{t_inc[i]})", va="center", fontsize=9.5, fontweight="bold", color="#111111")
    ax.set_xlim(0, 465)
    ax.set_xlabel("Số chuỗi 32x117-d đưa vào học (Train)", fontsize=10.5)
    ax.set_title(f"3. SAU TĂNG CƯỜNG DỮ LIỆU TẬP HỌC (x{aug_mult})\n~{sum(t_cor)+sum(t_inc):,} chuỗi Train (3 pha Offset x Lật gương)\n(+~{n_test_est} mẫu gốc giữ nguyên để Test)", fontsize=11.5, fontweight="bold", color="#1A5276")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9.5, frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_source_and_groups(df: pd.DataFrame, out_path: Path):
    """Hình 2: Phân bố 3 nguồn dữ liệu & Nhóm video_id chống rò rỉ dữ liệu."""
    exs = GROUP_A_EXERCISES
    y_pos = np.arange(len(exs))

    cnt_efa = [int(((df["exercise"] == e) & (df["source"] == "EFA")).sum()) for e in exs]
    cnt_gym = [int(((df["exercise"] == e) & (df["source"] == "GYMED")).sum()) for e in exs]
    cnt_yt = [int(((df["exercise"] == e) & (df["source"] == "YouTube")).sum()) for e in exs]

    seg_cnt = [int((df["exercise"] == e).sum()) for e in exs]
    vid_cnt = [int(df[df["exercise"] == e]["video_id"].nunique()) for e in exs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    fig.suptitle(
        "CƠ CẤU NGUỒN DỮ LIỆU & NHÓM ĐỘC LẬP CHỐNG RÒ RỈ DỮ LIỆU (ZERO DATA LEAKAGE)",
        fontsize=14.5, fontweight="bold", y=0.98
    )

    # Panel trái: Đóng góp 3 nguồn EFA, GYMED, YouTube
    ax = axes[0]
    ax.barh(y_pos, cnt_efa, color=COLOR_EFA, label=f"EFA ({sum(cnt_efa)} mẫu)")
    ax.barh(y_pos, cnt_gym, left=cnt_efa, color=COLOR_GYMED, label=f"GYMED ({sum(cnt_gym)} mẫu)")
    left_yt = [cnt_efa[i] + cnt_gym[i] for i in range(len(exs))]
    ax.barh(y_pos, cnt_yt, left=left_yt, color="#F39C12", label=f"YouTube bổ sung ({sum(cnt_yt)} mẫu)")
    for i in range(len(exs)):
        if cnt_efa[i] >= 5:
            ax.text(cnt_efa[i] / 2, i, str(cnt_efa[i]), va="center", ha="center", color="white", fontweight="bold")
        if cnt_gym[i] >= 5:
            ax.text(cnt_efa[i] + cnt_gym[i] / 2, i, str(cnt_gym[i]), va="center", ha="center", color="white", fontweight="bold")
        if cnt_yt[i] >= 5:
            ax.text(left_yt[i] + cnt_yt[i] / 2, i, str(cnt_yt[i]), va="center", ha="center", color="#111111", fontweight="bold")
        ax.text(seg_cnt[i] + 1.0, i, str(seg_cnt[i]), va="center", fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e.title() for e in exs], fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    max_seg = max(seg_cnt)
    ax.set_xlim(0, max(78, max_seg + 10))
    ax.set_xlabel("Số lượng mẫu rep", fontsize=10.5)
    ax.set_title(f"Đóng góp theo Nguồn Video (Tổng: {len(df)} mẫu)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5, frameon=True)

    # Panel phải: Số Segments vs Số Unique video_id
    ax = axes[1]
    h_bar = 0.36
    ax.barh(y_pos - h_bar / 2, seg_cnt, height=h_bar, color="#3498DB", label=f"Số mẫu Rep (Segments = {sum(seg_cnt)})")
    ax.barh(y_pos + h_bar / 2, vid_cnt, height=h_bar, color="#2C3E50", label=f"Số Video Nguồn độc lập (video_id = {sum(vid_cnt)})")
    for i in range(len(exs)):
        ax.text(seg_cnt[i] + 0.8, i - h_bar / 2, str(seg_cnt[i]), va="center", fontsize=9.5, fontweight="bold", color="#2471A3")
        ax.text(vid_cnt[i] + 0.8, i + h_bar / 2, str(vid_cnt[i]), va="center", fontsize=9.5, fontweight="bold", color="#2C3E50")
    ax.set_xlim(0, max(78, max_seg + 10))
    ax.set_xlabel("Số lượng", fontsize=10.5)
    ax.set_title("Gom nhóm theo video_id (Không bao giờ tách đôi video cùng nguồn)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9.5, frameon=True)

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_error_reasons(df: pd.DataFrame, out_path: Path):
    """Hình 3: Lưới 2x4 thống kê chi tiết 32 mã lỗi cơ sinh học theo 8 bài tập."""
    df_err = df[df["form_label"] == "incorrect"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    fig.suptitle(
        f"PHÂN BỐ {df_err['error_reason'].nunique()} MÃ LỖI CƠ SINH HỌC TRÊN {len(df_err)} MẪU SAI (INCORRECT) CỦA 8 BÀI TẬP",
        fontsize=15, fontweight="bold", y=0.98
    )

    palette = plt.cm.tab10.colors
    for idx, ex in enumerate(GROUP_A_EXERCISES):
        ax = axes[idx // 4, idx % 4]
        sub = df_err[df_err["exercise"] == ex]
        vc = sub["error_reason"].value_counts().sort_values(ascending=True)
        y = np.arange(len(vc))
        bars = ax.barh(y, vc.values, color=palette[idx % len(palette)], alpha=0.88)
        for bar, val in zip(bars, vc.values):
            ax.text(val + 0.18, bar.get_y() + bar.get_height() / 2, str(int(val)), va="center", fontweight="bold", fontsize=10)
        ax.set_yticks(y)
        ax.set_yticklabels(vc.index.tolist(), fontsize=9.5)
        ax.set_xlim(0, max(vc.values) + 2.5)
        ax.set_title(f"{ex.title()} ({len(sub)} mẫu Sai / {len(vc)} mã lỗi)", fontsize=11.5, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_duration_and_sampling(df: pd.DataFrame, out_path: Path):
    """Hình 4: Phân bố độ dài Rep & Minh họa Dynamic Strided Sampling (T=32) trên dữ liệu thật."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.35, wspace=0.22)
    fig.suptitle(
        "ĐỘ DÀI CHU KỲ 1 REP & CƠ CHẾ LẤY MẪU NHẢY CÓC CÁCH ĐỀU (DYNAMIC STRIDED SAMPLING T=32)",
        fontsize=14.5, fontweight="bold", y=0.98
    )

    # Panel trên trái: Boxplot num_frames theo bài tập
    ax1 = fig.add_subplot(gs[0, 0])
    data_frames = [df[df["exercise"] == ex]["num_frames"].values for ex in GROUP_A_EXERCISES]
    bp = ax1.boxplot(data_frames, patch_artist=True, tick_labels=[e.title() for e in GROUP_A_EXERCISES])
    for patch in bp["boxes"]:
        patch.set_facecolor("#5DADE2")
        patch.set_alpha(0.75)
    f_min = int(df["num_frames"].min())
    f_med = int(df["num_frames"].median())
    f_max = int(df["num_frames"].max())
    ax1.axhline(NUM_FRAMES, color=COLOR_ALERT, linestyle="--", linewidth=1.8, label=f"Chuẩn T = {NUM_FRAMES} mốc")
    ax1.set_ylabel("Số lượng khung hình (num_frames)", fontsize=10.5)
    ax1.set_title(f"Độ dài Frame của 1 Rep (Min: {f_min}, Median: {f_med}, Max: {f_max})", fontsize=11.5, fontweight="bold")
    ax1.tick_params(axis="x", rotation=25)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right")

    # Panel trên phải: Histogram duration_sec
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["duration_sec"], bins=24, color="#48C9B0", edgecolor="#1B4F72", alpha=0.85)
    med_dur = float(df["duration_sec"].median())
    ax2.axvline(med_dur, color=COLOR_INCORRECT, linestyle="--", linewidth=2, label=f"Trung vị = {med_dur:.2f}s")
    ax2.set_xlabel("Thời lượng 1 rep (giây)", fontsize=10.5)
    ax2.set_ylabel("Số lượng mẫu rep", fontsize=10.5)
    ax2.set_title(f"Phân bố Thời lượng thực tế của {len(df)} Rep", fontsize=11.5, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    # Panel dưới (trải rộng 2 cột): Minh họa Dynamic Strided Sampling vs Sliding Window trên quỹ đạo gối thật
    ax3 = fig.add_subplot(gs[1, :])
    sample_row = df[(df["exercise"] == "squat") & (df["form_label"] == "correct") & (df["num_frames"] >= 85) & (df["num_frames"] <= 110)].iloc[0]
    npy_path = POSE_NPY_DIR / f"{sample_row['segment_id']}.npy"
    arr = np.load(npy_path)
    knee_angles = _best_side_angle_series(
        arr[:, :, :3], arr[:, :, 5],
        (J_IDX["left_hip"], J_IDX["left_knee"], J_IDX["left_ankle"]),
        (J_IDX["right_hip"], J_IDX["right_knee"], J_IDX["right_ankle"]),
    )
    L = len(knee_angles)
    t_axis = np.arange(L)

    ax3.plot(t_axis, knee_angles, color="#2C3E50", linewidth=2.4, label=f"Quỹ đạo góc đầu gối thật ({sample_row['segment_id']}, L={L} frames)")
    # Vùng Sliding Window cắt 32 frame liên tiếp đầu tiên (Task 1) -> Mất đáy Squat!
    ax3.axvspan(0, min(31, L - 1), color="#E74C3C", alpha=0.18, label="Cắt 32 frame liên tiếp đầu (Task 1) -> BỊ MẤT ĐÁY SQUAT!")
    # 32 mốc nhảy cóc cách đều (Dynamic Strided Sampling T=32)
    stride_idx = np.linspace(0, L - 1, NUM_FRAMES).round().astype(int)
    ax3.scatter(stride_idx, knee_angles[stride_idx], color="#1B9E77", s=65, zorder=5, edgecolor="black", label="Lấy mẫu nhảy cóc T=32 mốc (Task 2) -> Gom trọn Bắt đầu + Đáy + Kết thúc!")
    min_t = int(np.argmin(knee_angles))
    ax3.annotate(
        f"Đáy Squat (Min={knee_angles[min_t]:.1f}° tại f{min_t})",
        xy=(min_t, knee_angles[min_t]),
        xytext=(max(5, min_t - 22), knee_angles[min_t] + 24),
        arrowprops=dict(arrowstyle="->", color=COLOR_ALERT, lw=1.8),
        fontsize=10.5, fontweight="bold", color=COLOR_ALERT,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLOR_ALERT, alpha=0.9),
    )
    ax3.set_ylim(45, 202)
    ax3.set_xlabel("Chỉ số khung hình gốc (Frame Index)", fontsize=10.5)
    ax3.set_ylabel("Góc đầu gối 3D (độ)", fontsize=10.5)
    ax3.set_title("Minh họa Lý do Task 2 BẮT BUỘC dùng Lấy mẫu nhảy cóc (Dynamic Strided Sampling T=32) thay vì cắt 32 frame liên tiếp", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.92)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _pick_clean_sample(df: pd.DataFrame, df_ext: pd.DataFrame, exercise: str, form_label: str, error_reason: str | None = None) -> pd.Series:
    sub = df[(df["exercise"] == exercise) & (df["form_label"] == form_label)]
    if error_reason is not None:
        sub = sub[sub["error_reason"] == error_reason]
    merged = sub.merge(df_ext[["segment_id", "det_rate", "mean_visibility"]], on="segment_id", how="left")
    merged = merged.sort_values(["det_rate", "mean_visibility", "num_frames"], ascending=[False, False, False])
    return merged.iloc[0]


def _draw_skeleton_strip(ax_list, row: pd.Series, color_bgr: tuple, title_prefix: str):
    """Vẽ 5 khung hình tiêu biểu (0%, 25%, 50%, 75%, 100%) kèm bộ khung xương 13 khớp từ file .npy."""
    vid_path = PROJECT_ROOT / row["video_path"]
    npy_path = POSE_NPY_DIR / f"{row['segment_id']}.npy"
    arr = np.load(npy_path)  # (L, 13, 6): wx, wy, wz, ix, iy, vis

    cap = cv2.VideoCapture(str(vid_path))
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()

    L = min(len(frames), len(arr))
    picks = np.linspace(0, L - 1, 5).round().astype(int)
    pct_labels = ["0% (Bắt đầu)", "25% (Hạ)", "50% (Đáy/Đỉnh)", "75% (Lên)", "100% (Kết thúc)"]

    for k, (ax, f_idx) in enumerate(zip(ax_list, picks)):
        img = frames[f_idx].copy()
        h, w = img.shape[:2]
        joints_2d = arr[f_idx, :, 3:5]
        vis = arr[f_idx, :, 5]

        # Vẽ xương nối 13 khớp
        for u, v in SKELETON_EDGES_13:
            if vis[u] > 0.15 and vis[v] > 0.15:
                pt1 = (int(joints_2d[u, 0] * w), int(joints_2d[u, 1] * h))
                pt2 = (int(joints_2d[v, 0] * w), int(joints_2d[v, 1] * h))
                cv2.line(img, pt1, pt2, color_bgr, thickness=max(2, int(min(w, h) / 140)))
        # Vẽ 13 điểm khớp
        for j in range(13):
            if vis[j] > 0.15:
                pt = (int(joints_2d[j, 0] * w), int(joints_2d[j, 1] * h))
                cv2.circle(img, pt, max(3, int(min(w, h) / 90)), (255, 255, 255), -1)
                cv2.circle(img, pt, max(2, int(min(w, h) / 120)), color_bgr, -1)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(title_prefix, fontsize=10.5, fontweight="bold", color="#111111")
        ax.set_title(f"{pct_labels[k]} (f{f_idx})", fontsize=9.5, fontweight="bold")


def plot_fig5_skeleton_comparison(df: pd.DataFrame, out_path: Path):
    """Hình 5: Vẽ khung xương 13 khớp trực tiếp từ .npy & đường cong góc khớp Đúng vs Sai."""
    ext_csv = REPORTS_DIR / "pose_extraction_summary.csv"
    df_ext = pd.read_csv(ext_csv)

    sq_cor = _pick_clean_sample(df, df_ext, "squat", "correct")
    sq_inc = _pick_clean_sample(df, df_ext, "squat", "incorrect", "partial_rom")
    pu_cor = df[df["segment_id"] == "push_up_correct_016"].iloc[0]
    pu_inc = df[df["segment_id"] == "push_up_incorrect_022"].iloc[0]

    fig = plt.figure(figsize=(21, 13.5))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1.85], wspace=0.14)
    gs_imgs = outer_gs[0, 0].subgridspec(4, 5, hspace=0.28, wspace=0.08)
    gs_plots = outer_gs[0, 1].subgridspec(2, 1, hspace=0.44)

    fig.suptitle(
        "TRỰC QUAN KHUNG XƯƠNG 13 KHỚP (RAW_POINTS_13) & ĐỒ THỊ GÓC KHỚP 3D: MẪU ĐÚNG vs MẪU SAI",
        fontsize=15.5, fontweight="bold", y=0.98
    )

    # Hàng 0 & 1: Squat
    _draw_skeleton_strip([fig.add_subplot(gs_imgs[0, c]) for c in range(5)], sq_cor, (40, 190, 40), f"SQUAT ĐÚNG\n{sq_cor['segment_id']}")
    _draw_skeleton_strip([fig.add_subplot(gs_imgs[1, c]) for c in range(5)], sq_inc, (40, 40, 230), f"SQUAT SAI\n(partial_rom)\n{sq_inc['segment_id']}")

    ax_sq = fig.add_subplot(gs_plots[0, 0])
    for row, lbl, col in [(sq_cor, f"ĐÚNG ({sq_cor['segment_id']})", COLOR_CORRECT), (sq_inc, f"SAI - partial_rom ({sq_inc['segment_id']})", COLOR_ALERT)]:
        arr = np.load(POSE_NPY_DIR / f"{row['segment_id']}.npy")
        ang = _best_side_angle_series(arr[:, :, :3], arr[:, :, 5], (J_IDX["left_hip"], J_IDX["left_knee"], J_IDX["left_ankle"]), (J_IDX["right_hip"], J_IDX["right_knee"], J_IDX["right_ankle"]))
        x_pct = np.linspace(0, 100, len(ang))
        ax_sq.plot(x_pct, ang, linewidth=2.6, color=col, label=f"{lbl} | Min={np.nanmin(ang):.1f}°")
    ax_sq.axhline(100, color="gray", linestyle="--", label="Ngưỡng sâu chuẩn (~100°)")
    ax_sq.set_ylim(30, 180)
    ax_sq.set_xlabel("Tiến trình Rep (%)", fontsize=10)
    ax_sq.set_ylabel("Góc Đầu Gối 3D (độ)", fontsize=10)
    ax_sq.set_title("So sánh Góc Đầu Gối (Squat)", fontsize=12, fontweight="bold")
    ax_sq.grid(True, linestyle="--", alpha=0.4)
    ax_sq.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.8, frameon=True)

    # Hàng 2 & 3: Push-up
    _draw_skeleton_strip([fig.add_subplot(gs_imgs[2, c]) for c in range(5)], pu_cor, (40, 190, 40), f"PUSH-UP ĐÚNG\n{pu_cor['segment_id']}")
    _draw_skeleton_strip([fig.add_subplot(gs_imgs[3, c]) for c in range(5)], pu_inc, (40, 40, 230), f"PUSH-UP SAI\n(waist_sagging)\n{pu_inc['segment_id']}")

    ax_pu = fig.add_subplot(gs_plots[1, 0])
    for row, lbl, col in [(pu_cor, f"ĐÚNG ({pu_cor['segment_id']})", COLOR_CORRECT), (pu_inc, f"SAI - waist_sagging ({pu_inc['segment_id']})", COLOR_ALERT)]:
        arr = np.load(POSE_NPY_DIR / f"{row['segment_id']}.npy")
        ang = _best_side_angle_series(arr[:, :, :3], arr[:, :, 5], (J_IDX["left_shoulder"], J_IDX["left_hip"], J_IDX["left_ankle"]), (J_IDX["right_shoulder"], J_IDX["right_hip"], J_IDX["right_ankle"]))
        x_pct = np.linspace(0, 100, len(ang))
        ax_pu.plot(x_pct, ang, linewidth=2.6, color=col, label=f"{lbl} | Min={np.nanmin(ang):.1f}°")
    ax_pu.axhline(162, color="gray", linestyle="--", label="Ngưỡng thẳng thân Plank (~162°)")
    ax_pu.set_ylim(135, 183)
    ax_pu.set_xlabel("Tiến trình Rep (%)", fontsize=10)
    ax_pu.set_ylabel("Góc Thân Vai-Hông-Cổ chân (độ)", fontsize=10)
    ax_pu.set_title("So sánh Độ Thẳng Lưng Plank (Push-up)", fontsize=12, fontweight="bold")
    ax_pu.grid(True, linestyle="--", alpha=0.4)
    ax_pu.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.8, frameon=True)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TASK2_CSV_PATH, keep_default_na=False)
    df = df[df["exercise"].isin(GROUP_A_EXERCISES)].reset_index(drop=True)

    f1 = FIGURES_DIR / "fig1_before_after_supplementation.png"
    f2 = FIGURES_DIR / "fig2_source_and_group_distribution.png"
    f3 = FIGURES_DIR / "fig3_error_reasons_breakdown.png"
    f4 = FIGURES_DIR / "fig4_rep_duration_and_strided_sampling.png"
    f5 = FIGURES_DIR / "fig5_skeleton_13joints_correct_vs_incorrect.png"

    print("Đang vẽ 5 biểu đồ Bước 3 vào:", FIGURES_DIR)
    plot_fig1_before_after(df, f1)
    print("  [1/5] Xong:", f1.name)
    plot_fig2_source_and_groups(df, f2)
    print("  [2/5] Xong:", f2.name)
    plot_fig3_error_reasons(df, f3)
    print("  [3/5] Xong:", f3.name)
    plot_fig4_duration_and_sampling(df, f4)
    print("  [4/5] Xong:", f4.name)
    plot_fig5_skeleton_comparison(df, f5)
    print("  [5/5] Xong:", f5.name)
    print("HOÀN TẤT TOÀN BỘ BƯỚC B2 (TRỰC QUAN HÓA DỮ LIỆU)!")


if __name__ == "__main__":
    main()
