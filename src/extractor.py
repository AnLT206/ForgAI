# -*- coding: utf-8 -*-
"""
src/extractor.py
Trích xuất tọa độ khung xương 13 khớp cốt lõi (RAW_POINTS_13) bằng MediaPipe PoseLandmarker
cho toàn bộ 493 video trong metadata/task2_assessment.csv.

Mỗi file .npy lưu tại data_processed/pose/<segment_id>.npy có shape (L, 13, 6) kiểu float32:
  - Kênh 0, 1, 2 (wx, wy, wz): Tọa độ không gian 3D thực (mét) từ pose_world_landmarks,
    miễn nhiễm méo tỷ lệ khung hình giữa video ngang 16:9 và dọc 9:16 (YouTube Shorts).
  - Kênh 3, 4    (ix, iy)    : Tọa độ 2D trên khung hình [0..1] từ pose_landmarks,
    dùng để vẽ khung xương đè chính xác lên video và trực quan hóa.
  - Kênh 5       (visibility): Độ rõ của khớp [0.0 .. 1.0].
"""

import os
import sys
import io
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    TASK2_CSV_PATH,
    POSE_NPY_DIR,
    POSE_MODEL_PATH,
    REPORTS_DIR,
    GROUP_A_EXERCISES,
    NUM_JOINTS,
)
from src.constants import RAW_POINTS_13

if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Lưu bytes model toàn cục trên mỗi worker process để tránh đọc đĩa nhiều lần
_MODEL_BYTES_CACHE: bytes | None = None


def _get_model_bytes() -> bytes:
    global _MODEL_BYTES_CACHE
    if _MODEL_BYTES_CACHE is None:
        _MODEL_BYTES_CACHE = POSE_MODEL_PATH.read_bytes()
    return _MODEL_BYTES_CACHE


def extract_pose_sequence(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray, int, float]:
    """
    Trích xuất chuỗi khung xương 13 khớp từ 1 video clip bằng RunningMode.VIDEO.

    Returns:
        pose_seq: np.ndarray shape (L, 13, 6) [wx, wy, wz, ix, iy, visibility]
        det_frames: số khung hình phát hiện được người tập
        mean_vis: độ rõ trung bình của 13 khớp trên các frame phát hiện được
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=_get_model_bytes()),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=conf_threshold,
        min_pose_presence_confidence=conf_threshold,
        min_tracking_confidence=conf_threshold,
    )

    seq_rows = []
    det_frames = 0
    vis_scores = []
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if end_frame is not None and frame_idx > end_frame:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Đảm bảo timestamp tăng nghiêm ngặt cho RunningMode.VIDEO
            ts_ms = int((frame_idx - start_frame) * 1000.0 / fps) + 1
            res = landmarker.detect_for_video(mp_img, ts_ms)

            joints_mat = np.zeros((NUM_JOINTS, 6), dtype=np.float32)
            if res.pose_landmarks and res.pose_world_landmarks:
                pl = res.pose_landmarks[0]
                wl = res.pose_world_landmarks[0]
                for j_out, mp_idx in enumerate(RAW_POINTS_13):
                    wlm = wl[mp_idx]
                    ilm = pl[mp_idx]
                    vis = float(getattr(wlm, "visibility", getattr(ilm, "visibility", 1.0)) or 0.0)
                    joints_mat[j_out] = [
                        float(wlm.x),
                        float(wlm.y),
                        float(wlm.z),
                        float(ilm.x),
                        float(ilm.y),
                        vis,
                    ]
                det_frames += 1
                vis_scores.append(float(joints_mat[:, 5].mean()))

            seq_rows.append(joints_mat)
            frame_idx += 1

    cap.release()
    if len(seq_rows) == 0:
        raise RuntimeError(f"Video không đọc được khung hình nào: {video_path}")

    pose_seq = np.stack(seq_rows, axis=0).astype(np.float32)
    mean_vis = float(np.mean(vis_scores)) if vis_scores else 0.0
    return pose_seq, det_frames, mean_vis


def _process_single_row(task: dict) -> dict:
    """Worker xử lý 1 dòng từ metadata."""
    seg_id = task["segment_id"]
    out_path = POSE_NPY_DIR / f"{seg_id}.npy"
    video_abs = PROJECT_ROOT / task["video_path"]
    sf = int(task["start_frame"])
    ef = int(task["end_frame"])
    expected_len = int(task["num_frames"])

    if out_path.exists() and not task["overwrite"]:
        arr = np.load(out_path)
        if arr.ndim == 3 and arr.shape[1:] == (NUM_JOINTS, 6) and len(arr) > 0 and not np.isnan(arr).any():
            det_mask = np.abs(arr[:, :, :3]).sum(axis=(1, 2)) > 0
            det_frames = int(det_mask.sum())
            mean_vis = float(arr[det_mask, :, 5].mean()) if det_frames > 0 else 0.0
            return {
                "segment_id": seg_id,
                "exercise": task["exercise"],
                "form_label": task["form_label"],
                "num_frames_npy": int(len(arr)),
                "expected_frames": expected_len,
                "det_frames": det_frames,
                "det_rate": round(det_frames / len(arr), 4),
                "mean_visibility": round(mean_vis, 4),
                "file_kb": round(out_path.stat().st_size / 1024.0, 2),
                "status": "cached",
            }

    pose_seq, det_frames, mean_vis = extract_pose_sequence(
        video_abs, start_frame=sf, end_frame=ef, conf_threshold=task["conf"]
    )
    if np.isnan(pose_seq).any() or np.isinf(pose_seq).any():
        raise ValueError(f"Phát hiện NaN/Inf trong {seg_id}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pose_seq)

    return {
        "segment_id": seg_id,
        "exercise": task["exercise"],
        "form_label": task["form_label"],
        "num_frames_npy": int(len(pose_seq)),
        "expected_frames": expected_len,
        "det_frames": det_frames,
        "det_rate": round(det_frames / len(pose_seq), 4),
        "mean_visibility": round(mean_vis, 4),
        "file_kb": round(out_path.stat().st_size / 1024.0, 2),
        "status": "extracted",
    }


def run_extraction(workers: int = 4, overwrite: bool = False, conf: float = 0.25, limit: int | None = None):
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Thiếu file model MediaPipe tại: {POSE_MODEL_PATH}")

    df = pd.read_csv(TASK2_CSV_PATH, keep_default_na=False)
    df = df[df["exercise"].isin(GROUP_A_EXERCISES)].reset_index(drop=True)
    if limit is not None:
        df = df.head(limit).copy()

    POSE_NPY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "segment_id": r["segment_id"],
            "exercise": r["exercise"],
            "form_label": r["form_label"],
            "video_path": r["video_path"],
            "start_frame": int(r["start_frame"]),
            "end_frame": int(r["end_frame"]),
            "num_frames": int(r["num_frames"]),
            "overwrite": overwrite,
            "conf": conf,
        }
        for _, r in df.iterrows()
    ]

    t0 = time.time()
    print("=" * 75)
    print(f"FORGAI STEP B1 - TRÍCH XUẤT KHUNG XƯƠNG 13 KHỚP (.NPY)")
    print(f"Số video: {len(tasks)} | Workers: {workers} | Confidence: {conf} | Overwrite: {overwrite}")
    print("=" * 75)

    results = []
    if workers <= 1:
        for idx, t in enumerate(tasks, 1):
            res = _process_single_row(t)
            results.append(res)
            if idx % 25 == 0 or idx == len(tasks):
                print(f"  Tiến độ: {idx}/{len(tasks)} ({time.time() - t0:.1f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_single_row, t): t["segment_id"] for t in tasks}
            for idx, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                results.append(res)
                if idx % 25 == 0 or idx == len(tasks):
                    print(f"  Tiến độ: {idx}/{len(tasks)} ({time.time() - t0:.1f}s)")

    df_rep = pd.DataFrame(results).sort_values("segment_id").reset_index(drop=True)
    report_csv = REPORTS_DIR / "pose_extraction_summary.csv"
    df_rep.to_csv(report_csv, index=False, encoding="utf-8-sig")

    total_mb = df_rep["file_kb"].sum() / 1024.0
    avg_det = df_rep["det_rate"].mean() * 100.0
    avg_vis = df_rep["mean_visibility"].mean()
    low_det = df_rep[df_rep["det_rate"] < 0.70]

    print("\n" + "=" * 75)
    print(f"HOÀN TẤT TRÍCH XUẤT ({time.time() - t0:.1f}s):")
    print(f"  - Tổng file .npy hợp lệ : {len(df_rep)}/{len(tasks)}")
    print(f"  - Tổng dung lượng       : {total_mb:.2f} MB (tại {POSE_NPY_DIR})")
    print(f"  - Tỷ lệ bắt khớp TB     : {avg_det:.2f}% | Độ rõ khớp TB: {avg_vis:.3f}")
    print(f"  - Báo cáo chi tiết lưu  : {report_csv}")
    print("\nThống kê theo từng bài tập:")
    ex_stats = (
        df_rep.groupby("exercise")
        .agg(
            samples=("segment_id", "count"),
            avg_det_rate=("det_rate", lambda s: round(s.mean() * 100, 1)),
            min_det_rate=("det_rate", lambda s: round(s.min() * 100, 1)),
            avg_vis=("mean_visibility", lambda s: round(s.mean(), 3)),
        )
        .reset_index()
    )
    print(ex_stats.to_string(index=False))

    if len(low_det) > 0:
        print(f"\n[CẢNH BÁO] Có {len(low_det)} clip có tỷ lệ bắt khớp < 70%:")
        print(low_det[["segment_id", "det_frames", "num_frames_npy", "det_rate", "mean_visibility"]].to_string(index=False))
    else:
        print("\n[XUẤT SẮC] 100% clip đều đạt tỷ lệ bắt khớp >= 70%!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất khung xương MediaPipe 13 khớp ra file .npy")
    parser.add_argument("--workers", type=int, default=4, help="Số tiến trình song song CPU")
    parser.add_argument("--overwrite", action="store_true", help="Trích xuất lại dù file .npy đã tồn tại")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence của MediaPipe Pose")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số clip chạy thử")
    args = parser.parse_args()

    run_extraction(workers=args.workers, overwrite=args.overwrite, conf=args.conf, limit=args.limit)
