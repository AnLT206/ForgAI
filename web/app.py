# -*- coding: utf-8 -*-
"""
web/app.py
Giao diện Web Cục bộ cho Bước 7 (ForgAI Web Studio):
  1. Tab Phân tích Video (Kéo & Thả): hỗ trợ cả Video 1 Rep (chấm sâu 4 nhánh AI + biểu đồ góc)
     và Video Nhiều Rep / Cả Hiệp (CycleRepCounter đếm & chấm từng rep). Tự chuyển mã H.264.
  2. Tab Camera Trực Tiếp (Realtime 30 FPS): stream MJPEG + nút chọn 8 bài, lật gương, reset rep.
  3. Tab Tổng quan Mô hình & Độ trễ: hiển thị báo cáo 5-Fold CV (E.3-LowRisk), benchmark CPU & biểu đồ.

Chạy nhanh: python web/app.py (hoặc python -m web.app) — tự mở trình duyệt tại http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    GROUP_A_EXERCISES,
    N_SPLITS,
    OUTPUTS_DIR,
    REPORTS_DIR,
)
from src.constants import J_IDX
from src.inference import ForgAITask2Predictor
from src.realtime import (
    CycleRepCounter,
    MediaPipeRealtimeTracker,
    REP_CYCLE_SPECS,
    RepCompletedEvent,
    compute_driver_signal_with_vis,
)
from tools.demo_inference import _JOINT_SHORT_VN, export_annotated_video
from tools.realtime_demo import (
    _draw_realtime_hud,
    _draw_skeleton,
    _load_fonts as _load_rt_fonts,
    _open_capture,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
WEB_SESSIONS_DIR = OUTPUTS_DIR / "web_sessions"
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024  # 250 MB
MAX_KEPT_SESSIONS = 25

# Thông tin thân thiện tiếng Việt cho 8 bài tập Nhóm A
EXERCISE_UI_META: dict[str, dict[str, str]] = {
    "squat": {"vi": "Squat (Gánh đùi / Ngồi xổm)", "driver_vi": "Góc đầu gối"},
    "push-up": {"vi": "Push-up (Chống đẩy)", "driver_vi": "Góc khuỷu tay"},
    "pull up": {"vi": "Pull up (Kéo xà đơn)", "driver_vi": "Góc khuỷu tay"},
    "leg extension": {"vi": "Leg Extension (Đá đùi trước)", "driver_vi": "Góc đầu gối"},
    "chest fly machine": {"vi": "Chest Fly Machine (Ép ngực máy)", "driver_vi": "Khoảng cách hai cổ tay"},
    "hip thrust": {"vi": "Hip Thrust (Đẩy hông tạ đòn)", "driver_vi": "Góc khớp hông"},
    "t bar row": {"vi": "T Bar Row (Kéo lưng chữ T)", "driver_vi": "Góc khuỷu tay"},
    "leg raises": {"vi": "Leg Raises (Nâng chân tập bụng)", "driver_vi": "Góc khớp hông"},
}

FIGURE_CAPTIONS: dict[str, str] = {
    "eval_fig1_training_curves.png": "Đánh giá 1: Đường cong huấn luyện & Composite Score trên Validation (5-Fold CV)",
    "eval_fig2_model_comparison_cv.png": "Đánh giá 2: So sánh hiệu năng 5-Fold CV (BiomechanicalRF 106-d E.3+, DL & Hybrid)",
    "eval_fig3_confusion_and_per_exercise.png": "Đánh giá 3: Ma trận nhầm lẫn Pooled & Macro-F1 theo 8 bài tập Nhóm A",
    "eval_fig4_attention_vs_biomechanics.png": "Đánh giá 4: Đối chiếu Temporal Attention với đường cong góc khớp cơ sinh học",
    "eval_fig5_explainability_and_features.png": "Đánh giá 5: Top-15 đặc trưng quan trọng & độ phủ chẩn đoán Rule Engine",
    "fig1_before_after_supplementation.png": "Dữ liệu 1: Đối chiếu Trước/Sau bổ sung YouTube & Augmentation x6",
    "fig2_source_and_group_distribution.png": "Dữ liệu 2: Phân bố 3 nguồn dữ liệu (EFA, GYMED, YouTube) & các nhóm video_id",
    "fig3_error_reasons_breakdown.png": "Dữ liệu 3: Thống kê chi tiết 32 mã lỗi cơ sinh học Nhóm A",
    "fig4_rep_duration_and_strided_sampling.png": "Dữ liệu 4: Phân bố độ dài Rep & minh họa lấy mẫu Dynamic Strided T=32",
    "fig5_skeleton_13joints_correct_vs_incorrect.png": "Dữ liệu 5: Trực quan hóa khung xương 13 khớp Đúng vs Sai",
}

_SESSION_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{4,64}$")
_SAFE_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")


def _cleanup_old_sessions(keep_latest: int = MAX_KEPT_SESSIONS) -> None:
    """Giữ thư mục tạm web_sessions gọn nhẹ, tránh tích tụ dung lượng ổ đĩa."""
    if not WEB_SESSIONS_DIR.exists():
        return
    dirs = [p for p in WEB_SESSIONS_DIR.iterdir() if p.is_dir()]
    if len(dirs) <= keep_latest:
        return
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for old_dir in dirs[keep_latest:]:
        shutil.rmtree(old_dir, ignore_errors=True)


def transcode_to_h264(
    raw_mp4: Path,
    final_mp4: Path,
    *,
    preset: str = "fast",
    crf: int = 23,
    cleanup_src: bool = True,
    allow_fallback_copy: bool = True,
) -> Path:
    """Chuyển mã video (MP4/MOV/HEVC/AVI) sang H.264 yuv420p để Chrome/Edge và OpenCV đọc chuẩn."""
    final_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(raw_mp4),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(final_mp4),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not final_mp4.exists() or final_mp4.stat().st_size == 0:
        if allow_fallback_copy:
            shutil.copy2(raw_mp4, final_mp4)
        else:
            err_tail = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()[-240:]
            raise ValueError(f"Không thể chuyển mã video sang H.264 ({err_tail or 'codec không hợp lệ'}).")
    if cleanup_src:
        try:
            if raw_mp4.exists() and raw_mp4 != final_mp4:
                raw_mp4.unlink()
        except OSError:
            pass
    return final_mp4


def _probe_video_meta(video_path: Path) -> tuple[float, int, int, int]:
    """Kiểm tra nhanh video đọc được bằng OpenCV hay không, trả (fps, n_frames, w, h)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 30.0, 0, 0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return fps, 0, 0, 0
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return fps, 0, 0, 0
    return fps, max(n_frames, 1), int(w), int(h)


def _prepare_cv2_input(src_path: Path, norm_mp4: Path) -> Path:
    """Chuẩn hóa .mov/.avi/.mkv (nhất là HEVC iPhone xoay chiều) sang MP4 H.264 trước khi phân tích."""
    ext = src_path.suffix.lower()
    if ext in {".mov", ".avi", ".mkv"}:
        return transcode_to_h264(
            src_path, norm_mp4, preset="veryfast", crf=18, cleanup_src=False, allow_fallback_copy=False
        )
    _, n_frames, w, h = _probe_video_meta(src_path)
    if n_frames > 0 and w > 0 and h > 0:
        return src_path
    return transcode_to_h264(
        src_path, norm_mp4, preset="veryfast", crf=18, cleanup_src=False, allow_fallback_copy=False
    )


async def _save_upload_to_disk(file: UploadFile, dst_dir: Path) -> tuple[Path, str]:
    """Lưu UploadFile ra thư mục job tạm kèm kiểm tra định dạng & giới hạn 250 MB."""
    orig_name = Path(file.filename or "upload.mp4").name
    ext = Path(orig_name).suffix.lower() or ".mp4"
    if ext not in ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng video '{ext}' chưa hỗ trợ. Hãy tải file .mp4, .mov, .avi, .mkv hoặc .webm.",
        )

    dst_dir.mkdir(parents=True, exist_ok=True)
    in_path = dst_dir / f"input{ext}"
    total_written = 0
    try:
        with in_path.open("wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File video vượt quá giới hạn 250 MB.")
                out_f.write(chunk)
    finally:
        await file.close()

    if total_written == 0:
        raise HTTPException(status_code=400, detail="File video tải lên rỗng (0 byte).")
    return in_path, orig_name


class SharedModelManager:
    """Nạp lười và giữ duy nhất 1 bộ trọng số trong RAM cho toàn bộ Web Studio."""

    def __init__(self, default_fold: int = 2):
        self.default_fold = int(default_fold)
        self._lock = threading.Lock()
        self._predictor: ForgAITask2Predictor | None = None

    def get_predictor(self, all_folds: bool = False) -> ForgAITask2Predictor:
        with self._lock:
            if self._predictor is None:
                self._predictor = ForgAITask2Predictor(
                    fold=self.default_fold,
                    ensemble_all_folds=bool(all_folds),
                )
            elif all_folds and len(self._predictor.runtimes) < N_SPLITS:
                from src.inference import _FoldRuntime

                for k in range(N_SPLITS):
                    if k not in self._predictor.runtimes:
                        self._predictor.runtimes[k] = _FoldRuntime(
                            k, self._predictor.checkpoints_dir, self._predictor.device
                        )
                self._predictor.ensemble_all_folds = True
            return self._predictor


MODEL_MANAGER = SharedModelManager(default_fold=2)


def _format_joint_names(joint_indices: list[int]) -> list[str]:
    return [_JOINT_SHORT_VN[j] for j in joint_indices if j in _JOINT_SHORT_VN]


def _serialize_single_result(res: dict[str, Any], video_url: str | None = None) -> dict[str, Any]:
    """Đóng gói kết quả suy luận 1 rep sạch JSON (bỏ các trường thô không cần thiết)."""
    return {
        "mode": "single_rep",
        "exercise": res["exercise"],
        "exercise_vi": EXERCISE_UI_META.get(res["exercise"], {}).get("vi", res["exercise"]),
        "folds_used": res["folds_used"],
        "num_frames": res["num_frames"],
        "fps": res["fps"],
        "duration_sec": res["duration_sec"],
        "det_rate": res["det_rate"],
        "predicted_label": res["predicted_label"],
        "label_vi": res["label_vi"],
        "is_incorrect": bool(res["is_incorrect"]),
        "form_score_pct": res["form_score_pct"],
        "threshold": res["threshold"],
        "branches": {
            "rf": res["p_rf"],
            "cnn_bilstm": res["p_cnn_bilstm"],
            "transformer": res["p_transformer"],
            "hybrid": res["p_hybrid"],
        },
        "predicted_error_code": res["predicted_error_code"],
        "vietnamese_error_name": res["vietnamese_error_name"],
        "measured_evidence": res["measured_evidence"],
        "coaching_advice": res["coaching_advice"],
        "highlight_joints_names": _format_joint_names(res.get("highlight_joints", [])),
        "top3_candidates": res.get("top3_candidates", []),
        "driver_key": res.get("driver_key", ""),
        "driver_curve": res.get("driver_curve", []),
        "timings_ms": res.get("timings_ms", {}),
        "video_url": video_url,
    }


def _to_rt_hud_payload(rep_res: dict[str, Any], rep_index: int, observed_rom: float) -> dict[str, Any]:
    """Chuyển đổi output của ForgAITask2Predictor sang định dạng cho _draw_realtime_hud."""
    return {
        "rep_index": rep_index,
        "exercise": rep_res["exercise"],
        "duration_sec": rep_res["duration_sec"],
        "is_incorrect": bool(rep_res["is_incorrect"]),
        "label_vi": rep_res["label_vi"],
        "p_incorrect": float(rep_res["p_hybrid"]),
        "infer_ms": float(rep_res["timings_ms"]["total_ai_ms"]),
        "vietnamese_error_name": rep_res["vietnamese_error_name"],
        "measured_evidence": rep_res["measured_evidence"],
        "coaching_advice": rep_res["coaching_advice"],
        "observed_rom": round(float(observed_rom), 2),
    }


def analyze_multi_rep_video(
    video_path: Path,
    exercise: str,
    raw_hud_out: Path,
    predictor: ForgAITask2Predictor,
    fold_override: int | None,
) -> dict[str, Any]:
    """Xử lý video chứa nhiều rep liên tiếp: đếm chu kỳ rep + chấm điểm từng rep + xuất video HUD."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở file video: {video_path.name}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if src_fps <= 1.0 or src_fps > 120.0:
        src_fps = 30.0

    fonts = _load_rt_fonts()
    counter = CycleRepCounter(exercise=exercise)
    writer: cv2.VideoWriter | None = None
    frame_idx = 0
    det_frames = 0
    correct_count = 0
    last_hud_res: dict[str, Any] | None = None
    rep_records: list[dict[str, Any]] = []
    rep_start_frames: dict[int, int] = {}

    t0 = time.perf_counter()

    def _finalize_rep(event: RepCompletedEvent, end_frame_idx: int) -> None:
        nonlocal correct_count, last_hud_res
        rep_res = predictor.predict_from_pose(
            pose_seq=event.pose_seq,
            exercise_name=event.exercise,
            duration_sec=event.duration_sec,
            fps=src_fps,
            fold_override=fold_override,
        )
        last_hud_res = _to_rt_hud_payload(rep_res, event.rep_index, event.observed_rom)
        if not rep_res["is_incorrect"]:
            correct_count += 1
        start_f = max(1, end_frame_idx - int(event.pose_seq.shape[0]) + 1)
        rep_records.append(
            {
                "rep_index": event.rep_index,
                "start_sec": round((start_f - 1) / src_fps, 2),
                "end_sec": round(end_frame_idx / src_fps, 2),
                "duration_sec": rep_res["duration_sec"],
                "observed_rom": round(float(event.observed_rom), 1),
                "is_incorrect": bool(rep_res["is_incorrect"]),
                "label_vi": rep_res["label_vi"],
                "form_score_pct": rep_res["form_score_pct"],
                "p_hybrid": rep_res["p_hybrid"],
                "branches": {
                    "rf": rep_res["p_rf"],
                    "cnn_bilstm": rep_res["p_cnn_bilstm"],
                    "transformer": rep_res["p_transformer"],
                    "hybrid": rep_res["p_hybrid"],
                },
                "predicted_error_code": rep_res["predicted_error_code"],
                "vietnamese_error_name": rep_res["vietnamese_error_name"],
                "measured_evidence": rep_res["measured_evidence"],
                "coaching_advice": rep_res["coaching_advice"],
                "highlight_joints_names": _format_joint_names(rep_res.get("highlight_joints", [])),
                "top3_candidates": rep_res.get("top3_candidates", []),
            }
        )

    try:
        with MediaPipeRealtimeTracker(conf_threshold=0.25) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                if frame.shape[1] > 720:
                    new_h = int(round(frame.shape[0] * (720.0 / frame.shape[1])))
                    frame = cv2.resize(frame, (720, max(240, new_h)))
                elif frame.shape[0] < 480:
                    scale = 480.0 / float(frame.shape[0])
                    frame = cv2.resize(frame, (max(320, int(round(frame.shape[1] * scale))), 480))

                ts_ms = int(frame_idx * 1000.0 / src_fps)
                ts_sec = ts_ms / 1000.0
                joints, detected = tracker.process_frame(frame, ts_ms=ts_ms)
                if detected:
                    det_frames += 1

                drv_raw, drv_vis = (
                    (0.0, False) if not detected else compute_driver_signal_with_vis(joints, counter.exercise)
                )
                drv_val = drv_raw if drv_vis else counter.ema_val

                prev_state = counter.state
                event = counter.update(joints, detected=detected, timestamp_sec=ts_sec)
                if prev_state != "IN_REP" and counter.state == "IN_REP":
                    rep_start_frames[counter.rep_count + 1] = frame_idx
                if event is not None:
                    _finalize_rep(event, frame_idx)

                if counter.state == "IN_REP":
                    skel_col = (0, 200, 255)
                elif last_hud_res is not None:
                    skel_col = (70, 70, 255) if last_hud_res["is_incorrect"] else (70, 230, 90)
                else:
                    skel_col = (220, 220, 90)

                disp = frame.copy()
                _draw_skeleton(disp, joints, detected, skel_col, mirror_display=False)

                elapsed = max(1e-4, time.perf_counter() - t0)
                proc_ms_avg = (elapsed * 1000.0) / frame_idx
                fps_avg = frame_idx / elapsed

                disp_hud = _draw_realtime_hud(
                    disp,
                    fonts=fonts,
                    exercise=counter.exercise,
                    counter=counter,
                    correct_count=correct_count,
                    last_result=last_hud_res,
                    scoring_pending=False,
                    fps_avg=fps_avg,
                    proc_ms_avg=proc_ms_avg,
                    driver_val=drv_val,
                )
                if writer is None:
                    raw_hud_out.parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(raw_hud_out), fourcc, src_fps, (disp_hud.shape[1], disp_hud.shape[0])
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Không thể khởi tạo VideoWriter cho {raw_hud_out}")
                writer.write(disp_hud)

            tail_event = counter.flush_if_valid()
            if tail_event is not None:
                _finalize_rep(tail_event, frame_idx)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_idx == 0:
        raise ValueError("Video rỗng hoặc không đọc được khung hình nào.")

    total_ms = (time.perf_counter() - t0) * 1000.0
    det_rate = det_frames / max(1, frame_idx)
    if det_rate < 0.05:
        raise ValueError(
            "Không phát hiện được khung xương người tập trong video (tỷ lệ < 5%). "
            "Vui lòng chọn video quay đủ sáng và thấy rõ khớp chính."
        )

    avg_score = (
        round(float(np.mean([r["form_score_pct"] for r in rep_records])), 1)
        if rep_records
        else None
    )
    return {
        "mode": "multi_rep",
        "exercise": exercise,
        "exercise_vi": EXERCISE_UI_META.get(exercise, {}).get("vi", exercise),
        "folds_used": [0, 1, 2, 3, 4] if fold_override is None else [fold_override],
        "num_frames": frame_idx,
        "fps": round(src_fps, 2),
        "duration_sec": round(frame_idx / src_fps, 2),
        "det_rate": round(det_rate, 4),
        "total_reps": len(rep_records),
        "correct_reps": correct_count,
        "incorrect_reps": len(rep_records) - correct_count,
        "avg_form_score_pct": avg_score,
        "timings_ms": {
            "end_to_end_total_ms": round(total_ms, 1),
            "end_to_end_per_frame_ms": round(total_ms / max(1, frame_idx), 2),
        },
        "reps": rep_records,
    }


def _prepare_webcam_frames(
    raw_frame: np.ndarray, mp_width: int = 640, disp_width: int = 960
) -> tuple[np.ndarray, np.ndarray]:
    """Tạo bản thu gọn cạnh dài tối đa 640px cho MediaPipe và bản hiển thị cạnh dài 960px đúng tỷ lệ."""
    h, w = raw_frame.shape[:2]
    if w <= 0 or h <= 0:
        return raw_frame, raw_frame

    long_side = max(w, h)
    if long_side > mp_width:
        scale = float(mp_width) / long_side
        mp_w = max(240, int(round(w * scale)))
        mp_h = max(240, int(round(h * scale)))
        mp_frame = cv2.resize(raw_frame, (mp_w, mp_h), interpolation=cv2.INTER_AREA)
    else:
        mp_frame = raw_frame

    if long_side != disp_width:
        scale = float(disp_width) / long_side
        disp_w = max(360, int(round(w * scale)))
        disp_h = max(360, int(round(h * scale)))
        interp = cv2.INTER_AREA if long_side > disp_width else cv2.INTER_LINEAR
        disp_base = cv2.resize(raw_frame, (disp_w, disp_h), interpolation=interp)
    else:
        disp_base = raw_frame
    return mp_frame, disp_base


class WebcamSessionManager:
    """Quản lý luồng Camera Realtime 30 FPS chạy nền an toàn trên Web (MJPEG + JSON Telemetry)."""

    def __init__(self, model_manager: SharedModelManager):
        self.model_manager = model_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._active = False
        self._camera_index = 0
        self._exercise = "squat"
        self._target_reps = 10
        self._use_all_folds = False
        self._mirror = True
        self._reset_requested = False
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._status: dict[str, Any] = self._empty_status()

    def _empty_status(self) -> dict[str, Any]:
        return {
            "active": False,
            "camera_index": 0,
            "exercise": "squat",
            "exercise_vi": EXERCISE_UI_META["squat"]["vi"],
            "target_reps": 10,
            "use_all_folds": False,
            "mirror": True,
            "state": "IDLE",
            "completed": False,
            "tracking_lost": False,
            "stabilizing": False,
            "waiting_ready": False,
            "display_width": 960,
            "display_height": 540,
            "progress": 0.0,
            "rep_count": 0,
            "correct_count": 0,
            "driver_key": REP_CYCLE_SPECS["squat"]["driver"],
            "driver_unit": REP_CYCLE_SPECS["squat"]["unit"],
            "driver_val": None,
            "fps": 0.0,
            "proc_ms": 0.0,
            "last_result": None,
            "recent_reps": [],
            "summary": None,
            "error": None,
        }

    @staticmethod
    def _build_summary(recent_reps: list[dict[str, Any]], correct_count: int, target_reps: int) -> dict[str, Any] | None:
        total = len(recent_reps)
        if total == 0:
            return None
        scores = [float(r["form_score_pct"]) for r in recent_reps if r.get("form_score_pct") is not None]
        avg_score = round(float(np.mean(scores)), 1) if scores else None
        return {
            "total_reps": total,
            "correct_count": int(correct_count),
            "incorrect_count": max(0, total - int(correct_count)),
            "target_reps": int(target_reps),
            "target_reached": bool(target_reps > 0 and total >= target_reps),
            "avg_form_score_pct": avg_score,
        }

    def start(
        self,
        exercise: str = "squat",
        target_reps: int = 10,
        use_all_folds: bool = False,
        camera_index: int = 0,
        mirror: bool = True,
    ) -> dict[str, Any]:
        ex = exercise.strip().lower()
        if ex not in GROUP_A_EXERCISES:
            raise ValueError(f"Bài tập không hợp lệ: {exercise}")
        t_reps = int(target_reps)
        if t_reps < 0 or t_reps > 50:
            raise ValueError("Số rep mục tiêu phải từ 0 (Tập tự do) đến 50.")

        self.stop()
        with self._lock:
            self._stop_event.clear()
            self._exercise = ex
            self._target_reps = t_reps
            self._use_all_folds = bool(use_all_folds)
            self._camera_index = int(camera_index)
            self._mirror = bool(mirror)
            self._reset_requested = False
            self._latest_jpeg = None
            self._frame_seq = 0
            self._status = self._empty_status()
            self._status.update(
                {
                    "active": True,
                    "camera_index": self._camera_index,
                    "exercise": ex,
                    "exercise_vi": EXERCISE_UI_META[ex]["vi"],
                    "target_reps": self._target_reps,
                    "use_all_folds": self._use_all_folds,
                    "mirror": self._mirror,
                    "driver_key": REP_CYCLE_SPECS[ex]["driver"],
                    "driver_unit": REP_CYCLE_SPECS[ex]["unit"],
                }
            )
            self._active = True
            self._thread = threading.Thread(target=self._run_loop, name="ForgAI-Webcam", daemon=True)
            self._thread.start()
        return self.get_status()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=3.0)
        with self._lock:
            self._active = False
            self._thread = None
            self._status["active"] = False
            if self._status.get("state") != "COMPLETED":
                self._status["state"] = "IDLE"
        return self.get_status()

    def update_control(
        self,
        exercise: str | None = None,
        mirror: bool | None = None,
        reset: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            if exercise is not None:
                ex = exercise.strip().lower()
                if ex not in GROUP_A_EXERCISES:
                    raise ValueError(f"Bài tập không hợp lệ: {exercise}")
                if ex != self._exercise:
                    self._exercise = ex
                    self._reset_requested = True
                    self._status["exercise"] = ex
                    self._status["exercise_vi"] = EXERCISE_UI_META[ex]["vi"]
                    self._status["driver_key"] = REP_CYCLE_SPECS[ex]["driver"]
                    self._status["driver_unit"] = REP_CYCLE_SPECS[ex]["unit"]
            if mirror is not None:
                self._mirror = bool(mirror)
                self._status["mirror"] = self._mirror
            if reset:
                self._reset_requested = True
        return self.get_status()

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._status, ensure_ascii=False))

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_latest_jpeg_with_seq(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest_jpeg, self._frame_seq

    def _run_loop(self) -> None:
        cap = None
        correct_count = 0
        recent_reps: list[dict[str, Any]] = []
        target_reps = 0
        try:
            with self._lock:
                cam_idx = self._camera_index
                cur_ex = self._exercise
                target_reps = self._target_reps
                use_all_folds = self._use_all_folds

            predictor = self.model_manager.get_predictor(all_folds=use_all_folds)
            fold_override = None if use_all_folds else self.model_manager.default_fold

            cap, _ = _open_capture(str(cam_idx), width=960, height=540)
            if not cap.isOpened():
                with self._lock:
                    self._status["active"] = False
                    self._status["state"] = "IDLE"
                    self._status["error"] = f"Không mở được Webcam #{cam_idx}. Hãy kiểm tra kết nối camera."
                return

            counter = CycleRepCounter(exercise=cur_ex, require_ready_pose=True)
            last_hud_res: dict[str, Any] | None = None
            pending_future: Future | None = None
            pending_event: RepCompletedEvent | None = None
            frame_idx = 0
            goal_reached = False
            loop_fps_ema = 30.0
            proc_ms_ema = 0.0
            last_loop_t = time.perf_counter()

            def _score_rep_sync(ev: RepCompletedEvent) -> dict[str, Any]:
                return predictor.predict_from_pose(
                    pose_seq=ev.pose_seq,
                    exercise_name=ev.exercise,
                    duration_sec=ev.duration_sec,
                    fold_override=fold_override,
                )

            def _collect_future(block: bool = False) -> None:
                nonlocal pending_future, pending_event, last_hud_res, correct_count
                if pending_future is None:
                    return
                if not block and not pending_future.done():
                    return
                rep_res = pending_future.result()
                if pending_event is not None:
                    hud_pay = _to_rt_hud_payload(rep_res, pending_event.rep_index, pending_event.observed_rom)
                    last_hud_res = hud_pay
                    if not rep_res["is_incorrect"]:
                        correct_count += 1
                    recent_reps.insert(
                        0,
                        {
                            "rep_index": pending_event.rep_index,
                            "exercise": rep_res["exercise"],
                            "duration_sec": rep_res["duration_sec"],
                            "is_incorrect": bool(rep_res["is_incorrect"]),
                            "label_vi": rep_res["label_vi"],
                            "form_score_pct": rep_res["form_score_pct"],
                            "p_hybrid": rep_res["p_hybrid"],
                            "vietnamese_error_name": rep_res["vietnamese_error_name"],
                            "measured_evidence": rep_res["measured_evidence"],
                            "coaching_advice": rep_res["coaching_advice"],
                        },
                    )
                    del recent_reps[50:]
                pending_future = None
                pending_event = None

            with MediaPipeRealtimeTracker(conf_threshold=0.25) as tracker, ThreadPoolExecutor(max_workers=1) as pool:
                while not self._stop_event.is_set():
                    with self._lock:
                        req_ex = self._exercise
                        mirror_display = self._mirror
                        do_reset = self._reset_requested
                        self._reset_requested = False

                    if do_reset or req_ex != counter.exercise:
                        _collect_future(block=True)
                        counter.reset(req_ex)
                        correct_count = 0
                        last_hud_res = None
                        recent_reps.clear()
                        with self._lock:
                            self._status.update(
                                {
                                    "completed": False,
                                    "rep_count": 0,
                                    "correct_count": 0,
                                    "last_result": None,
                                    "recent_reps": [],
                                    "summary": None,
                                }
                            )

                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.02)
                        continue
                    frame_idx += 1

                    mp_frame, disp_base = _prepare_webcam_frames(frame, mp_width=640, disp_width=960)

                    t_p0 = time.perf_counter()
                    ts_ms = int(time.perf_counter() * 1000.0)
                    joints, detected = tracker.process_frame(mp_frame, ts_ms=ts_ms)
                    drv_raw, drv_vis = (
                        (0.0, False) if not detected else compute_driver_signal_with_vis(joints, counter.exercise)
                    )
                    drv_val = drv_raw if drv_vis else counter.ema_val

                    event = counter.update(joints, detected=detected, timestamp_sec=ts_ms / 1000.0)
                    _collect_future(block=False)
                    if event is not None:
                        if pending_future is not None:
                            _collect_future(block=True)
                        pending_event = event
                        pending_future = pool.submit(_score_rep_sync, event)

                    goal_reached = bool(target_reps > 0 and counter.rep_count >= target_reps)
                    if goal_reached:
                        _collect_future(block=True)

                    disp = cv2.flip(disp_base, 1) if mirror_display else disp_base.copy()
                    if counter.state == "IN_REP":
                        skel_col = (0, 200, 255)
                    elif last_hud_res is not None:
                        skel_col = (70, 70, 255) if last_hud_res["is_incorrect"] else (70, 230, 90)
                    else:
                        skel_col = (220, 220, 90)
                    _draw_skeleton(disp, joints, detected, skel_col, mirror_display=mirror_display)

                    proc_ms = (time.perf_counter() - t_p0) * 1000.0
                    proc_ms_ema = proc_ms if frame_idx == 1 else (0.15 * proc_ms + 0.85 * proc_ms_ema)
                    now_t = time.perf_counter()
                    dt = max(1e-4, now_t - last_loop_t)
                    last_loop_t = now_t
                    inst_fps = min(60.0, 1.0 / dt)
                    loop_fps_ema = inst_fps if frame_idx == 1 else (0.15 * inst_fps + 0.85 * loop_fps_ema)

                    ok_enc, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok_enc:
                        with self._lock:
                            self._latest_jpeg = buf.tobytes()
                            self._frame_seq += 1
                            self._status.update(
                                {
                                    "active": not goal_reached,
                                    "state": "COMPLETED" if goal_reached else counter.state,
                                    "completed": goal_reached,
                                    "tracking_lost": bool(counter.is_tracking_lost),
                                    "stabilizing": bool(counter.is_stabilizing),
                                    "waiting_ready": bool(counter.is_waiting_ready),
                                    "display_width": int(disp.shape[1]),
                                    "display_height": int(disp.shape[0]),
                                    "progress": 1.0 if goal_reached else round(float(counter.last_progress), 3),
                                    "rep_count": int(counter.rep_count),
                                    "correct_count": int(correct_count),
                                    "driver_val": round(float(drv_val), 1) if drv_val is not None else None,
                                    "fps": round(float(loop_fps_ema), 1),
                                    "proc_ms": round(float(proc_ms_ema), 1),
                                    "last_result": last_hud_res,
                                    "recent_reps": list(recent_reps),
                                    "summary": (
                                        self._build_summary(recent_reps, correct_count, target_reps)
                                        if goal_reached
                                        else None
                                    ),
                                    "error": None,
                                }
                            )
                    if goal_reached:
                        break
                if not goal_reached and counter.state == "IN_REP":
                    tail_ev = counter.flush_if_valid()
                    if tail_ev is not None:
                        _collect_future(block=True)
                        pending_event = tail_ev
                        pending_future = pool.submit(_score_rep_sync, tail_ev)
                _collect_future(block=True)
                with self._lock:
                    self._status.update(
                        {
                            "rep_count": int(counter.rep_count),
                            "correct_count": int(correct_count),
                            "last_result": last_hud_res,
                            "recent_reps": list(recent_reps),
                            "summary": self._build_summary(recent_reps, correct_count, target_reps),
                        }
                    )
        except Exception as exc:
            with self._lock:
                self._status["active"] = False
                self._status["state"] = "IDLE"
                self._status["error"] = f"Lỗi luồng Camera: {exc}"
        finally:
            if cap is not None:
                cap.release()
            with self._lock:
                self._active = False
                self._status["active"] = False
                if not self._status.get("completed") and recent_reps:
                    self._status["summary"] = self._build_summary(recent_reps, correct_count, target_reps)


WEBCAM_MANAGER = WebcamSessionManager(MODEL_MANAGER)
atexit.register(WEBCAM_MANAGER.stop)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    WEB_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_sessions()
    yield
    WEBCAM_MANAGER.stop()


app = FastAPI(title="ForgAI Web Studio", version="1.0.0", lifespan=_lifespan)


@app.middleware("http")
async def _no_cache_ui_assets(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class WebcamStartPayload(BaseModel):
    exercise: str = "squat"
    target_reps: int = 10
    use_all_folds: bool = False
    camera_index: int = 0
    mirror: bool = True


class WebcamControlPayload(BaseModel):
    exercise: str | None = None
    mirror: bool | None = None
    reset: bool = False


@app.get("/", response_class=HTMLResponse)
def get_index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Thiếu file giao diện web/static/index.html")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
def get_favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/exercises")
def get_exercises() -> JSONResponse:
    items = []
    for ex in GROUP_A_EXERCISES:
        spec = REP_CYCLE_SPECS[ex]
        meta = EXERCISE_UI_META[ex]
        items.append(
            {
                "id": ex,
                "name_vi": meta["vi"],
                "driver": spec["driver"],
                "driver_vi": meta["driver_vi"],
                "unit": spec["unit"],
                "start_ref": spec["start_ref"],
                "peak_ref": spec["peak_ref"],
                "min_rom": spec["min_rom"],
            }
        )
    return JSONResponse({"exercises": items})


@app.post("/api/preview-video")
async def preview_video(file: UploadFile = File(...)) -> JSONResponse:
    """Chuyển mã video trình duyệt không tự giải mã được (.MOV/HEVC/AVI/MKV) sang H.264 để xem trước."""
    job_id = f"prev_{uuid.uuid4().hex[:10]}"
    job_dir = WEB_SESSIONS_DIR / job_id
    try:
        in_path, orig_name = await _save_upload_to_disk(file, job_dir)
        norm_mp4 = job_dir / "normalized_h264.mp4"
        transcode_to_h264(
            in_path,
            norm_mp4,
            preset="veryfast",
            crf=19,
            cleanup_src=True,
            allow_fallback_copy=False,
        )
        fps, n_frames, w, h = _probe_video_meta(norm_mp4)
        if w <= 0 or h <= 0 or n_frames <= 0:
            raise ValueError("Không đọc được khung hình hợp lệ từ file video tải lên.")
        _cleanup_old_sessions()
        return JSONResponse(
            {
                "preview_job_id": job_id,
                "preview_url": f"/api/sessions/{job_id}/normalized_h264.mp4",
                "source_filename": orig_name,
                "width": w,
                "height": h,
                "num_frames": n_frames,
                "fps": round(fps, 2),
                "duration_sec": round(n_frames / max(fps, 1.0), 2),
            }
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Không chuyển đổi được bản xem trước: {exc}") from exc


@app.post("/api/analyze-upload")
async def analyze_upload(
    file: UploadFile | None = File(None),
    prepared_job_id: str | None = Form(None),
    exercise: str = Form(...),
    analysis_mode: str = Form("single_rep"),
    use_all_folds: bool = Form(False),
) -> JSONResponse:
    ex = exercise.strip().lower()
    if ex not in GROUP_A_EXERCISES:
        if file is not None:
            await file.close()
        raise HTTPException(status_code=400, detail=f"Bài tập không thuộc Nhóm A: {exercise}")

    mode = analysis_mode.strip().lower()
    if mode not in {"single_rep", "multi_rep"}:
        if file is not None:
            await file.close()
        raise HTTPException(status_code=400, detail=f"Chế độ phân tích không hợp lệ: {analysis_mode}")

    prep_source: Path | None = None
    prep_id_clean = (prepared_job_id or "").strip()
    if prep_id_clean:
        if not _SESSION_ID_RE.match(prep_id_clean):
            if file is not None:
                await file.close()
            raise HTTPException(status_code=400, detail="Mã phiên xem trước không hợp lệ.")
        candidate = (WEB_SESSIONS_DIR / prep_id_clean / "normalized_h264.mp4").resolve()
        base = WEB_SESSIONS_DIR.resolve()
        if base in candidate.parents and candidate.is_file():
            prep_source = candidate
        elif file is None:
            raise HTTPException(
                status_code=410,
                detail="Bản xem trước trên máy chủ đã hết hạn, hệ thống đang tải lại file gốc...",
            )

    if prep_source is None and file is None:
        raise HTTPException(status_code=400, detail="Vui lòng chọn file video để phân tích.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = WEB_SESSIONS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        if prep_source is not None:
            if file is not None:
                await file.close()
            analyze_input = prep_source
            orig_name = Path(file.filename).name if (file is not None and file.filename) else f"{prep_id_clean}.mp4"
        else:
            assert file is not None
            in_path, orig_name = await _save_upload_to_disk(file, job_dir)
            analyze_input = _prepare_cv2_input(in_path, job_dir / "normalized_h264.mp4")

        predictor = MODEL_MANAGER.get_predictor(all_folds=bool(use_all_folds))
        fold_override = None if use_all_folds else MODEL_MANAGER.default_fold
        raw_hud = job_dir / "raw_hud.mp4"
        final_h264 = job_dir / "hud_h264.mp4"
        video_url = f"/api/sessions/{job_id}/hud_h264.mp4"

        if mode == "single_rep":
            res, pose_seq = predictor.predict_from_video(
                video_path=analyze_input,
                exercise_name=ex,
                fold_override=fold_override,
            )
            if float(res["det_rate"]) < 0.05:
                raise ValueError(
                    "Không phát hiện được người tập trong video (tỷ lệ bắt khớp < 5%). "
                    "Hãy kiểm tra lại góc quay và ánh sáng."
                )
            export_annotated_video(analyze_input, pose_seq, res, raw_hud)
            transcode_to_h264(raw_hud, final_h264)
            payload = _serialize_single_result(res, video_url=video_url)
        else:
            payload = analyze_multi_rep_video(
                video_path=analyze_input,
                exercise=ex,
                raw_hud_out=raw_hud,
                predictor=predictor,
                fold_override=fold_override,
            )
            transcode_to_h264(raw_hud, final_h264)
            payload["video_url"] = video_url

        payload["job_id"] = job_id
        payload["source_filename"] = orig_name
        _cleanup_old_sessions()
        return JSONResponse(payload)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except ValueError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý video: {exc}") from exc


@app.get("/api/sessions/{job_id}/{filename}")
def get_session_file(job_id: str, filename: str, download: bool = False) -> FileResponse:
    if not _SESSION_ID_RE.match(job_id) or not _SAFE_FILENAME_RE.match(filename):
        raise HTTPException(status_code=400, detail="Đường dẫn session không hợp lệ.")
    target = (WEB_SESSIONS_DIR / job_id / filename).resolve()
    base = WEB_SESSIONS_DIR.resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file kết quả.")
    if download:
        return FileResponse(path=target, media_type="video/mp4", filename=filename)
    return FileResponse(path=target, media_type="video/mp4")


@app.post("/api/webcam/start")
def start_webcam(payload: WebcamStartPayload) -> JSONResponse:
    try:
        st = WEBCAM_MANAGER.start(
            exercise=payload.exercise,
            target_reps=payload.target_reps,
            use_all_folds=payload.use_all_folds,
            camera_index=payload.camera_index,
            mirror=payload.mirror,
        )
        return JSONResponse(st)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/webcam/stop")
def stop_webcam() -> JSONResponse:
    return JSONResponse(WEBCAM_MANAGER.stop())


@app.post("/api/webcam/control")
def control_webcam(payload: WebcamControlPayload) -> JSONResponse:
    try:
        return JSONResponse(
            WEBCAM_MANAGER.update_control(
                exercise=payload.exercise,
                mirror=payload.mirror,
                reset=payload.reset,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/webcam/status")
def webcam_status() -> JSONResponse:
    return JSONResponse(WEBCAM_MANAGER.get_status())


@app.get("/api/webcam/stream")
def webcam_stream() -> StreamingResponse:
    def _gen():
        last_seq = -1
        for _ in itertools.count():
            st = WEBCAM_MANAGER.get_status()
            frame_bytes, seq = WEBCAM_MANAGER.get_latest_jpeg_with_seq()
            if frame_bytes is not None and seq != last_seq:
                last_seq = seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            if not st["active"]:
                break
            time.sleep(0.02)

    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/dashboard")
def get_dashboard() -> JSONResponse:
    eval_path = REPORTS_DIR / "task2_eval_summary.json"
    bench_path = REPORTS_DIR / "step7_latency_benchmark.json"
    eval_data = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    bench_data = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}

    figs: list[dict[str, str]] = []
    fig_dir = REPORTS_DIR / "figures"
    for fname, caption in FIGURE_CAPTIONS.items():
        p = fig_dir / fname
        if p.exists():
            figs.append(
                {
                    "key": p.stem,
                    "filename": p.name,
                    "title": caption,
                    "group": "eval" if fname.startswith("eval_") else "data",
                    "url": f"/api/figures/{p.name}",
                }
            )
    return JSONResponse(
        {
            "zero_leakage_audit": eval_data.get("zero_leakage_audit", {}),
            "cv_summary": eval_data.get("cv_summary", {}),
            "per_exercise": eval_data.get("per_exercise", []),
            "rule_engine_metrics": eval_data.get("rule_engine_metrics", {}),
            "benchmark": bench_data,
            "figures": figs,
        }
    )


@app.get("/api/figures/{filename}")
def get_figure_file(filename: str) -> FileResponse:
    if not _SAFE_FILENAME_RE.match(filename) or not filename.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Tên file ảnh không hợp lệ.")
    fig_dir = (REPORTS_DIR / "figures").resolve()
    target = (fig_dir / filename).resolve()
    if fig_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy biểu đồ.")
    return FileResponse(path=target, media_type="image/png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForgAI Web Studio Server (Bước 7)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Địa chỉ IP lắng nghe")
    parser.add_argument("--port", type=int, default=8000, help="Cổng HTTP server")
    parser.add_argument("--no-browser", action="store_true", help="Không tự động mở trình duyệt khi khởi chạy")
    return parser.parse_args()


def main() -> None:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args()
    url = f"http://{args.host}:{args.port}"
    print("=" * 72)
    print(f"  FORGAI WEB STUDIO ĐANG KHỞI ĐỘNG TẠI: {url}")
    print("  Nhấn Ctrl+C trong cửa sổ này để dừng máy chủ an toàn.")
    print("=" * 72)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
