# -*- coding: utf-8 -*-
"""
tools/realtime_demo.py
Ứng dụng Demo & Kiểm thử Realtime (30 FPS) cho ForgAI Task 2 (Bước 7).
Hỗ trợ chạy trực tiếp từ Webcam (--source 0) hoặc file video bất kỳ (--source path/to/video.mp4),
cùng chế độ không màn hình (--no-gui) để đo hiệu năng, FPS, độ trễ và chấm rep tự động.

Phím tắt khi mở cửa sổ GUI:
  - [1]..[8]: Chọn bài tập trong 8 bài Nhóm A
  - [Space] : Tạm dừng / Tiếp tục
  - [M]     : Bật / tắt lật gương hiển thị
  - [R]     : Đặt lại bộ đếm rep & trạng thái
  - [Q/ESC] : Thoát an toàn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import GROUP_A_EXERCISES, MIN_VISIBILITY_THRESHOLD
from src.constants import SKELETON_EDGES_13
from src.realtime import (
    CycleRepCounter,
    ForgAIRealtimeScorer,
    MediaPipeRealtimeTracker,
    RepCompletedEvent,
    compute_driver_signal_with_vis,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
)


def _load_fonts() -> tuple[Any, Any, Any]:
    for font_path in _FONT_CANDIDATES:
        if font_path.exists():
            return (
                ImageFont.truetype(str(font_path), 18),
                ImageFont.truetype(str(font_path), 14),
                ImageFont.truetype(str(font_path), 13),
            )
    fallback = ImageFont.load_default()
    return fallback, fallback, fallback


def _wrap_text_px(draw: ImageDraw.ImageDraw, text: str, font: Any, max_px: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip() if cur else w
        if draw.textlength(cand, font=font) <= max_px:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_skeleton(
    canvas: np.ndarray,
    joints: np.ndarray,
    detected: bool,
    color_bgr: tuple[int, int, int],
    mirror_display: bool,
) -> None:
    if not detected:
        return
    h, w = canvas.shape[:2]
    pts: list[tuple[int, int] | None] = []
    for j in range(joints.shape[0]):
        vis = float(joints[j, 5])
        if vis < MIN_VISIBILITY_THRESHOLD:
            pts.append(None)
            continue
        ix = float(joints[j, 3])
        iy = float(joints[j, 4])
        if mirror_display:
            ix = 1.0 - ix
        px = int(np.clip(ix * w, 0, w - 1))
        py = int(np.clip(iy * h, 0, h - 1))
        pts.append((px, py))

    for a, b in SKELETON_EDGES_13:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(canvas, pts[a], pts[b], color_bgr, 3, lineType=cv2.LINE_AA)
    for p in pts:
        if p is not None:
            cv2.circle(canvas, p, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, 3, color_bgr, -1, lineType=cv2.LINE_AA)


def _draw_realtime_hud(
    frame_bgr: np.ndarray,
    fonts: tuple[Any, Any, Any],
    exercise: str,
    counter: CycleRepCounter,
    correct_count: int,
    last_result: dict[str, Any] | None,
    scoring_pending: bool,
    fps_avg: float,
    proc_ms_avg: float,
    driver_val: float | None,
    target_reps: int = 0,
) -> np.ndarray:
    """Đóng khung đệm trên/dưới cố định để không che người tập và vẽ HUD tiếng Việt bằng Pillow."""
    fh, fw = frame_bgr.shape[:2]
    header_h = 84
    footer_h = 138
    canvas_w = max(fw, 760)
    canvas_h = header_h + fh + footer_h

    canvas = np.full((canvas_h, canvas_w, 3), (20, 22, 28), dtype=np.uint8)
    x_off = (canvas_w - fw) // 2
    canvas[header_h : header_h + fh, x_off : x_off + fw] = frame_bgr

    font_title, font_body, font_small = fonts
    bg_rgb = (28, 22, 20)
    hdr_pil = Image.new("RGB", (canvas_w, header_h), bg_rgb)
    hdr_draw = ImageDraw.Draw(hdr_pil)
    ftr_pil = Image.new("RGB", (canvas_w, footer_h), bg_rgb)
    ftr_draw = ImageDraw.Draw(ftr_pil)

    ex_idx = GROUP_A_EXERCISES.index(exercise) + 1
    if counter.state == "IN_REP":
        state_vi = "ĐANG THỰC HIỆN REP"
        state_col = (225, 230, 235)
    elif getattr(counter, "is_tracking_lost", False):
        state_vi = "MẤT KHUNG XƯƠNG - HÃY ĐỨNG TRONG KHUNG HÌNH"
        state_col = (255, 165, 95)
    elif getattr(counter, "is_stabilizing", False):
        state_vi = "ĐANG ỔN ĐỊNH KHUNG XƯƠNG..."
        state_col = (140, 225, 255)
    elif getattr(counter, "is_waiting_ready", False):
        state_vi = "CHỜ TƯ THẾ CHUẨN BỊ - GIỮ ỔN ĐỊNH"
        state_col = (255, 215, 110)
    else:
        state_vi = "SẴN SÀNG / NGHỈ"
        state_col = (225, 230, 235)
    rep_part = f"Rep: {counter.rep_count}/{target_reps}" if target_reps > 0 else f"Rep: {counter.rep_count}"
    header_l = f"FORGAI  |  [{ex_idx}/8] {exercise.upper()}  |  {rep_part} (Đúng: {correct_count})"
    header_r = f"{fps_avg:.1f} FPS  |  {proc_ms_avg:.1f} ms/frame"
    hdr_draw.text((16, 8), header_l, font=font_title, fill=(255, 220, 90))
    r_w = hdr_draw.textlength(header_r, font=font_body)
    hdr_draw.text((canvas_w - r_w - 16, 11), header_r, font=font_body, fill=(140, 235, 255))

    unit = counter.spec["unit"]
    drv_txt = f"{counter.spec['driver']} = {driver_val:.1f}{unit}" if driver_val is not None else "Chưa phát hiện người tập"
    hdr_draw.text((16, 36), f"Trạng thái: {state_vi}   |   {drv_txt}", font=font_body, fill=state_col)
    hdr_draw.text((16, 58), "Phím: [1-8] Đổi bài   [Space] Dừng   [M] Lật gương   [R] Reset   [Q] Thoát", font=font_small, fill=(165, 175, 190))

    bar_w = 155
    bar_x1, bar_y1 = canvas_w - bar_w - 16, 48
    pct_txt = f"Tiến độ rep: {int(counter.last_progress * 100)}%"
    hdr_draw.text((bar_x1, 31), pct_txt, font=font_small, fill=(210, 218, 228))

    max_txt_px = canvas_w - 32
    if last_result is None:
        msg = "Đang chấm điểm rep vừa hoàn thành..." if scoring_pending else "Hãy thực hiện trọn vẹn 1 rep trước camera để AI chấm điểm và góp ý..."
        ftr_draw.text((16, 26), msg, font=font_body, fill=(200, 210, 220))
    else:
        is_inc = bool(last_result["is_incorrect"])
        badge_col = (255, 95, 95) if is_inc else (85, 235, 130)
        p_pct = float(last_result["p_incorrect"]) * 100.0
        pend_tag = "  (Đang chấm rep mới...)" if scoring_pending else ""
        title_line = (
            f"Rep #{last_result.get('rep_index', counter.rep_count)}: {last_result['label_vi']}  |  "
            f"P(Sai)={p_pct:.1f}%  |  {last_result['duration_sec']}s  |  AI {last_result['infer_ms']:.0f}ms{pend_tag}"
        )
        ftr_draw.text((16, 10), title_line, font=font_title, fill=badge_col)

        err_line = f"Chẩn đoán: {last_result['vietnamese_error_name']} — {last_result['measured_evidence']}"
        advice_line = f"Gợi ý: {last_result['coaching_advice']}"
        y_curr = 36
        for line in _wrap_text_px(ftr_draw, err_line, font_small, max_txt_px)[:3]:
            ftr_draw.text((16, y_curr), line, font=font_small, fill=(255, 235, 175) if is_inc else (210, 250, 220))
            y_curr += 19
        for line in _wrap_text_px(ftr_draw, advice_line, font_small, max_txt_px)[:2]:
            ftr_draw.text((16, y_curr), line, font=font_small, fill=(235, 240, 245))
            y_curr += 19

    canvas[:header_h, :] = np.asarray(hdr_pil)[:, :, ::-1]
    canvas[header_h + fh :, :] = np.asarray(ftr_pil)[:, :, ::-1]
    cv2.line(canvas, (0, header_h - 1), (canvas_w, header_h - 1), (60, 66, 80), 1)
    cv2.line(canvas, (0, header_h + fh), (canvas_w, header_h + fh), (60, 66, 80), 1)
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 12), (65, 70, 82), -1)
    fill_w = int(bar_w * np.clip(counter.last_progress, 0.0, 1.0))
    bar_col = (0, 200, 255) if counter.state == "IN_REP" else (110, 210, 120)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y1 + 12), bar_col, -1)
    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + bar_w, bar_y1 + 12), (180, 188, 200), 1)
    return canvas


def _open_capture(source: str, width: int = 640, height: int = 480) -> tuple[cv2.VideoCapture, bool]:
    is_cam = source.strip().isdigit()
    if is_cam:
        cam_idx = int(source.strip())
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
                return cap, True
            cap.release()
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        return cap, True
    return cv2.VideoCapture(source), False


def run_realtime(
    source: str = "0",
    exercise: str = "squat",
    fold: int = 2,
    mirror: bool = False,
    no_gui: bool = False,
    max_frames: int | None = None,
    save_out: str | None = None,
) -> dict[str, Any]:
    """Chạy trọn vẹn pipeline Realtime: Capture -> MediaPipe -> RepCounter -> ThreadPool Scorer -> HUD."""
    if exercise not in GROUP_A_EXERCISES:
        raise ValueError(f"Bài tập '{exercise}' không thuộc 8 bài Nhóm A: {GROUP_A_EXERCISES}")

    print(f"[Init] Đang nạp ForgAIRealtimeScorer (Fold {fold}, bài mặc định='{exercise}')...")
    scorer = ForgAIRealtimeScorer(fold=fold)
    cap, is_cam = _open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở nguồn video/webcam: {source}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if src_fps <= 1.0 or src_fps > 120.0:
        src_fps = 30.0

    fonts = _load_fonts()
    counter = CycleRepCounter(exercise=exercise, require_ready_pose=is_cam)
    window_name = "FORGAI - Realtime Gym Technique Analyzer (Step 7)"

    if not no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 640)

    writer: cv2.VideoWriter | None = None
    mirror_display = bool(mirror or is_cam)
    paused = False
    frame_idx = 0
    correct_count = 0
    last_result: dict[str, Any] | None = None
    rep_logs: list[dict[str, Any]] = []
    pending_future: Future | None = None
    pending_event: RepCompletedEvent | None = None

    proc_times_ms: list[float] = []
    loop_fps_ema = src_fps
    proc_ms_ema = 0.0
    last_loop_t = time.perf_counter()

    def _collect_future_if_ready(block: bool = False) -> None:
        nonlocal pending_future, pending_event, last_result, correct_count
        if pending_future is None:
            return
        if not block and not pending_future.done():
            return
        res = pending_future.result()
        if pending_event is not None:
            res["rep_index"] = pending_event.rep_index
            res["observed_rom"] = round(pending_event.observed_rom, 2)
        last_result = res
        if not res["is_incorrect"]:
            correct_count += 1
        rep_logs.append(res)
        print(
            f"  [Rep #{res.get('rep_index', len(rep_logs))}] {res['exercise']} ({res['duration_sec']}s) -> "
            f"{res['label_vi']} | P(Sai)={res['p_incorrect']*100:.1f}% | {res['feedback_vi']}"
        )
        pending_future = None
        pending_event = None

    with MediaPipeRealtimeTracker(conf_threshold=0.25) as tracker, ThreadPoolExecutor(max_workers=1) as pool:
        try:
            while True:
                if not no_gui and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                    if max_frames is not None and frame_idx > max_frames:
                        break

                    # Chuẩn hóa kích thước tối đa bề rộng 640px để bảo đảm tốc độ CPU < 20ms/frame
                    if frame.shape[1] > 640:
                        new_h = int(round(frame.shape[0] * (640.0 / frame.shape[1])))
                        frame = cv2.resize(frame, (640, max(240, new_h)))

                    t_proc0 = time.perf_counter()
                    ts_ms = int(time.perf_counter() * 1000.0) if is_cam else int(frame_idx * 1000.0 / src_fps)
                    ts_sec = ts_ms / 1000.0

                    joints, detected = tracker.process_frame(frame, ts_ms=ts_ms)
                    drv_raw, drv_vis = (0.0, False) if not detected else compute_driver_signal_with_vis(joints, counter.exercise)
                    drv_val = drv_raw if drv_vis else counter.ema_val

                    event = counter.update(joints, detected=detected, timestamp_sec=ts_sec)
                    _collect_future_if_ready(block=False)
                    if event is not None:
                        if pending_future is not None:
                            _collect_future_if_ready(block=True)
                        pending_event = event
                        pending_future = pool.submit(
                            scorer.score_rep,
                            event.pose_seq,
                            event.exercise,
                            event.duration_sec,
                        )

                    disp = cv2.flip(frame, 1) if mirror_display else frame.copy()
                    if counter.state == "IN_REP":
                        skel_col = (0, 200, 255)
                    elif last_result is not None:
                        skel_col = (70, 70, 255) if last_result["is_incorrect"] else (70, 230, 90)
                    else:
                        skel_col = (220, 220, 90)

                    _draw_skeleton(disp, joints, detected, skel_col, mirror_display=mirror_display)

                    proc_ms = (time.perf_counter() - t_proc0) * 1000.0
                    proc_times_ms.append(proc_ms)
                    proc_ms_ema = proc_ms if frame_idx == 1 else (0.15 * proc_ms + 0.85 * proc_ms_ema)

                    now_t = time.perf_counter()
                    dt = max(1e-4, now_t - last_loop_t)
                    last_loop_t = now_t
                    inst_fps = min(120.0, 1.0 / dt)
                    loop_fps_ema = inst_fps if frame_idx == 1 else (0.15 * inst_fps + 0.85 * loop_fps_ema)

                    if not no_gui or save_out:
                        disp_hud = _draw_realtime_hud(
                            disp,
                            fonts=fonts,
                            exercise=counter.exercise,
                            counter=counter,
                            correct_count=correct_count,
                            last_result=last_result,
                            scoring_pending=(pending_future is not None),
                            fps_avg=loop_fps_ema,
                            proc_ms_avg=proc_ms_ema,
                            driver_val=drv_val,
                        )
                        if save_out:
                            if writer is None:
                                out_path = Path(save_out)
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                writer = cv2.VideoWriter(str(out_path), fourcc, src_fps, (disp_hud.shape[1], disp_hud.shape[0]))
                            writer.write(disp_hud)
                        if not no_gui:
                            cv2.imshow(window_name, disp_hud)

                if not no_gui:
                    key = cv2.waitKey(1 if not paused else 30) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key == ord("m"):
                        mirror_display = not mirror_display
                    elif key == ord("r"):
                        _collect_future_if_ready(block=True)
                        counter.reset()
                        correct_count = 0
                        last_result = None
                    elif ord("1") <= key <= ord("8"):
                        new_ex = GROUP_A_EXERCISES[key - ord("1")]
                        _collect_future_if_ready(block=True)
                        counter.reset(new_ex)
                        correct_count = 0
                        last_result = None
                        print(f"[Switch] Đã chuyển sang bài tập: {new_ex}")

            # Kết thúc video offline: kiểm tra nốt rep cuối đang hoàn thành dở (nếu đạt ngưỡng)
            if not is_cam:
                tail_event = counter.flush_if_valid()
                if tail_event is not None:
                    if pending_future is not None:
                        _collect_future_if_ready(block=True)
                    pending_event = tail_event
                    pending_future = pool.submit(
                        scorer.score_rep,
                        tail_event.pose_seq,
                        tail_event.exercise,
                        tail_event.duration_sec,
                    )
            _collect_future_if_ready(block=True)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if not no_gui:
                cv2.destroyAllWindows()

    avg_ms = float(np.mean(proc_times_ms)) if proc_times_ms else 0.0
    p95_ms = float(np.percentile(proc_times_ms, 95)) if proc_times_ms else 0.0
    summary = {
        "source": source,
        "exercise": counter.exercise,
        "fold": fold,
        "frames_processed": frame_idx,
        "reps_completed": len(rep_logs),
        "correct_reps": correct_count,
        "avg_frame_ms": round(avg_ms, 2),
        "p95_frame_ms": round(p95_ms, 2),
        "effective_fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        "reps": rep_logs,
    }
    print(
        f"[Summary] Frames={frame_idx} | Reps={len(rep_logs)} (Đúng={correct_count}) | "
        f"Avg={avg_ms:.2f} ms/frame (p95={p95_ms:.2f} ms, ~{summary['effective_fps']:.1f} FPS)"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForgAI Step 7 - Demo Realtime Webcam 30 FPS")
    parser.add_argument("--source", type=str, default="0", help="Chỉ số webcam ('0') hoặc đường dẫn file video .mp4")
    parser.add_argument("--exercise", type=str, default="squat", choices=GROUP_A_EXERCISES, help="Bài tập khởi đầu")
    parser.add_argument("--fold", type=int, default=2, choices=[0, 1, 2, 3, 4], help="Checkpoint Fold Bước E.2 (0..4)")
    parser.add_argument("--mirror", action="store_true", help="Lật gương khung hình hiển thị")
    parser.add_argument("--no-gui", action="store_true", help="Chạy không mở cửa sổ GUI (dùng cho test & benchmark)")
    parser.add_argument("--max-frames", type=int, default=None, help="Giới hạn số frame xử lý tối đa")
    parser.add_argument("--save-out", type=str, default=None, help="Đường dẫn lưu video xuất kèm HUD & khung xương")
    parser.add_argument("--json-out", type=str, default=None, help="Đường dẫn lưu JSON kết quả tóm tắt phiên chạy")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res_summary = run_realtime(
        source=args.source,
        exercise=args.exercise,
        fold=args.fold,
        mirror=args.mirror,
        no_gui=args.no_gui,
        max_frames=args.max_frames,
        save_out=args.save_out,
    )
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res_summary, ensure_ascii=False, indent=2), encoding="utf-8")
