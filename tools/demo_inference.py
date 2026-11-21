# -*- coding: utf-8 -*-
"""
tools/demo_inference.py
Công cụ Demo Trực Quan & Kiểm Chuẩn Bước 7 (Deployment - ForgAI Task 2):
  1. --segment-id <ID> / --video <PATH> --exercise <NAME>:
       Hiển thị bảng kết quả màu sắc (rich UI) trên Terminal kèm xuất video .mp4
       vẽ khung xương 13 khớp (tô đỏ/cam các khớp phạm lỗi) + HUD tiếng Việt.
  2. --showcase:
       Chạy thử nghiệm tiêu biểu qua đủ 8 bài tập Nhóm A (8 Đúng + 8 Sai, dùng đúng
       Fold Out-of-Fold không rò rỉ dữ liệu), in bảng tổng hợp rich & xuất video demo.
  3. --benchmark:
       Đo kiểm chuẩn độ trễ (100 lượt chạy liên tiếp) cho Single-Fold vs 5-Fold
       Ensemble và MediaPipe + End-to-End so với ngưỡng < 50 ms/frame.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    GROUP_A_EXERCISES,
    MIN_VISIBILITY_THRESHOLD,
    OUTPUTS_DIR,
    POSE_NPY_DIR,
    REPORTS_DIR,
    TASK2_CSV_PATH,
)
from src.constants import ERROR_REASON_VN_DICT, J_IDX, SKELETON_EDGES_13
from src.inference import ForgAITask2Predictor

DEMO_OUT_DIR = OUTPUTS_DIR / "demo"

_FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
)
_JOINT_SHORT_VN = {i: name.replace("left_", "L ").replace("right_", "R ") for name, i in J_IDX.items()}


def _make_console() -> Console:
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    return Console()


def _load_fonts() -> tuple[Any, Any, Any]:
    for fp in _FONT_CANDIDATES:
        if fp.exists():
            return (
                ImageFont.truetype(str(fp), 20),
                ImageFont.truetype(str(fp), 15),
                ImageFont.truetype(str(fp), 13),
            )
    fb = ImageFont.load_default()
    return fb, fb, fb


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: Any, max_px: int) -> list[str]:
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


def _render_rich_result(console: Console, res: dict[str, Any], meta_info: dict[str, Any] | None = None) -> None:
    """In bảng phân tích kết quả màu sắc chi tiết bằng thư viện rich."""
    is_inc = bool(res["is_incorrect"])
    border_style = "red" if is_inc else "green"
    verdict_badge = "[bold red]SAI KỸ THUẬT (INCORRECT)[/bold red]" if is_inc else "[bold green]ĐÚNG KỸ THUẬT (CORRECT)[/bold green]"
    score_pct = float(res["form_score_pct"])
    bar_filled = int(round(score_pct / 5.0))
    score_bar = f"[{'red' if is_inc else 'green'}]{'█' * bar_filled}{'░' * (20 - bar_filled)}[/] {score_pct:.1f}%"

    header_lines = [
        f"[bold cyan]Bài tập:[/bold cyan] {res['exercise'].upper()}   |   [bold cyan]Trạng thái:[/bold cyan] {verdict_badge}",
        f"[bold cyan]Điểm kỹ thuật (Form Score):[/bold cyan] {score_bar}   |   [bold cyan]P(Sai) Hybrid:[/bold cyan] {res['p_hybrid']*100:.1f}% (Ngưỡng {res['threshold']*100:.0f}%)",
        f"[bold cyan]Thời lượng rep:[/bold cyan] {res['duration_sec']}s ({res['num_frames']} frames @ {res['fps']} FPS)   |   [bold cyan]Tỷ lệ bắt khớp:[/bold cyan] {res['det_rate']*100:.1f}%   |   [bold cyan]Folds:[/bold cyan] {res['folds_used']}",
    ]
    if meta_info:
        gt_lbl = str(meta_info.get("form_label", "")).upper()
        gt_err = str(meta_info.get("error_reason", "none"))
        sid = str(meta_info.get("segment_id", ""))
        header_lines.append(
            f"[bold magenta]Mẫu kiểm thử:[/bold magenta] {sid} ({meta_info.get('source', '')})   |   [bold magenta]Nhãn gốc:[/bold magenta] {gt_lbl} ({gt_err})"
        )

    console.print(Panel("\n".join(header_lines), title="FORGAI STEP 7 - PHÂN TÍCH KỸ THUẬT TẬP GYM", border_style=border_style))

    # Bảng xác suất 4 nhánh
    branch_tbl = Table(title="Chi tiết Xác suất Sai theo 4 Nhánh Mô hình & Độ trễ", box=box.ROUNDED, border_style="cyan")
    branch_tbl.add_column("Nhánh mô hình", style="bold white")
    branch_tbl.add_column("P(Incorrect)", justify="right")
    branch_tbl.add_column("Dự đoán nhánh", justify="center")
    branch_tbl.add_column("Độ trễ thành phần", justify="right")

    tm = res["timings_ms"]
    rows_b = [
        ("BiomechanicalRF (114-d E.3+ + Prior Shift)", res["p_rf"], f"Đặc trưng: {tm['feature_extract_ms']:.2f} ms"),
        ("CNN-BiLSTM + FiLM", res["p_cnn_bilstm"], f"Forward AI: {tm['model_inference_ms']:.2f} ms"),
        ("Transformer + FiLM", res["p_transformer"], f"Rule Engine: {tm['rule_engine_ms']:.2f} ms"),
        ("Hybrid Ensemble (0.50 / 0.25 / 0.25)", res["p_hybrid"], f"Tổng AI lõi: {tm['total_ai_ms']:.2f} ms/rep"),
    ]
    for name, p_val, lat_txt in rows_b:
        tag = "[red]INCORRECT[/red]" if p_val >= res["threshold"] else "[green]CORRECT[/green]"
        branch_tbl.add_row(name, f"{p_val*100:.1f}%", tag, lat_txt)
    console.print(branch_tbl)

    # Bảng chẩn đoán lỗi & lời khuyên
    diag_lines = [
        f"[bold]Chẩn đoán chính:[/bold] {res['vietnamese_error_name']} (`{res['predicted_error_code']}`)",
        f"[bold]Bằng chứng cơ sinh học:[/bold] {res['measured_evidence']}",
        f"[bold yellow]Gợi ý chỉnh sửa:[/bold yellow] {res['coaching_advice']}",
    ]
    if res["highlight_joints"]:
        hi_names = ", ".join(_JOINT_SHORT_VN[j] for j in res["highlight_joints"] if j in _JOINT_SHORT_VN)
        diag_lines.append(f"[bold red]Khớp đánh dấu cảnh báo trên HUD:[/bold red] {hi_names}")
    console.print(Panel("\n".join(diag_lines), title="BIOMECHANICAL EXPLAINABLE FEEDBACK", border_style="yellow" if is_inc else "green"))

    if res.get("top3_candidates"):
        c_tbl = Table(title="Top-3 Nguyên nhân Cơ sinh học Tiềm ẩn", box=box.SIMPLE_HEAVY)
        c_tbl.add_column("#", justify="center")
        c_tbl.add_column("Mã lỗi")
        c_tbl.add_column("Tên lỗi tiếng Việt")
        c_tbl.add_column("Điểm tin cậy", justify="right")
        c_tbl.add_column("Điểm luật vật lý", justify="right")
        for idx, cand in enumerate(res["top3_candidates"], 1):
            c_tbl.add_row(
                str(idx),
                str(cand["error_reason"]),
                str(cand["vietnamese_name"]),
                f"{cand['score']*100:.1f}%",
                f"{cand['rule_score']*100:.1f}%",
            )
        console.print(c_tbl)


def export_annotated_video(
    video_path: Path,
    pose_seq: np.ndarray,
    result: dict[str, Any],
    out_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Path:
    """
    Xuất video .mp4 minh họa kèm khung xương 13 khớp (tô đỏ khớp lỗi),
    bảng HUD bán trong suốt góc trên bên trái & dải cảnh báo tiếng Việt phía dưới.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video gốc để xuất demo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or result.get("fps", 30.0))
    if fps <= 1.0 or fps > 120.0:
        fps = 30.0

    fonts = _load_fonts()
    font_title, font_body, font_small = fonts
    is_inc = bool(result["is_incorrect"])
    hi_joints = set(result.get("highlight_joints", []))
    hi_edges = {tuple(e) for e in result.get("highlight_edges", [])}
    driver_key = str(result.get("driver_key", ""))
    driver_curve = result.get("driver_curve", [])
    unit = "" if "dist" in driver_key else "°"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None

    frame_idx = 0
    local_idx = 0
    max_seq = len(pose_seq)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx < start_frame:
                frame_idx += 1
                continue
            if end_frame is not None and frame_idx > end_frame:
                break
            if local_idx >= max_seq:
                break

            # Đảm bảo chiều cao hiển thị tối thiểu ~560px để chữ HUD đọc nét
            fh, fw = frame.shape[:2]
            if fh < 540:
                scale = 540.0 / float(fh)
                frame = cv2.resize(frame, (max(320, int(round(fw * scale))), 540))
            fh, fw = frame.shape[:2]

            # Tạo canvas đủ rộng tối thiểu 760px có dải chân trang riêng cho giải thích tiếng Việt
            canvas_w = max(fw, 760)
            footer_h = 92
            canvas_h = fh + footer_h
            canvas = np.full((canvas_h, canvas_w, 3), (18, 20, 26), dtype=np.uint8)
            x_off = (canvas_w - fw) // 2

            # Vẽ khung xương trực tiếp lên bản sao frame video
            disp = frame.copy()
            joints = pose_seq[local_idx]
            has_pose = bool(np.abs(joints[:, :3]).sum() > 1e-6)
            if has_pose:
                pts: list[tuple[int, int] | None] = []
                for j in range(joints.shape[0]):
                    if float(joints[j, 5]) < MIN_VISIBILITY_THRESHOLD:
                        pts.append(None)
                    else:
                        px = int(np.clip(float(joints[j, 3]) * fw, 0, fw - 1))
                        py = int(np.clip(float(joints[j, 4]) * fh, 0, fh - 1))
                        pts.append((px, py))

                base_edge_col = (0, 210, 255) if is_inc else (80, 230, 110)
                fault_edge_col = (40, 40, 255)
                for a, b in SKELETON_EDGES_13:
                    if pts[a] is not None and pts[b] is not None:
                        is_fault_edge = (a, b) in hi_edges or (b, a) in hi_edges
                        col = fault_edge_col if is_fault_edge else base_edge_col
                        thick = 5 if is_fault_edge else 3
                        cv2.line(disp, pts[a], pts[b], col, thick, lineType=cv2.LINE_AA)

                for j, p in enumerate(pts):
                    if p is None:
                        continue
                    if j in hi_joints:
                        cv2.circle(disp, p, 10, (0, 140, 255), 2, lineType=cv2.LINE_AA)
                        cv2.circle(disp, p, 6, (40, 40, 255), -1, lineType=cv2.LINE_AA)
                    else:
                        cv2.circle(disp, p, 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                        cv2.circle(disp, p, 3, base_edge_col, -1, lineType=cv2.LINE_AA)

            canvas[:fh, x_off : x_off + fw] = disp

            # Hộp HUD bán trong suốt ở góc trên bên trái
            hud_w = min(430, canvas_w - 20)
            hud_h = 98
            overlay = canvas.copy()
            cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (14, 18, 28), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0.0, canvas)
            border_bgr = (70, 70, 255) if is_inc else (80, 225, 115)
            cv2.rectangle(canvas, (10, 10), (10 + hud_w, 10 + hud_h), border_bgr, 2)

            cv2.line(canvas, (0, fh), (canvas_w, fh), border_bgr, 2)

            pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            ex_str = result["exercise"].upper()
            status_str = "SAI KỸ THUẬT" if is_inc else "ĐÚNG KỸ THUẬT"
            status_rgb = (255, 95, 95) if is_inc else (95, 240, 135)
            draw.text((20, 16), f"{ex_str}  |  {status_str}", font=font_title, fill=status_rgb)

            score_str = f"Form Score: {result['form_score_pct']:.1f}%  (P_Sai={result['p_hybrid']*100:.1f}%)"
            draw.text((20, 44), score_str, font=font_body, fill=(235, 240, 248))

            drv_v = driver_curve[local_idx] if local_idx < len(driver_curve) else 0.0
            frame_txt = f"{driver_key}: {drv_v:.1f}{unit}   |   Frame {local_idx + 1}/{max_seq}"
            draw.text((20, 70), frame_txt, font=font_small, fill=(170, 225, 255))

            # Dải chân trang giải thích lỗi & gợi ý chỉnh sửa
            y_bot = fh + 10
            err_msg = f"Chẩn đoán: {result['vietnamese_error_name']} — {result['measured_evidence']}"
            advice_msg = f"Gợi ý: {result['coaching_advice']}"
            max_px = canvas_w - 28
            for line in _wrap_text(draw, err_msg, font_small, max_px)[:2]:
                draw.text((14, y_bot), line, font=font_small, fill=(255, 225, 165) if is_inc else (200, 250, 210))
                y_bot += 20
            for line in _wrap_text(draw, advice_msg, font_small, max_px)[:2]:
                draw.text((14, y_bot), line, font=font_small, fill=(225, 232, 242))
                y_bot += 19

            out_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_bgr.shape[1], out_bgr.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Không thể khởi tạo VideoWriter mp4v cho: {out_path}")
            writer.write(out_bgr)

            frame_idx += 1
            local_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return out_path


def run_single_demo(args: argparse.Namespace, console: Console) -> None:
    df_meta = pd.read_csv(TASK2_CSV_PATH, keep_default_na=False)
    oof_path = REPORTS_DIR / "cv_oof_test_predictions.csv"
    oof_lookup = pd.read_csv(oof_path).set_index("segment_id") if oof_path.exists() else None

    if args.segment_id:
        matches = df_meta[df_meta["segment_id"] == args.segment_id]
        if matches.empty:
            raise ValueError(f"Không tìm thấy segment_id '{args.segment_id}' trong {TASK2_CSV_PATH}")
        row = matches.iloc[0]
        video_path = PROJECT_ROOT / str(row["video_path"])
        exercise = str(row["exercise"])
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        meta_info = row.to_dict()
        fold_use = args.fold
        if fold_use is None and not args.all_folds and oof_lookup is not None and args.segment_id in oof_lookup.index:
            fold_use = int(oof_lookup.loc[args.segment_id, "fold"])
        out_name = f"demo_{args.segment_id}.mp4"
    else:
        if not args.video:
            raise ValueError("Cần chỉ định `--video <path>` hoặc `--segment-id <id>`!")
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")
        if not args.exercise:
            raise ValueError("Khi chạy với `--video`, vui lòng truyền thêm `--exercise <tên_bài_tập>`!")
        exercise = args.exercise
        start_frame = 0
        end_frame = None
        meta_info = None
        fold_use = 2 if (args.fold is None and not args.all_folds) else args.fold
        out_name = f"demo_{video_path.stem}.mp4"

    predictor = ForgAITask2Predictor(
        fold=fold_use,
        ensemble_all_folds=bool(args.all_folds),
    )
    res, pose_seq = predictor.predict_from_video(
        video_path=video_path,
        exercise_name=exercise,
        start_frame=start_frame,
        end_frame=end_frame,
        fold_override=None if args.all_folds else fold_use,
    )
    _render_rich_result(console, res, meta_info=meta_info)

    if not args.no_video:
        out_video = Path(args.out_video) if args.out_video else (DEMO_OUT_DIR / out_name)
        saved = export_annotated_video(
            video_path=video_path,
            pose_seq=pose_seq,
            result=res,
            out_path=out_video,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        console.print(f"[bold green]Đã xuất video demo kèm HUD & khung xương tại:[/bold green] {saved}")


def run_showcase(args: argparse.Namespace, console: Console) -> None:
    """Chạy kiểm chứng thực tế 16 mẫu tiêu biểu (1 Đúng + 1 Sai cho mỗi bài trong 8 bài Nhóm A) ở đúng Fold OOF."""
    df_meta = pd.read_csv(TASK2_CSV_PATH, keep_default_na=False).set_index("segment_id")
    oof_path = REPORTS_DIR / "cv_oof_test_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Thiếu {oof_path}. Hãy chạy train_task2.py trước!")
    oof_df = pd.read_csv(oof_path)

    predictor = ForgAITask2Predictor(ensemble_all_folds=True)
    selected_rows: list[pd.Series] = []
    for ex in GROUP_A_EXERCISES:
        cor = oof_df[(oof_df["exercise"] == ex) & (oof_df["y_true"] == 0) & (oof_df["pred_hybrid"] == 0)].sort_values(
            "prob_hybrid", ascending=True
        )
        if cor.empty:
            cor = oof_df[(oof_df["exercise"] == ex) & (oof_df["y_true"] == 0)]
        inc = oof_df[(oof_df["exercise"] == ex) & (oof_df["y_true"] == 1) & (oof_df["pred_hybrid"] == 1)].sort_values(
            "prob_hybrid", ascending=False
        )
        if inc.empty:
            inc = oof_df[(oof_df["exercise"] == ex) & (oof_df["y_true"] == 1)]
        selected_rows.append(cor.iloc[0])
        selected_rows.append(inc.iloc[0])

    tbl = Table(
        title="FORGAI STEP 7 SHOWCASE — 16 MẪU OUT-OF-FOLD TIÊU BIỂU TRÊN 8 BÀI NHÓM A (ZERO LEAKAGE)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    tbl.add_column("Bài tập", style="bold")
    tbl.add_column("Segment ID")
    tbl.add_column("Nhãn gốc", justify="center")
    tbl.add_column("Dự đoán", justify="center")
    tbl.add_column("Form Score", justify="right")
    tbl.add_column("P(Sai)", justify="right")
    tbl.add_column("Chẩn đoán Luật / Lỗi gốc")
    tbl.add_column("Video HUD", style="dim")

    n_ok = 0
    exported_videos: list[str] = []
    for r in selected_rows:
        sid = str(r["segment_id"])
        ex = str(r["exercise"])
        oof_fold = int(r["fold"])
        meta = df_meta.loc[sid]
        v_path = PROJECT_ROOT / str(meta["video_path"])
        sf, ef = int(meta["start_frame"]), int(meta["end_frame"])

        res, pose_seq = predictor.predict_from_video(
            video_path=v_path,
            exercise_name=ex,
            start_frame=sf,
            end_frame=ef,
            fold_override=oof_fold,
        )
        match = (int(res["is_incorrect"]) == int(r["y_true"]))
        if match:
            n_ok += 1

        video_tag = "-"
        if not args.no_video:
            lbl_short = "correct" if int(r["y_true"]) == 0 else "incorrect"
            out_v = DEMO_OUT_DIR / f"showcase_{ex.replace(' ', '_').replace('-', '_')}_{lbl_short}.mp4"
            export_annotated_video(v_path, pose_seq, res, out_v, start_frame=sf, end_frame=ef)
            exported_videos.append(str(out_v))
            video_tag = out_v.name

        gt_str = "[green]CORRECT[/green]" if int(r["y_true"]) == 0 else "[red]INCORRECT[/red]"
        pr_str = "[red]INCORRECT[/red]" if res["is_incorrect"] else "[green]CORRECT[/green]"
        err_disp = (
            f"{res['predicted_error_code']} (gốc: {r['error_reason']})"
            if int(r["y_true"]) == 1
            else res["predicted_error_code"]
        )
        tbl.add_row(
            ex,
            f"{sid} (F{oof_fold})",
            gt_str,
            pr_str,
            f"{res['form_score_pct']:.1f}%",
            f"{res['p_hybrid']*100:.1f}%",
            err_disp,
            video_tag,
        )

    console.print(tbl)
    console.print(
        f"[bold green]Showcase hoàn tất:[/bold green] Dự đoán chính xác [bold]{n_ok}/{len(selected_rows)}[/bold] mẫu OOF"
        + (f" | Đã lưu [bold]{len(exported_videos)}[/bold] video HUD vào `{DEMO_OUT_DIR}`." if exported_videos else ".")
    )


def run_benchmark(args: argparse.Namespace, console: Console) -> None:
    """Đo tốc độ 100 lượt suy luận liên tiếp + tốc độ MediaPipe End-to-End và xuất JSON."""
    df_meta = pd.read_csv(TASK2_CSV_PATH, keep_default_na=False)
    sample_row = df_meta[df_meta["exercise"] == "squat"].iloc[0]
    sid = str(sample_row["segment_id"])
    npy_path = POSE_NPY_DIR / f"{sid}.npy"
    video_path = PROJECT_ROOT / str(sample_row["video_path"])
    pose_seq = np.load(npy_path)
    dur = float(sample_row["duration_sec"])

    predictor_1f = ForgAITask2Predictor(fold=args.fold or 0, ensemble_all_folds=False)
    predictor_5f = ForgAITask2Predictor(ensemble_all_folds=True)

    b_1f = predictor_1f.benchmark_latency(pose_seq, "squat", duration_sec=dur, n_runs=args.runs)
    b_5f = predictor_5f.benchmark_latency(pose_seq, "squat", duration_sec=dur, n_runs=args.runs)
    res_vid, _ = predictor_1f.predict_from_video(
        video_path=video_path,
        exercise_name="squat",
        start_frame=int(sample_row["start_frame"]),
        end_frame=int(sample_row["end_frame"]),
    )

    tbl = Table(
        title=f"FORGAI STEP 7 LATENCY BENCHMARK ({args.runs} LƯỢT LIÊN TIẾP TRÊN CPU, MẪU `{sid}` = {len(pose_seq)} FRAMES)",
        box=box.ROUNDED,
        border_style="cyan",
    )
    tbl.add_column("Cấu hình / Công đoạn", style="bold")
    tbl.add_column("Đặc trưng (ms/rep)", justify="right")
    tbl.add_column("Suy luận AI (ms/rep)", justify="right")
    tbl.add_column("Rule Engine (ms/rep)", justify="right")
    tbl.add_column("Tổng Lõi AI (ms/rep)", justify="right")
    tbl.add_column("p95 (ms/rep)", justify="right")
    tbl.add_column("TB ms / frame", justify="right")
    tbl.add_column("So với < 50ms/frame", justify="center")

    for b in (b_1f, b_5f):
        ok_tag = "[bold green]PASS[/bold green]" if b["per_frame_ai_ms"] < 50.0 else "[red]FAIL[/red]"
        tbl.add_row(
            b["mode"],
            f"{b['feature_extract_mean_ms']:.2f}",
            f"{b['model_inference_mean_ms']:.2f}",
            f"{b['rule_engine_mean_ms']:.2f}",
            f"{b['total_ai_mean_ms']:.2f}",
            f"{b['total_ai_p95_ms']:.2f}",
            f"{b['per_frame_ai_ms']:.4f}",
            ok_tag,
        )

    tm_v = res_vid["timings_ms"]
    e2e_ok = "[bold green]PASS[/bold green]" if tm_v["end_to_end_per_frame_ms"] < 50.0 else "[red]FAIL[/red]"
    tbl.add_row(
        "End-to-End Video (MediaPipe + 1-Fold AI)",
        f"{tm_v['feature_extract_ms']:.2f}",
        f"{tm_v['model_inference_ms']:.2f}",
        f"{tm_v['rule_engine_ms']:.2f}",
        f"{tm_v['end_to_end_total_ms']:.2f}",
        "-",
        f"{tm_v['end_to_end_per_frame_ms']:.2f}",
        e2e_ok,
    )
    console.print(tbl)

    payload = {
        "sample_segment_id": sid,
        "sample_frames": int(len(pose_seq)),
        "threshold_ms_per_frame": 50.0,
        "core_ai_single_fold": b_1f,
        "core_ai_five_fold_ensemble": b_5f,
        "end_to_end_video_single_fold": {
            "mediapipe_extract_ms": tm_v["mediapipe_extract_ms"],
            "mediapipe_per_frame_ms": tm_v["mediapipe_per_frame_ms"],
            "total_ai_ms": tm_v["total_ai_ms"],
            "end_to_end_total_ms": tm_v["end_to_end_total_ms"],
            "end_to_end_per_frame_ms": tm_v["end_to_end_per_frame_ms"],
            "fps_equivalent": round(1000.0 / max(tm_v["end_to_end_per_frame_ms"], 1e-6), 1),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = REPORTS_DIR / "step7_latency_benchmark.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[bold green]Đã lưu báo cáo Benchmark tại:[/bold green] {out_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForgAI Step 7 - Deployment & Inference Demo (rich UI + HUD Video)")
    parser.add_argument("--segment-id", type=str, default=None, help="Chạy thử 1 segment_id trong metadata")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn file video .mp4 bất kỳ")
    parser.add_argument("--exercise", type=str, default=None, choices=GROUP_A_EXERCISES, help="Tên bài tập khi truyền --video")
    parser.add_argument("--fold", type=int, default=None, choices=[0, 1, 2, 3, 4], help="Fold dùng suy luận đơn")
    parser.add_argument("--all-folds", action="store_true", help="Bật chế độ Ensemble cả 5 Folds")
    parser.add_argument("--out-video", type=str, default=None, help="Đường dẫn lưu video HUD xuất ra")
    parser.add_argument("--no-video", action="store_true", help="Chỉ hiển thị kết quả rich UI trên Terminal, không ghi file .mp4")
    parser.add_argument("--showcase", action="store_true", help="Chạy thử nghiệm 16 mẫu OOF tiêu biểu qua đủ 8 bài Nhóm A")
    parser.add_argument("--benchmark", action="store_true", help="Đo kiểm chuẩn độ trễ 100 lượt & xuất step7_latency_benchmark.json")
    parser.add_argument("--runs", type=int, default=100, help="Số lượt lặp lại khi chạy --benchmark")
    return parser.parse_args()


def main() -> None:
    console = _make_console()
    args = parse_args()
    if args.benchmark:
        run_benchmark(args, console)
    elif args.showcase:
        run_showcase(args, console)
    elif args.segment_id or args.video:
        run_single_demo(args, console)
    else:
        console.print("[yellow]Vui lòng chọn 1 chế độ: `--segment-id`, `--video ... --exercise ...`, `--showcase`, hoặc `--benchmark`.[/yellow]")


if __name__ == "__main__":
    main()
