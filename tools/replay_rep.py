"""
tools/replay_rep.py
Công cụ phát lại (Replay) các rep đã duyệt trong metadata/task2_assessment.csv
Giúp xem lại tiêu chuẩn cắt 1 rep hoàn chỉnh (Start Frame -> End Frame) của từng bài.
Hỗ trợ xem rep Đúng (correct), rep Sai (incorrect), hoặc toàn bộ (all).

Phím tắt:
- [Space]: Tạm dừng / Xem tiếp
- [N] hoặc [D]: Xem rep đã duyệt kế tiếp
- [P] hoặc [A]: Xem lại rep trước đó
- [Q] hoặc [ESC]: Thoát
"""

import os
import sys
import io
import argparse
import cv2
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def replay_reps(exercise: str, base_dir: str, label: str = "correct"):
    t2_path = os.path.join(base_dir, "metadata", "task2_assessment.csv")
    if not os.path.exists(t2_path):
        print(f"[LỖI] Không tìm thấy file: {t2_path}")
        return

    df_t2 = pd.read_csv(t2_path)
    clean_ex = exercise.strip().lower()
    clean_lbl = label.strip().lower()

    cond = (df_t2["exercise"].str.lower() == clean_ex)
    if clean_lbl in ["correct", "incorrect"]:
        cond = cond & (df_t2["form_label"].str.lower() == clean_lbl)

    matches = df_t2[cond].reset_index(drop=True)

    if len(matches) == 0:
        print(f"[THÔNG BÁO] Không tìm thấy rep '{clean_lbl}' nào cho bài tập: {exercise}")
        return

    lbl_display = "TẤT CẢ" if clean_lbl == "all" else ("ĐÚNG (CORRECT)" if clean_lbl == "correct" else "SAI (INCORRECT)")
    print("=" * 70)
    print(f"FORGAI REPLAY TOOL - PHÁT LẠI REP ĐÃ DUYỆT")
    print(f"Bài tập: {exercise.upper()} | Nhãn: {lbl_display} | Tổng số rep: {len(matches)}")
    print("Phím tắt: [Space] Pause/Play | [N] Rep tiếp | [P] Rep trước | [Q] Thoát")
    print("=" * 70)

    rep_idx = 0
    window_name = f"FORGAI - Replay: {exercise.upper()} ({lbl_display})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    while 0 <= rep_idx < len(matches):
        row = matches.iloc[rep_idx]
        raw_path = str(row["video_path"]).strip()
        v_path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
        s_frame = int(row["start_frame"])
        e_frame = int(row["end_frame"])
        dur = row["duration_sec"]
        seg_id = row["segment_id"]
        form_lbl = str(row["form_label"]).strip()
        err_reason = str(row.get("error_reason", "none")).strip()

        if not os.path.exists(v_path):
            print(f"[CẢNH BÁO] Không tìm thấy file video: {v_path}")
            rep_idx += 1
            continue

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"[LỖI] Không thể mở video: {v_path}")
            rep_idx += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay = max(1, int(1000.0 / fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
        curr_pos = s_frame
        is_paused = False

        print(f"\n-> Đang phát rep [{rep_idx + 1}/{len(matches)}]: {seg_id} ({form_lbl.upper()})")
        if form_lbl == "incorrect":
            print(f"   Mã lỗi kỹ thuật: {err_reason}")
        print(f"   Khung hình: {s_frame} -> {e_frame} | Thời lượng: {dur}s | FPS: {fps:.1f}")

        action_next = False
        while True:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                cv2.destroyAllWindows()
                return

            if not is_paused:
                if curr_pos > e_frame:
                    curr_pos = s_frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)

                ret, frame = cap.read()
                if not ret:
                    curr_pos = s_frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
                    continue

                curr_pos += 1
                disp = frame.copy()

                # Vẽ overlay thông tin
                h, w = disp.shape[:2]
                banner_color = (0, 0, 180) if form_lbl == "incorrect" else (20, 20, 20)
                cv2.rectangle(disp, (0, 0), (w, 85), banner_color, -1)
                
                title_tag = f"REP SAI [{rep_idx + 1}/{len(matches)}]" if form_lbl == "incorrect" else f"REP MAU [{rep_idx + 1}/{len(matches)}]"
                err_text = f" | Lỗi: {err_reason}" if form_lbl == "incorrect" else ""
                title = f"{title_tag}: {exercise.upper()}{err_text} (Frame {s_frame} -> {e_frame}, {dur}s)"
                cv2.putText(disp, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                
                sub = f"Progress: {curr_pos}/{e_frame} | [Space] Pause | [N] Next | [P] Prev | [Q] Quit"
                cv2.putText(disp, sub, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.imshow(window_name, disp)
            else:
                cv2.waitKey(30)

            key = cv2.waitKey(delay if not is_paused else 30) & 0xFF
            if key in [ord('q'), 27]: # Q or ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                is_paused = not is_paused
            elif key in [ord('n'), ord('d')]:
                rep_idx = min(len(matches) - 1, rep_idx + 1)
                action_next = True
                break
            elif key in [ord('p'), ord('a')]:
                rep_idx = max(0, rep_idx - 1)
                action_next = True
                break

        cap.release()
        if not action_next:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay các rep đã duyệt trong task2_assessment.csv")
    parser.add_argument("--exercise", type=str, default="chest fly machine", help="Tên bài tập cần xem lại rep mẫu")
    parser.add_argument("--label", type=str, default="correct", choices=["correct", "incorrect", "all"], help="Chọn lọc rep Đúng (correct), Sai (incorrect) hoặc Tất cả (all)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    replay_reps(args.exercise, project_root, args.label)
