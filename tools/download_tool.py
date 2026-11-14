"""
tools/download_tool.py
Công cụ tải và cắt video lỗi sai từ YouTube (hoặc YouTube Shorts) bằng yt-dlp và ffmpeg tích hợp.
Tự động cập nhật mẫu bổ sung vào metadata/task2_assessment.csv.

Nâng cấp kỹ thuật:
1. Trích xuất mã YouTube ID (11 ký tự) làm `video_id = YT_<youtube_id>` để các đoạn cắt từ cùng 1 link
   luôn có chung `video_id` -> Chống Data Leakage 100% khi chia tập bằng GroupShuffleSplit.
2. Bộ nhớ đệm (Cache) thông minh khi chạy Batch: Nếu nhiều dòng cắt từ cùng 1 link YouTube, chỉ tải
   video gốc 1 lần duy nhất rồi cắt nhiều đoạn, tiết kiệm 70-80% thời gian tải.
3. Bảo toàn toàn bộ các Sheet phụ (như Sheet "Tra cứu mã lỗi") và định dạng màu sắc trong file Excel
   khi cập nhật cột `status = done`.
"""

import os
import re
import sys
import io
import shutil
import hashlib
import argparse
import subprocess
import uuid
import pandas as pd
import imageio_ffmpeg
import yt_dlp

# Chống lỗi UnicodeEncodeError trên terminal Windows (cp1252)
if sys.stdout and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def extract_youtube_id(url: str) -> str:
    """
    Trích xuất mã YouTube Video ID (11 ký tự) từ URL để gán làm video_id gốc.
    Đảm bảo nhiều đoạn rep cắt từ cùng 1 video YouTube có chung video_id -> Chống Data Leakage.
    """
    url_clean = str(url).strip()
    patterns = [
        r"(?:v=|\/shorts\/|youtu\.be\/|\/embed\/|\/v\/|\/live\/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        match = re.search(pat, url_clean)
        if match:
            return match.group(1)
    # Fallback nếu link rút gọn đặc biệt: băm cố định 11 ký tự từ URL chuẩn hóa
    return hashlib.md5(url_clean.encode("utf-8")).hexdigest()[:11]


def parse_time_to_seconds(time_str: str) -> float:
    """Chuyển đổi chuỗi thời gian 'MM:SS' hoặc 'HH:MM:SS' hoặc 'SS' thành số giây float."""
    parts = str(time_str).strip().split(':')
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError(f"Định dạng thời gian không hợp lệ: {time_str}")
    except ValueError as e:
        raise ValueError(f"Không thể phân tích thời gian '{time_str}': {e}")


def update_excel_status_preserve_sheets(batch_file: str, df_row_idx: int, status_val: str):
    """
    Cập nhật giá trị cột 'status' tại dòng tương ứng trên Sheet đầu tiên của file Excel
    mà KHÔNG làm mất Sheet 2 ('Tra cứu mã lỗi') hay định dạng bảng.
    """
    import openpyxl

    while True:
        try:
            wb = openpyxl.load_workbook(batch_file)
            ws = wb.worksheets[0]

            # Tìm cột 'status' ở dòng tiêu đề (dòng 1)
            status_col_idx = None
            for col_i in range(1, ws.max_column + 1):
                cell_val = str(ws.cell(row=1, column=col_i).value or "").strip().lower()
                if cell_val == "status":
                    status_col_idx = col_i
                    break

            if status_col_idx is None:
                status_col_idx = ws.max_column + 1
                ws.cell(row=1, column=status_col_idx, value="status")

            # Dòng dữ liệu trong Excel bắt đầu từ dòng 2 (tương ứng df_row_idx = 0)
            excel_row = int(df_row_idx) + 2
            ws.cell(row=excel_row, column=status_col_idx, value=status_val)
            wb.save(batch_file)
            wb.close()
            break
        except PermissionError:
            print(f"\n[CẢNH BÁO] File Excel '{batch_file}' đang mở. Vui lòng đóng Excel để lưu trạng thái!")
            input(">> Bấm [Enter] sau khi đã đóng Excel: ")


def download_and_cut(url: str, start_time: str, end_time: str, exercise: str,
                     label: str, error_reason: str, base_dir: str,
                     keep_cache: bool = False):
    """Tải video từ YouTube (có hỗ trợ cache theo YouTube ID), cắt đoạn rep và cập nhật metadata."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    clean_ex = exercise.strip().lower()

    # Thư mục lưu trữ video bổ sung và thư mục cache tạm
    supp_dir = os.path.join(base_dir, "data_supplementary", clean_ex.replace(" ", "_"))
    cache_dir = os.path.join(base_dir, "data_supplementary", "_cache")
    os.makedirs(supp_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    t_start = parse_time_to_seconds(start_time)
    t_end = parse_time_to_seconds(end_time)
    dur = round(t_end - t_start, 2)

    if dur <= 0:
        raise ValueError(f"Thời gian kết thúc ({end_time}) phải lớn hơn thời gian bắt đầu ({start_time})")

    # Gắn video_id theo mã gốc YouTube để gom nhóm các đoạn cắt từ cùng 1 link (Chống Data Leakage)
    yt_id = extract_youtube_id(url)
    vid_id = f"YT_{yt_id}"

    # Tạo mã segment_id duy nhất cho từng đoạn cắt
    time_tag = f"{int(round(t_start))}s_{int(round(t_end))}s"
    short_uid = str(uuid.uuid4())[:6]
    seg_id = f"SEG_YT_{yt_id}_{time_tag}_{short_uid}"

    raw_temp_path = os.path.join(cache_dir, f"raw_{vid_id}.mp4")
    out_final_path = os.path.join(supp_dir, f"{seg_id}.mp4")

    if os.path.exists(raw_temp_path) and os.path.getsize(raw_temp_path) > 0:
        print(f"\n[1/3] Tái sử dụng video gốc trong bộ nhớ đệm ({vid_id}) — không cần tải lại!")
    else:
        print(f"\n[1/3] Đang tải video từ: {url} (Mã video: {vid_id}) ...")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'ffmpeg_location': ffmpeg_exe,
            'outtmpl': raw_temp_path,
            'extractor_args': {'youtube': {'player_client': ['android', 'ios', 'web']}},
            'quiet': True,
            'no_warnings': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if not os.path.exists(raw_temp_path):
            raise FileNotFoundError("Tải video thất bại, không tìm thấy file tạm.")

    print(f"[2/3] Đang cắt đoạn rep: {start_time} -> {end_time} ({dur}s) bằng ffmpeg ...")
    cmd = [
        ffmpeg_exe, "-y",
        "-ss", str(t_start),
        "-i", raw_temp_path,
        "-t", str(dur),
        "-r", "30",
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",
        out_final_path
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        if not keep_cache and os.path.exists(raw_temp_path):
            os.remove(raw_temp_path)
        raise RuntimeError(f"FFmpeg cắt video thất bại: {res.stderr.decode('utf-8', errors='ignore')}")

    # Nếu chạy đơn lẻ (không phải batch), xóa luôn file tạm sau khi cắt
    if not keep_cache and os.path.exists(raw_temp_path):
        os.remove(raw_temp_path)

    print(f"[3/3] Đã lưu video thành công: {out_final_path}")

    # Cập nhật metadata Task 2 với đầy đủ 12 cột chuẩn
    t2_path = os.path.join(base_dir, "metadata", "task2_assessment.csv")
    cols_order = [
        'segment_id', 'video_id', 'exercise', 'source',
        'form_label', 'error_reason', 'video_path',
        'start_frame', 'end_frame', 'duration_sec', 'num_frames', 'fps'
    ]
    df_t2 = pd.read_csv(t2_path) if os.path.exists(t2_path) else pd.DataFrame(columns=cols_order)

    fps = 30.0
    num_frames = max(1, int(round(dur * fps)))
    new_row = {
        "segment_id": seg_id,
        "video_id": vid_id,
        "exercise": clean_ex,
        "source": "YouTube_supplementary",
        "form_label": label.strip().lower(),
        "error_reason": error_reason.strip() if error_reason else "none",
        "video_path": out_final_path,
        "start_frame": 0,
        "end_frame": num_frames - 1,
        "duration_sec": dur,
        "num_frames": num_frames,
        "fps": fps
    }

    df_t2 = pd.concat([df_t2, pd.DataFrame([new_row])], ignore_index=True)
    final_cols = [c for c in cols_order if c in df_t2.columns]

    while True:
        try:
            df_t2[final_cols].to_csv(t2_path, index=False, encoding='utf-8-sig')
            break
        except PermissionError:
            print(f"\n[CẢNH BÁO] File '{t2_path}' đang bị khóa bởi Excel. Vui lòng đóng Excel lại!")
            input(">> Bấm [Enter] sau khi đã đóng Excel: ")

    print(f"-> Đã ghi nhận mẫu: {seg_id} (video_id={vid_id}) | Nhãn: {label.upper()} ({error_reason})")
    print(f"-> Tổng số mẫu hiện có trong Task 2: {len(df_t2)}")
    return True


def process_batch_file(batch_file: str, base_dir: str):
    """
    Xử lý danh sách link từ file Excel (.xlsx) hoặc file CSV (.csv).
    Tự động tái sử dụng video gốc nếu trùng URL, và cập nhật cột 'status' thành 'done'
    mà vẫn giữ nguyên Sheet 'Tra cứu mã lỗi' trong Excel.
    """
    if not os.path.exists(batch_file):
        alt_path = os.path.join(base_dir, batch_file)
        if os.path.exists(alt_path):
            batch_file = alt_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file danh sách: {batch_file}")

    print(f"\nBắt đầu xử lý file danh sách tải hàng loạt: {batch_file}")
    is_excel = batch_file.lower().endswith(('.xlsx', '.xls'))

    if is_excel:
        df_batch = pd.read_excel(batch_file, sheet_name=0)
    else:
        df_batch = pd.read_csv(batch_file)

    # Chuẩn hóa tên cột
    col_map = {c: c.strip().lower() for c in df_batch.columns}
    df_batch.rename(columns=col_map, inplace=True)

    # Đảm bảo các cột tối thiểu
    req_cols = ['url', 'start_time', 'end_time', 'exercise']
    for rc in req_cols:
        if rc not in df_batch.columns:
            raise ValueError(
                f"File batch thiếu cột bắt buộc: '{rc}'. "
                f"Cần có: url, start_time, end_time, exercise, form_label, error_reason"
            )

    if 'form_label' not in df_batch.columns:
        df_batch['form_label'] = 'incorrect'
    if 'error_reason' not in df_batch.columns:
        df_batch['error_reason'] = 'unspecified_fault'
    if 'status' not in df_batch.columns:
        df_batch['status'] = ''

    success = 0
    total = len(df_batch)
    cache_dir = os.path.join(base_dir, "data_supplementary", "_cache")

    try:
        for idx, row in df_batch.iterrows():
            status = str(row.get('status', '')).strip().lower()
            if status in ['done', 'downloaded', 'success', 'ok']:
                print(f"\n[{idx+1}/{total}] Dòng {idx+1} đã tải trước đó (status='{status}'). Bỏ qua.")
                continue

            url = str(row['url']).strip()
            start = str(row['start_time']).strip()
            end = str(row['end_time']).strip()
            ex = str(row['exercise']).strip()
            lbl = str(row.get('form_label', 'incorrect')).strip().lower()
            reason = str(row.get('error_reason', 'unspecified_fault')).strip()

            if not url or url == 'nan' or url.startswith('#'):
                continue

            print(f"\n[{idx+1}/{total}] Đang xử lý: {ex.upper()} | {url} ({start} -> {end}) | Lỗi: {reason}")
            try:
                download_and_cut(url, start, end, ex, lbl, reason, base_dir, keep_cache=True)
                df_batch.at[idx, 'status'] = 'done'
                success += 1

                # Lưu trạng thái ngay lập tức mà không làm mất Sheet 2 của Excel
                if is_excel:
                    update_excel_status_preserve_sheets(batch_file, idx, 'done')
                else:
                    df_batch.to_csv(batch_file, index=False, encoding='utf-8-sig')
            except Exception as e:
                err_msg = f"error: {str(e)[:50]}"
                print(f"[LỖI DÒNG {idx+1}] Tải thất bại: {e}")
                df_batch.at[idx, 'status'] = err_msg
                try:
                    if is_excel:
                        update_excel_status_preserve_sheets(batch_file, idx, err_msg)
                    else:
                        df_batch.to_csv(batch_file, index=False, encoding='utf-8-sig')
                except Exception:
                    pass
    finally:
        # Dọn dẹp thư mục cache tạm sau khi chạy xong toàn bộ batch
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(f"HOÀN TẤT BATCH DOWNLOAD. Đã tải thành công: {success}/{total} đoạn video.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Công cụ tải và cắt video bổ sung cho ForgAI (yt-dlp + ffmpeg)")
    parser.add_argument("--url", type=str, help="Đường dẫn video YouTube hoặc YouTube Shorts")
    parser.add_argument("--start", type=str, help="Thời gian bắt đầu (vd: '00:15' hoặc '15')")
    parser.add_argument("--end", type=str, help="Thời gian kết thúc (vd: '00:19' hoặc '19')")
    parser.add_argument("--exercise", type=str, help="Tên bài tập (vd: 'leg extension', 'tricep dips')")
    parser.add_argument("--label", type=str, default="incorrect", choices=["correct", "incorrect"],
                        help="Nhãn kỹ thuật (mặc định 'incorrect')")
    parser.add_argument("--reason", type=str, default="", help="Lý do lỗi sai (vd: 'partial_rom', 'knee_valgus')")
    parser.add_argument("--batch", type=str, help="Đường dẫn file Excel (.xlsx) hoặc file CSV chứa danh sách để tải hàng loạt")

    args = parser.parse_args()
    base_dir = r"D:\datasets-gym\FORGAI"

    if args.batch:
        process_batch_file(args.batch, base_dir)
    elif args.url and args.start and args.end and args.exercise:
        download_and_cut(args.url, args.start, args.end, args.exercise, args.label, args.reason, base_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
