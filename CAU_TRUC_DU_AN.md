# 🏛️ CẤU TRÚC CHUẨN MỰC TOÀN DIỆN DỰ ÁN: FORGAI

> **Quy chuẩn:** Cấu trúc dự án theo chuẩn Kỹ thuật phần mềm & AI Quốc tế (Academic & Production Standard).  
> **Cam kết:** Đây là kiến trúc **cố định và thống nhất từ đầu đến cuối dự án**, được dọn dẹp sạch sẽ, không có file rác hay thư mục thừa.

---

## 🌳 PHẦN 1: SƠ ĐỒ CÂY THƯ MỤC CHÍNH THỨC CỦA DỰ ÁN

Toàn bộ dự án nằm tại thư mục `D:\datasets-gym\FORGAI\`:

```text
D:\datasets-gym\FORGAI\
│
├── 📄 TRANG_THAI_DU_AN.md        # Nhật ký theo dõi tiến độ theo 7 bước AI SIC của Thầy
├── 📄 CAU_TRUC_DU_AN.md          # File này (Sổ tay cấu trúc chuẩn & giải thích dữ liệu)
├── 📄 run_web.bat                # Script khởi chạy nhanh ForgAI Web Studio bằng 1 cú nhấp chuột (Windows)
│
├── 📁 configs/                   # [THIẾT LẬP THAM SỐ MẪU] Cấu hình chung & Model MediaPipe nội bộ
│   ├── config.py                 # Tham số tập trung 8 bài tập Nhóm A, lấy mẫu T=32, Split, Seed
│   └── pose_landmarker_full.task # Trọng số MediaPipe Pose lưu cục bộ
│
├── 📁 metadata/                  # [CHỈ LƯU 1 FILE CHUẨN DUY NHẤT]
│   └── task2_assessment.csv      # Danh bạ 521 mẫu (8 bài tập) đã chuẩn hóa tên & đường dẫn nội bộ
│
├── 📁 data_videos/               # [KHO VIDEO 1-REP ĐỘC LẬP] 521 video chuẩn hóa theo từng bài tập
│   ├── chest_fly_machine/        # chest_fly_machine_correct_001.mp4 ...
│   ├── hip_thrust/               # hip_thrust_correct_001.mp4 ...
│   ├── leg_extension/            # leg_extension_correct_001.mp4 ...
│   ├── leg_raises/               # leg_raises_correct_001.mp4 ...
│   ├── pull_up/                  # pull_up_correct_001.mp4 ...
│   ├── push_up/                  # push_up_correct_001.mp4 ...
│   ├── squat/                    # squat_correct_001.mp4 ...
│   └── t_bar_row/                # t_bar_row_correct_001.mp4 ...
│
├── 📁 data_supplementary/
│   └── _cache/                   # Kho video gốc YouTube dự phòng (giữ lại để re-cut khi cần)
│
├── 📁 data_processed/            # [DỮ LIỆU ĐÃ TRÍCH XUẤT] Lưu file khung xương .npy siêu nhẹ
│   └── pose/                     # 521 file khung xương (squat_correct_001.npy ...)
│
├── 📁 tools/                     # [BỘ CÔNG CỤ CLI & HỖ TRỢ]
│   ├── download_tool.py          # Công cụ tải & cắt rep YouTube tự động (yt-dlp + ffmpeg, gắn YT_<id>)
│   ├── replay_rep.py             # Phát lại kiểm tra từng rep trong metadata/task2_assessment.csv
│   ├── visualize_task2.py        # Vẽ biểu đồ thống kê Trước/Sau bổ sung dữ liệu & khung xương 13 khớp
│   ├── demo_inference.py         # CLI Bước 7 (Terminal UI rich, HUD video, --showcase 16 mẫu, --benchmark)
│   └── realtime_demo.py          # Giao diện OpenCV + PIL Realtime 30 FPS (Webcam hoặc file video)
│
├── 📁 web/                       # [GIAO DIỆN WEB STUDIO] Tách biệt hoàn toàn khỏi thư mục CLI tools/
│   ├── __init__.py               # Đánh dấu gói Python chuẩn (hỗ trợ python -m web.app / import web.app)
│   ├── app.py                    # FastAPI Backend cho ForgAI Web Studio (chấm upload, MJPEG webcam, H.264)
│   └── static/                   # Frontend HTML/CSS/JS cho ForgAI Web Studio (index.html, styles.css, app.js)
│
├── 📁 src/                       # [MÃ NGUỒN LÕI - CORE LIBRARY] Các hàm và lớp tái sử dụng
│   ├── constants.py              # 13 khớp giải phẫu, bản đồ lật gương SWAP_MAP, từ điển 32 mã lỗi
│   ├── extractor.py              # Trích xuất MediaPipe Pose 13 khớp ra file .npy
│   ├── features.py               # Hip-centering, Torso normalization, 117-d Mix, Điểm Đáy/Đỉnh, Golden-Rep
│   ├── dataset.py                # Dynamic Strided Sampling (T=32), Augmentation x6, GroupSplit chống rò rỉ
│   ├── models.py                 # BiomechanicalRF (103-d E.3 Pure-AI), 1D-CNN + Bi-LSTM, Transformer, Fixed-Simplex Ensemble
│   ├── rules.py                  # Bộ luật cơ sinh học giải thích mã lỗi tiếng Việt
│   ├── inference.py              # ForgAITask2Predictor đóng gói suy luận 1-Fold / 5-Fold & tô khớp lỗi HUD
│   └── realtime.py               # MediaPipeRealtimeTracker, CycleRepCounter, ForgAIRealtimeScorer
│
├── 📁 experiments/               # [NƠI CHẠY THỰC NGHIỆM] Các file Python trực tiếp chạy
│   ├── train_task2.py            # Huấn luyện 5-Fold CV + Holdout cho Task 2 (hỗ trợ --reuse-dl)
│   └── evaluate.py               # So sánh mô hình, chấm điểm từng bài, vẽ Confusion Matrix & xuất báo cáo
│
└── 📁 outputs/                   # [KẾT QUẢ ĐẦU RA] Nơi lưu mô hình, báo cáo và video demo
    ├── checkpoints/              # Trọng số mô hình tốt nhất (.pt, .pkl) của 5 Folds
    ├── demo/                     # Video .mp4 minh họa kèm HUD tiếng Việt & khung xương (Bước 7)
    ├── web_sessions/             # Thư mục tạm chứa video H.264 & kết quả phiên tải lên trên Web Studio
    └── reports/                  # Báo cáo, benchmark độ trễ & thư mục figures/ chứa biểu đồ trực quan hóa
```

---

## 🎯 PHẦN 2: PHÂN CÔNG NHIỆM VỤ TỪNG THƯ MỤC

| Thư mục | Vai trò | Vì sao bắt buộc phải có? |
| :--- | :--- | :--- |
| **`configs/`** | Cấu hình & Model Pose | Quản lý tập trung toàn bộ tham số 8 bài tập, seed, chia tập và chứa file `pose_landmarker_full.task` nội bộ. |
| **`metadata/`** | **1 file danh bạ chuẩn duy nhất** | Chỉ lưu duy nhất `task2_assessment.csv` (521 mẫu của 8 bài tập, tên chuẩn hóa dễ hiểu, không lưu cột split tĩnh). |
| **`data_videos/`** | Kho Video 1-rep độc lập | Chứa toàn bộ 521 video đã cắt chuẩn 1 rep (`data_videos/<bài_tập>/<bài_tập>_<correct/incorrect>_<001>.mp4`), giúp dự án độc lập 100% (`603.63 MB`). |
| **`data_supplementary/_cache/`** | Kho video gốc dự phòng | Giữ nguyên các video gốc YouTube tải về để có thể re-cut lại bất cứ lúc nào nếu cần. |
| **`data_processed/pose/`** | Kho lưu khung xương | Chứa 521 file `.npy` tọa độ 13 khớp. Nhẹ tênh (`12.61 MB`), giúp train nhanh gấp 100 lần. |
| **`tools/`** | Công cụ CLI tải, xem, vẽ & demo | Chứa `download_tool.py` (tải & cắt rep YouTube), `replay_rep.py` (xem lại rep), `visualize_task2.py` (biểu đồ), `demo_inference.py` (CLI Bước 7 + showcase + benchmark) và `realtime_demo.py` (demo Realtime Webcam/Video 30 FPS). |
| **`web/`** | Giao diện Web Studio | Gói web độc lập chứa `app.py` (FastAPI backend, H.264, MJPEG webcam) và `static/` (`index.html`, `styles.css`, `app.js`), khởi chạy qua `run_web.bat`. |
| **`src/`** | Bộ máy xử lý cốt lõi | Chứa toàn bộ logic trích xuất, tiền xử lý `117-d`, mô hình AI (`100-d/103-d Pure-AI`), bộ luật cơ sinh học giải thích lỗi và engine Suy luận/Realtime (`inference.py`, `realtime.py`). |
| **`experiments/`** | Phòng điều khiển thực nghiệm | Chứa `train_task2.py` (`--reuse-dl`) và `evaluate.py` để huấn luyện 5-Fold CV và đối sánh các mô hình. |
| **`outputs/`** | Thành quả nghiên cứu | Chứa file trọng số mô hình tốt nhất (`checkpoints/`), video HUD minh họa (`demo/`) và báo cáo + biểu đồ (`reports/`). |

---

## 📊 PHẦN 3: GIẢI THÍCH CHI TIẾT FILE METADATA CHUẨN DUY NHẤT (`metadata/task2_assessment.csv`)

Thư mục `metadata/` chỉ lưu **1 file chuẩn duy nhất** với **521 dòng** (8 bài tập Nhóm A), định danh theo chuẩn quốc tế:

* **`segment_id`**: Mã định danh chuẩn duy nhất của từng rep (`<exercise>_<correct|incorrect>_<NNN>`, ví dụ: `squat_correct_001`, `squat_incorrect_005`).
* **`video_id`**: Mã video nguồn gốc / người tập (`<exercise>_src_<NNN>`, ví dụ: `squat_src_001`). Các đoạn rep cắt từ cùng 1 video gốc hoặc cùng 1 người tập (kể cả khi xuất hiện chéo giữa GYMED / EFA / YouTube) mang chung 1 mã `src` để gom nhóm chống rò rỉ dữ liệu 100% (tổng cộng 371 nhóm độc lập).
* **`exercise`**: Tên bài tập trong 8 bài Nhóm A (`pull up`, `push-up`, `leg extension`, `squat`, `chest fly machine`, `hip thrust`, `t bar row`, `leg raises`).
* **`source`**: Nguồn gốc video được chuẩn hóa gọn về 3 giá trị (`EFA`, `GYMED`, `YouTube`) phục vụ đối chiếu Trước & Sau bổ sung dữ liệu.
* **`form_label`**: Nhãn kỹ thuật: **`correct`** (Đúng — 330 mẫu) hoặc **`incorrect`** (Sai — 191 mẫu).
* **`error_reason`**: Mã lỗi chi tiết chuẩn hóa cho mẫu Sai (`partial_rom`, `knee_valgus`, `waist_sagging`...), hoặc `none` cho mẫu Đúng.
* **`video_path`**: Đường dẫn tương đối bên trong dự án (`data_videos/<exercise>/<segment_id>.mp4`), giúp dự án chạy độc lập trên mọi máy.
* **`start_frame`**, **`end_frame`**: Vì video trong `data_videos/` đã được cắt chuẩn đúng 1 rep duy nhất, mọi mẫu đều bắt đầu tại `0` và kết thúc tại `num_frames - 1`.
* **`duration_sec`**, **`num_frames`**, **`fps`**: Thời lượng (giây), tổng số khung hình thực tế và tốc độ khung hình của rep.

---

## 🔄 PHẦN 4: CHỐNG RÒ RỈ DỮ LIỆU ĐỘNG (KHÔNG HARDCODE `splits/`)

Code huấn luyện chia tập động ngay trong RAM bằng `StratifiedGroupKFold` (5-Fold) và Holdout Split dựa trên cột `video_id` (`*_src_*`), đảm bảo mọi đoạn cắt từ cùng 1 video nguồn không bao giờ bị chia tách giữa tập Học (Train) và tập Kiểm tra/Thi (Val/Test).
