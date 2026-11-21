# ForgAI — Hệ thống AI Phân tích & Chấm điểm Kỹ thuật Tập Gym

**ForgAI** là hệ thống đánh giá kỹ thuật tập Gym (**Form Assessment**) và phản hồi cơ sinh học giải thích được (**Explainable Biomechanical Feedback**) cho **8 bài tập Nhóm A**:
`squat`, `push-up`, `pull up`, `leg extension`, `chest fly machine`, `hip thrust`, `t bar row`, `leg raises`.

---

## Kiến trúc & Điểm nổi bật

1. **Chuẩn hóa khung xương 13 khớp (`src/extractor.py`, `src/features.py`):**
   - Dùng MediaPipe Pose (`configs/pose_landmarker_full.task`) trích xuất 13 khớp động học cốt lõi (`[wx, wy, wz, ix, iy, vis]`), nội suy frame bị che khuất, chuẩn hóa gốc tọa độ 3D về trung điểm hông (*Hip-midpoint centering*) và tỷ lệ theo chiều dài thân (*Torso normalization*).
   - Kết hợp chuỗi **120-d** (117-d Biomechanical Mix + 3 kênh vị trí pha) với **Dynamic Strided Sampling ($T = 32$)** và **Phase-Aligned Golden Prototype**.
   - Bổ sung đặc trưng **thời gian thực** (`rep_duration_sec`, `ecc_duration_sec`, vận tốc/gia tốc theo `°/s` & `°/s²`) và **6 đặc trưng dao động 2D tương đối** hông/cổ chân so với cổ tay bất biến với phép lật gương (`103 -> 111` chiều bảng).
2. **Mô hình Hybrid Ensemble (`src/models.py`):**
   - **BiomechanicalRandomForest (`114-d E.3+`)**: Đặc trưng bảng cơ sinh học `111-d` + 3 điểm tổng hợp từ `RuleEngine` chuẩn hóa theo mẫu Đúng từng bài.
   - **Exercise-Conditioned CNN-BiLSTM + FiLM** & **Transformer + FiLM**: Kết hợp chuỗi 32 frame (`120-d`) với đặc trưng bảng `111-d` (chuẩn hóa `0.65 * z_ex + 0.35 * z_global`).
   - **HybridFormEnsemble (`0.50 RF + 0.25 CNN-BiLSTM + 0.25 Transformer`)**, bật mặc định Hội đồng 5-Fold cho luồng phân tích video tải lên trên Web Studio.
3. **Bộ luật Cơ sinh học (`src/rules.py`):**
   - Phủ **32/32 mã lỗi / 50 cặp `(bài tập, lỗi)`** chi tiết bằng tiếng Việt kèm bằng chứng đo đạc (góc khớp, tốc độ `°/s`, thời lượng `s`, độ vung 2D) và gợi ý chỉnh sửa tư thế.
4. **Zero Data Leakage (`src/dataset.py`):**
   - Chia tập động theo `StratifiedGroupKFold` gom nhóm theo `video_id` (**`561` mẫu** [`330 Đúng / 231 Sai`] / **`405` nhóm độc lập**, `0` rò rỉ giữa các Fold).

---

## Kết quả Đánh giá (5-Fold Stratified Group CV, $N = 561$)

| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `BiomechanicalRF (114-d E.3+)` | 76.47 ± 2.58% | 76.16 ± 2.58% | 75.91 ± 2.61% | 74.44 ± 3.37% | **0.843 ± 0.027** |
| `CNN-BiLSTM + FiLM` | 72.38 ± 4.39% | 72.93 ± 3.96% | 72.08 ± 4.28% | 76.17 ± 4.18% | 0.810 ± 0.026 |
| `Transformer + FiLM` | 70.59 ± 3.61% | 71.24 ± 3.51% | 70.33 ± 3.55% | 74.90 ± 4.67% | 0.792 ± 0.037 |
| **`Hybrid Ensemble (0.50/0.25/0.25)`** | **77.01 ± 3.02%** | **76.94 ± 2.75%** | **76.55 ± 2.94%** | **76.61 ± 2.47%** | 0.835 ± 0.030 |

- **BiomechanicalRuleEngine (trên 231 mẫu Sai):** Top-1 Recall = **42.4%**, Top-3 Recall = **82.7%** (tỷ lệ rep Đúng không bị báo lỗi nhầm = **77.3%**).
- **Tốc độ suy luận CPU:** `1.45 ms/frame` (`82.53 ms/rep` 1-Fold), `7.94 ms/frame` (`452.86 ms/rep` 5-Fold Ensemble), End-to-End Video `31.81 ms/frame` (**`31.4 FPS`**).

---

## Hướng dẫn Cài đặt & Chạy nhanh

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy ForgAI Web Studio (Giao diện Upload Video, Webcam Realtime & Dashboard)
- Trên Windows: nhấp đúp [`run_web.bat`](run_web.bat) hoặc chạy:
```bash
python -m web.app
```
Mở trình duyệt tại `http://127.0.0.1:8000`.

### 3. Huấn luyện & Đánh giá lại 5-Fold CV
```bash
python experiments/train_task2.py --folds all
python experiments/evaluate.py
```

---

## Cấu trúc Thư mục

```text
FORGAI/
├── configs/            # Cấu hình tham số & pose_landmarker_full.task
├── metadata/           # task2_assessment.csv (561 mẫu, 405 nhóm video_id)
├── data_processed/pose/# Dữ liệu khung xương .npy đã trích xuất (561 file)
├── src/                # Thư viện lõi (extractor, features, dataset, models, rules, inference, realtime)
├── experiments/        # train_task2.py, evaluate.py
├── tools/              # download_tool.py, replay_rep.py, visualize_task2.py, demo_inference.py, realtime_demo.py
├── web/                # FastAPI Web Studio (app.py + static/)
└── outputs/
    ├── checkpoints/    # Trọng số 5-Fold CV (.pt, .pkl)
    └── reports/        # Báo cáo đánh giá & biểu đồ figures/
```

> **Lưu ý về dữ liệu video:** Các thư mục video nặng (`data_videos/`, `data_supplementary/_cache/`, `outputs/demo/`) được loại trừ khỏi Git qua `.gitignore`. Các file `.npy` trong `data_processed/pose/` và trọng số trong `outputs/checkpoints/` đã được đính kèm sẵn để có thể chạy Web Studio, huấn luyện và đánh giá ngay sau khi clone.
