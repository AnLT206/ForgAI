# ForgAI — Hệ thống AI Phân tích & Chấm điểm Kỹ thuật Tập Gym

**ForgAI** là hệ thống đánh giá kỹ thuật tập Gym (**Form Assessment**) và phản hồi cơ sinh học giải thích được (**Explainable Biomechanical Feedback**) cho **8 bài tập Nhóm A**:
`squat`, `push-up`, `pull up`, `leg extension`, `chest fly machine`, `hip thrust`, `t bar row`, `leg raises`.

---

## Kiến trúc & Điểm nổi bật

1. **Chuẩn hóa khung xương 13 khớp (`src/extractor.py`, `src/features.py`):**
   - Dùng MediaPipe Pose (`configs/pose_landmarker_full.task`) trích xuất 13 khớp động học cốt lõi, nội suy frame bị che khuất, chuẩn hóa gốc tọa độ về trung điểm hông (*Hip-midpoint centering*) và tỷ lệ theo chiều dài thân (*Torso normalization*).
   - Vector chuỗi **120-d** (117-d Biomechanical Mix + 3 kênh vị trí pha) với **Dynamic Strided Sampling ($T = 32$)** và **Phase-Aligned Golden Prototype**.
2. **Mô hình Hybrid Ensemble (`src/models.py`):**
   - **BiomechanicalRandomForest (`106-d E.3+`)**: Đặc trưng bảng cơ sinh học + khoảng cách Golden-Rep + 3 đặc trưng không thứ nguyên + điểm luật chuẩn hóa nội bài.
   - **Exercise-Conditioned CNN-BiLSTM + FiLM** & **Transformer + FiLM**: Kết hợp chuỗi 32 frame với đặc trưng bảng `103-d` (chuẩn hóa `0.65 * z_ex + 0.35 * z_global`).
   - **HybridFormEnsemble (`0.50 RF + 0.25 CNN-BiLSTM + 0.25 Transformer`)**.
3. **Bộ luật Cơ sinh học (`src/rules.py`):**
   - Phủ **32/32 mã lỗi** chi tiết bằng tiếng Việt kèm bằng chứng đo đạc (góc khớp/tỷ lệ thân) và gợi ý chỉnh sửa tư thế.
4. **Zero Data Leakage (`src/dataset.py`):**
   - Chia tập động theo `StratifiedGroupKFold` gom nhóm theo `video_id` (`521` mẫu / `371` nhóm độc lập, `0` rò rỉ giữa các Fold).

---

## Kết quả Đánh giá (5-Fold Stratified Group CV, $N = 521$)

| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `BiomechanicalRF (106-d E.3+)` | 71.99 ± 3.23% | 69.64 ± 3.60% | 69.60 ± 3.46% | 60.81 ± 8.72% | 0.769 ± 0.045 |
| `CNN-BiLSTM + FiLM` | 69.87 ± 2.00% | 70.13 ± 1.95% | 68.87 ± 1.90% | 71.18 ± 4.27% | 0.762 ± 0.024 |
| `Transformer + FiLM` | 67.79 ± 5.09% | 67.39 ± 4.43% | 66.51 ± 4.81% | 65.97 ± 4.08% | 0.747 ± 0.037 |
| **`Hybrid Ensemble (0.50/0.25/0.25)`** | **73.52 ± 2.73%** | **72.27 ± 2.38%** | **71.90 ± 2.57%** | **67.58 ± 3.48%** | **0.777 ± 0.035** |

- **BiomechanicalRuleEngine (trên 191 mẫu Sai):** Top-1 Recall = **47.6%**, Top-3 Recall = **81.2%**.
- **Tốc độ suy luận CPU:** `1.46 ms/frame` (`82.99 ms/rep` 1-Fold), `7.71 ms/frame` (5-Fold Ensemble).

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
├── metadata/           # task2_assessment.csv (521 mẫu, 371 nhóm video_id)
├── data_processed/pose/# Dữ liệu khung xương .npy đã trích xuất (521 file)
├── src/                # Thư viện lõi (extractor, features, dataset, models, rules, inference, realtime)
├── experiments/        # train_task2.py, evaluate.py
├── tools/              # download_tool.py, replay_rep.py, visualize_task2.py, demo_inference.py, realtime_demo.py
├── web/                # FastAPI Web Studio (app.py + static/)
└── outputs/
    ├── checkpoints/    # Trọng số 5-Fold CV (.pt, .pkl)
    └── reports/        # Báo cáo đánh giá & biểu đồ figures/
```

> **Lưu ý về dữ liệu video:** Các thư mục video nặng (`data_videos/`, `data_supplementary/_cache/`, `outputs/demo/`) được loại trừ khỏi Git qua `.gitignore`. Các file `.npy` trong `data_processed/pose/` và trọng số trong `outputs/checkpoints/` đã được đính kèm sẵn để có thể chạy Web Studio, huấn luyện và đánh giá ngay sau khi clone.
