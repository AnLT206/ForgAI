# -*- coding: utf-8 -*-
"""
configs/config.py
Cấu hình trung tâm cho dự án ForgAI - Tập trung Task 2 (Nhóm A - 8 bài tập).
"""

from pathlib import Path

# ==============================================================================
# 1. ĐƯỜNG DẪN THƯ MỤC CHUẨN CỦA DỰ ÁN
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "metadata"
TASK2_CSV_PATH = METADATA_DIR / "task2_assessment.csv"

DATA_VIDEOS_DIR = PROJECT_ROOT / "data_videos"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
POSE_NPY_DIR = DATA_PROCESSED_DIR / "pose"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Đường dẫn model MediaPipe Pose Landmarker lưu nội bộ trong configs/
POSE_MODEL_PATH = PROJECT_ROOT / "configs" / "pose_landmarker_full.task"

# ==============================================================================
# 2. DANH SÁCH 8 BÀI TẬP NHÓM A (ĐÃ ĐỦ CHỈ TIÊU ĐÚNG & SAI - 493 VIDEO)
# ==============================================================================
GROUP_A_EXERCISES = [
    "pull up",
    "push-up",
    "leg extension",
    "squat",
    "chest fly machine",
    "hip thrust",
    "t bar row",
    "leg raises",
]

EXERCISE_TO_ID = {ex: idx for idx, ex in enumerate(GROUP_A_EXERCISES)}
ID_TO_EXERCISE = {idx: ex for idx, ex in enumerate(GROUP_A_EXERCISES)}
NUM_EXERCISES = len(GROUP_A_EXERCISES)  # 8 bài tập

# ==============================================================================
# 3. THAM SỐ TIỀN XỬ LÝ KHUNG XƯƠNG & LẤY MẪU (THEO SKELGYM & GIẢNG VIÊN)
# ==============================================================================
NUM_JOINTS = 13                 # 13 khớp động học chính (RAW_POINTS_13)
COORD_DIM = 39                  # 13 khớp x 3 tọa độ (X, Y, Z)
ANGLE_PAIRS_DIM = 78            # Tổ hợp C(13, 2) = 78 cặp góc nghiêng 3D
FEATURE_DIM = 117               # 39 tọa độ + 78 góc = 117 chiều (117-d Biomechanical Mix)

NUM_FRAMES = 32                 # Số mốc thời gian chuẩn trên 1 rep (Dynamic Strided Sampling T=32)
MIN_VISIBILITY_THRESHOLD = 0.20 # Ngưỡng độ rõ dưới mức này được coi là bị che khuất cần nội suy

# Tham số Tăng cường dữ liệu an toàn cho Task 2 (Chỉ bật trên tập Train):
# 3 pha dịch điểm bắt đầu lấy frame nhảy cóc x 2 trạng thái lật gương Trái-Phải = x6 lần
TRAIN_STRIDE_OFFSETS = 3
ENABLE_HORIZONTAL_FLIP = True

# ==============================================================================
# 4. THAM SỐ CHIA TẬP (CHỐNG RÒ RỈ DỮ LIỆU - ZERO DATA LEAKAGE) & HUẤN LUYỆN
# ==============================================================================
RANDOM_SEED = 42
N_SPLITS = 5                    # 5-Fold StratifiedGroupKFold (gom theo video_id)
HOLDOUT_TEST_FOLD = 0           # Fold 0 được cố định làm tập Holdout Test độc lập
VAL_RATIO_WITHIN_TRAIN = 0.15   # Tách 15% từ phần Train làm tập Validation để chọn mô hình

# Siêu tham số mô hình Deep Learning (Thiết kế gọn nhẹ chống Overfitting trên 493 mẫu)
BATCH_SIZE = 32
LEARNING_RATE = 4e-4
WEIGHT_DECAY = 3e-2
EPOCHS = 80
EARLY_STOPPING_PATIENCE = 18
DROPOUT_RATE = 0.42
LABEL_SMOOTHING = 0.06
TRAIN_NOISE_STD = 0.05
MIN_RF_ENSEMBLE_WEIGHT = 0.35
EXERCISE_EMBED_DIM = 16         # Thẻ nhận diện bài tập (8 bài -> vector 16 chiều)
HIDDEN_DIM = 64                 # Kích thước ẩn vừa vặn (~80K-120K tham số)
