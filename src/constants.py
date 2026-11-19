# -*- coding: utf-8 -*-
"""
src/constants.py
Định nghĩa 13 khớp giải phẫu (RAW_POINTS_13), bản đồ lật gương Trái-Phải (SWAP_MAP),
78 cặp khớp tính góc nghiêng 3D, và từ điển giải thích 32 mã lỗi tiếng Việt cho Nhóm A.
"""

from itertools import combinations

# ==============================================================================
# 1. 13 KHỚP ĐỘNG HỌC CỐT LÕI (THEO BÀI BÁO SKELGYM - ESWA 2026)
# ==============================================================================
# Thứ tự chỉ số trong 33 điểm gốc của Google MediaPipe Pose:
# 0: Mũi (Đại diện cho Đầu/Cổ)
# 11, 12: Vai Trái, Vai Phải
# 13, 14: Khuỷu tay Trái, Khuỷu tay Phải
# 15, 16: Cổ tay Trái, Cổ tay Phải
# 23, 24: Hông Trái, Hông Phải
# 25, 26: Đầu gối Trái, Đầu gối Phải
# 27, 28: Cổ chân Trái, Cổ chân Phải
RAW_POINTS_13 = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 0]

# Vị trí index (từ 0 đến 12) sau khi đã lọc ra mảng 13 khớp:
JOINT_NAMES_13 = [
    "left_shoulder",   # 0 (MediaPipe 11)
    "right_shoulder",  # 1 (MediaPipe 12)
    "left_elbow",      # 2 (MediaPipe 13)
    "right_elbow",     # 3 (MediaPipe 14)
    "left_wrist",      # 4 (MediaPipe 15)
    "right_wrist",     # 5 (MediaPipe 16)
    "left_hip",        # 6 (MediaPipe 23)
    "right_hip",       # 7 (MediaPipe 24)
    "left_knee",       # 8 (MediaPipe 25)
    "right_knee",      # 9 (MediaPipe 26)
    "left_ankle",      # 10 (MediaPipe 27)
    "right_ankle",     # 11 (MediaPipe 28)
    "nose",            # 12 (MediaPipe 0)
]

J_IDX = {name: i for i, name in enumerate(JOINT_NAMES_13)}

# Các đoạn nối xương (Bones) dùng để vẽ hình trực quan hóa khung xương 13 khớp:
SKELETON_EDGES_13 = [
    (0, 1),   # Vai Trái - Vai Phải
    (6, 7),   # Hông Trái - Hông Phải
    (0, 6),   # Vai Trái - Hông Trái (Thân trái)
    (1, 7),   # Vai Phải - Hông Phải (Thân phải)
    (0, 2), (2, 4),   # Tay Trái: Vai -> Khuỷu -> Cổ tay
    (1, 3), (3, 5),   # Tay Phải: Vai -> Khuỷu -> Cổ tay
    (6, 8), (8, 10),  # Chân Trái: Hông -> Gối -> Cổ chân
    (7, 9), (9, 11),  # Chân Phải: Hông -> Gối -> Cổ chân
    (0, 12), (1, 12), # Vai -> Mũi/Đầu
]

# ==============================================================================
# 2. BẢN ĐỒ LẬT GƯƠNG TRÁI <-> PHẢI (SWAP_MAP TRONG MẢNG 13 KHỚP)
# ==============================================================================
# Khi lật gương ngang (X -> -X), các cặp khớp đối xứng đổi chỗ cho nhau:
# (0 <-> 1: Vai), (2 <-> 3: Khuỷu), (4 <-> 5: Cổ tay),
# (6 <-> 7: Hông), (8 <-> 9: Gối), (10 <-> 11: Cổ chân), (12 giữ nguyên: Mũi)
SWAP_MAP_13 = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12]

# ==============================================================================
# 3. 78 CẶP KHỚP TÍNH GÓC NGHIÊNG 3D (TỔ HỢP C(13, 2) = 78 CẶP)
# ==============================================================================
PAIRWISE_JOINT_INDICES_78 = list(combinations(range(13), 2))  # Đúng 78 cặp (i, j)

# ==============================================================================
# 4. NHÃN PHÂN LOẠI CHẤT LƯỢNG KỸ THUẬT (TASK 2)
# ==============================================================================
FORM_LABEL_TO_ID = {"correct": 0, "incorrect": 1}
ID_TO_FORM_LABEL = {0: "correct", 1: "incorrect"}

# ==============================================================================
# 5. TỪ ĐIỂN GIẢI THÍCH TIẾNG VIỆT CHO 32 MÃ LỖI CƠ SINH HỌC (NHÓM A)
# ==============================================================================
ERROR_REASON_VN_DICT = {
    "none": "Kỹ thuật chuẩn (Không có lỗi)",
    # Lỗi dùng chung nhiều bài
    "partial_rom": "Chưa đi hết biên độ động tác (Hạ chưa đủ sâu hoặc kéo/đẩy chưa tới đỉnh)",
    "using_momentum": "Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp",
    "uncontrolled_eccentric": "Hạ tạ rơi tự do quá nhanh, thiếu kiểm soát ở pha âm (Eccentric)",
    "shoulders_shrugging": "Nhún hai bả vai lên sát tai (Dùng cơ cầu vai thay vì ép chặt bả vai)",
    "leaning_forward": "Đổ gập thân người về phía trước quá mức làm lệch trọng tâm",
    "improper_foot_placement": "Đặt vị trí bàn chân sai khoảng cách tiêu chuẩn cơ sinh học",
    # 1. Pull up
    "unequal_pull": "Kéo lệch hai tay (Hai vai và khuỷu tay lên không đều nhau)",
    "kipping": "Đánh lắc hông và vung chân lấy đà (Kipping) thay vì kéo xà nghiêm ngặt",
    "neck_extension": "Ngửa cổ rướn cằm quá mức để cố qua xà thay vì kéo ngực lên",
    # 2. Push-up
    "waist_sagging": "Võng lưng dưới / sập hông xuống sàn do không gồng chặt cơ bụng (Core)",
    "elbow_excessive_flare": "Bạnh khuỷu tay ngang 90° hình chữ T (Gây chèn ép khớp vai)",
    "hips_too_high": "Chổng mông / nhô hông quá cao làm mất đường thẳng Plank từ vai xuống chân",
    "snaking_push_up": "Đẩy người uốn sóng kiểu con rắn (Ngực và vai nhấc lên trước, hông còn nằm dưới sàn)",
    # 3. Leg extension
    "torso_swinging": "Đung đưa / giật ngả thân trên theo từng nhịp đá để mượn đà nâng tạ",
    "butt_lifting": "Nhấc mông rời khỏi mặt ghế khi đá tạ lên",
    "hands_on_knees": "Đặt tay tì đè hoặc đẩy hỗ trợ lên đầu gối thay vì nắm tay cầm hai bên ghế",
    # 4. Squat
    "hips_rising_first": "Hông bật lên nhanh hơn vai khi thoát đáy (Biến Squat thành cúi gập lưng Good-Morning)",
    "heels_lifting": "Nhón gót chân rời khỏi mặt sàn khi hạ người xuống sâu",
    "butt_wink": "Cụp xương chậu / cuộn tròn lưng dưới ở điểm đáy Squat (Butt Wink)",
    "hips_too_low": "Thả rơi mông quá sâu chạm bắp chân làm mất độ căng kiểm soát ở đáy",
    "knee_valgus": "Sập chụm hai đầu gối vào trong hình chữ X (Nguy cơ chấn thương dây chằng gối)",
    # 5. Chest fly machine
    "shoulder_protraction": "Nhô bả vai về phía trước khi ép tay (Làm mất thế ưỡn ngực cố định trên ghế)",
    "elbow_lockout": "Duỗi khóa thẳng đơ khớp khuỷu tay 180° thay vì giữ độ cong nhẹ tự nhiên",
    "excessive_elbow_bend": "Gập khuỷu tay quá nhọn (< 100°) biến bài ép ngực thành bài đẩy tay sau",
    "elbows_dropping": "Hạ chúc hai cùi chỏ xuống thấp hơn đường ngang ngực khi mở/ép tạ",
    # 6. Hip thrust
    "head_looking_up": "Ngửa cổ nhìn thẳng lên trần nhà thay vì thu nhẹ cằm nhìn về phía trước",
    "hyperextended_back": "Ưỡn võng cột sống thắt lưng quá mức ở đỉnh rep thay vì khóa bằng cơ mông",
    # 7. T bar row
    "standing_too_upright": "Đứng thân trên quá thẳng (> 60° so với sàn) làm lực chuyển sang cầu vai thay vì lưng giữa",
    "round_back": "Cong gù cột sống lưng trong khi kéo tạ (Nguy cơ thoát vị đĩa đệm)",
    "straight_legs": "Khóa thẳng đơ hai đầu gối thay vì giữ độ chùng gối nhẹ để bảo vệ lưng dưới",
    # 8. Leg raises
    "hip_flexor_dominance": "Chỉ gập khớp hông nhấc chân thấp mà không cuộn xương chậu (Không vào cơ bụng dưới)",
    "arching_lower_back": "Võng ưỡn thắt lưng quá mức hoặc rời khỏi mặt phẳng tựa",
}
