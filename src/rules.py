# -*- coding: utf-8 -*-
"""
src/rules.py
Hệ thống Luật Cơ Sinh Học Giải Thích 32 Mã Lỗi Kỹ Thuật cho 8 Bài Tập Nhóm A (Explainable Feedback).

Kiến trúc Lai 2 Tầng chống Rò rỉ Dữ liệu (CHỈ fit trên tập Train của Fold hiện tại):
  - Tầng 1 (Biomechanical Threshold Score - S_rule):
      Đo mức độ vi phạm chuẩn hóa theo trung vị (median) & độ phân tán kháng nhiễu (IQR/std)
      của các rep ĐÚNG trên Train, đánh giá theo đúng hướng vật lý (gt / lt / dev) của từng lỗi.
      Không phụ thuộc số lượng mẫu -> bắt được cả các mã lỗi hiếm (1-2 mẫu) như knee_valgus, kipping.
  - Tầng 2 (Error Deviation Prototype Bank - S_proto):
      Học Vector Chữ Ký Độ Lệch Đặc Trưng (Normalized Z-Deviation Signature) từ các mẫu Sai trên
      tập Train để phân tách rạch ròi các lỗi hay chồng lấn (ví dụ using_momentum vs partial_rom).
  - Đầu ra hoàn chỉnh:
      predicted_error_code, vietnamese_error_name, measured_evidence, coaching_advice, top3_candidates.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import GROUP_A_EXERCISES
from src.constants import ERROR_REASON_VN_DICT
from src.features import TABULAR_FEATURE_NAMES

_FEAT_IDX = {name: i for i, name in enumerate(TABULAR_FEATURE_NAMES)}
# 3 đặc trưng động học bổ trợ đóng vai trò tín hiệu hiệp đồng, không làm loãng mẫu số chuẩn hóa của luật chính
_SYNERGY_FEATURES = frozenset({"driver_peak_to_mean_speed", "driver_dimless_jerk", "torso_driver_corr"})


def _infer_min_std(feat_name: str) -> float:
    """Ngưỡng độ phân tán tối thiểu theo đơn vị vật lý để tránh khuếch đại nhiễu nhỏ."""
    if any(k in feat_name for k in ("peak_to_mean", "jerk", "corr")):
        return 0.25
    if any(k in feat_name for k in ("angle", "tilt", "asym", "lag")):
        if "shoulder_y_asym" in feat_name:
            return 0.04
        return 6.0
    if any(k in feat_name for k in ("speed", "bouncing")):
        return 2.0
    if any(k in feat_name for k in ("ratio", "duration", "frac")):
        return 0.20
    if any(k in feat_name for k in ("sag", "bridge", "drop", "chin", "dist", "compression")):
        return 0.06
    return 0.25


def _infer_unit(feat_name: str) -> str:
    """Đơn vị đo thân thiện dùng khi hiển thị bằng chứng số đo cho người dùng."""
    if "peak_to_mean" in feat_name or "ratio" in feat_name:
        return " lần"
    if "jerk" in feat_name or "corr" in feat_name:
        return ""
    if any(k in feat_name for k in ("angle", "tilt", "lag")) or feat_name in (
        "elbow_asym_max",
        "elbow_asym_mean",
        "knee_asym_max",
    ):
        return "°"
    if "duration" in feat_name:
        return "s"
    if "speed" in feat_name or "bouncing" in feat_name:
        return "°/step"
    return " đơn vị thân"


# ==============================================================================
# 1. BẢNG LUẬT CƠ SINH HỌC TƯỜNG MINH CHO 8 BÀI TẬP (45 CẶP BÀI-LỖI / 32 MÃ LỖI)
# Cấu trúc: (tên_đặc_trưng, hướng ['gt'|'lt'|'dev'], trọng_số, mô_tả_tiếng_Việt)
# ==============================================================================
EXERCISE_ERROR_RULES: dict[str, dict[str, list[tuple[str, str, float, str]]]] = {
    "pull up": {
        "partial_rom": [
            ("chin_above_wrist_max", "lt", 1.8, "Độ cao cằm vượt cổ tay"),
            ("elbow_angle_min", "gt", 1.3, "Góc gập khuỷu tay nhỏ nhất"),
            ("torso_tilt_rom", "lt", 0.5, "Biên độ kéo thân lên"),
        ],
        "using_momentum": [
            ("hip_angle_rom", "gt", 1.5, "Biên độ vung khớp hông"),
            ("torso_tilt_rom", "gt", 1.1, "Biên độ đung đưa thân người"),
            ("driver_peak_speed", "gt", 1.0, "Tốc độ giật kéo cực đại"),
            ("driver_peak_to_mean_speed", "gt", 0.90, "Tỷ lệ giật tốc độ cực đại so với trung bình"),
            ("torso_driver_corr", "gt", 0.85, "Độ lắc thân người ăn gian theo nhịp chuyển động"),
        ],
        "shoulders_shrugging": [
            ("neck_to_sh_dist_min", "lt", 1.6, "Khoảng cách đầu-vai bị rụt"),
            ("neck_angle_min", "gt", 1.1, "Góc cổ rướn vai"),
            ("chin_above_wrist_max", "gt", 0.5, "Mức kéo vượt tay"),
        ],
        "unequal_pull": [
            ("elbow_asym_mean", "gt", 1.5, "Độ lệch góc khuỷu 2 tay"),
            ("shoulder_y_asym_max", "gt", 1.4, "Chênh lệch độ cao 2 vai"),
            ("elbow_asym_max", "gt", 1.1, "Chênh lệch góc tay cực đại"),
        ],
        "uncontrolled_eccentric": [
            ("ecc_conc_speed_ratio", "gt", 1.9, "Tỷ lệ tốc độ thả rơi / kéo lên"),
            ("driver_phase2_speed", "gt", 1.0, "Vận tốc pha hạ người"),
            ("driver_dimless_jerk", "gt", 0.85, "Độ giật thiếu kiểm soát trong chu kỳ"),
        ],
        "kipping": [
            ("driver_speed_bouncing", "gt", 1.6, "Gia tốc đánh hông đổi chiều"),
            ("hip_angle_rom", "gt", 1.2, "Biên độ vung hông lấy đà"),
            ("torso_tilt_rom", "gt", 1.2, "Biên độ lắc thân"),
        ],
        "neck_extension": [
            ("neck_angle_min", "lt", 1.6, "Góc ngửa cổ rướn cằm"),
            ("neck_to_sh_dist_min", "lt", 1.0, "Khoảng cách đầu-vai"),
            ("driver_speed_bouncing", "gt", 0.8, "Gia tốc giật cằm qua xà"),
        ],
    },
    "push-up": {
        "waist_sagging": [
            ("waist_sag_signed_at_inflection", "gt", 1.7, "Độ võng hông tại đáy"),
            ("waist_sag_signed_max", "gt", 1.3, "Độ võng hông cực đại"),
        ],
        "elbow_excessive_flare": [
            ("elbow_flare_ratio_at_inflection", "gt", 1.7, "Tỷ lệ bạnh khuỷu/vai tại đáy"),
            ("elbow_flare_ratio_max", "gt", 1.2, "Độ rộng khuỷu cực đại"),
        ],
        "hips_too_high": [
            ("waist_sag_signed_at_inflection", "lt", 1.7, "Độ nhô hông chổng lên cao"),
            ("waist_sag_signed_min", "lt", 1.3, "Mức nhô hông cực đại"),
        ],
        "partial_rom": [
            ("elbow_angle_min", "gt", 1.6, "Góc khuỷu tay tại đáy"),
            ("elbow_angle_rom", "lt", 1.3, "Biên độ gập duỗi tay"),
        ],
        "snaking_push_up": [
            ("snaking_lag", "gt", 1.9, "Độ trễ hông nâng sau ngực"),
            ("waist_sag_signed_max", "gt", 0.9, "Độ võng hông pha đẩy"),
        ],
    },
    "leg extension": {
        "partial_rom": [
            ("knee_angle_max", "lt", 1.5, "Góc duỗi gối tối đa"),
            ("knee_angle_rom", "lt", 1.6, "Biên độ đá khớp gối"),
        ],
        "leaning_forward": [
            ("hip_angle_min", "lt", 1.7, "Góc gập hông đổ về trước"),
            ("torso_tilt_min", "lt", 1.1, "Góc ngả thân trên"),
            ("wrist_to_knee_dist_min", "lt", 0.6, "Độ vươn tay về phía gối"),
            ("torso_driver_corr", "gt", 0.80, "Độ ngả thân người đồng pha với khớp chủ đạo"),
        ],
        "torso_swinging": [
            ("torso_tilt_rom", "gt", 1.4, "Biên độ lắc thân trên"),
            ("torso_tilt_peak_speed", "gt", 1.2, "Tốc độ giật ngửa thân"),
            ("torso_tilt_std", "gt", 1.1, "Độ dao động thân trên"),
            ("torso_driver_corr", "gt", 0.80, "Độ ngả thân người đồng pha với khớp chủ đạo"),
        ],
        "using_momentum": [
            ("driver_speed_bouncing", "gt", 1.4, "Gia tốc giật đá tạ"),
            ("driver_peak_speed", "gt", 1.3, "Vận tốc đá cực đại"),
            ("ecc_conc_speed_ratio", "gt", 1.0, "Tỷ lệ pha hạ / đá"),
            ("driver_peak_to_mean_speed", "gt", 0.90, "Tỷ lệ giật tốc độ cực đại so với trung bình"),
            ("torso_driver_corr", "gt", 0.85, "Độ lắc thân người ăn gian theo nhịp chuyển động"),
        ],
        "hands_on_knees": [
            ("wrist_to_knee_dist_min", "lt", 2.2, "Khoảng cách cổ tay chạm đầu gối"),
            ("wrist_to_knee_dist_mean", "lt", 1.4, "Khoảng cách TB tay-gối"),
        ],
        "butt_lifting": [
            ("torso_tilt_rom", "gt", 1.3, "Biên độ nhấc hông/thân"),
            ("hip_below_bridge_min", "lt", 1.3, "Độ nhô hông rời mặt ghế"),
            ("hip_angle_min", "lt", 0.9, "Góc hông co rút"),
        ],
    },
    "squat": {
        "partial_rom": [
            ("knee_angle_at_inflection", "gt", 1.7, "Góc đầu gối tại đáy Squat"),
            ("hip_angle_at_inflection", "gt", 1.5, "Góc gập hông tại đáy Squat"),
            ("knee_angle_rom", "lt", 1.1, "Biên độ gập gối"),
        ],
        "hips_rising_first": [
            ("hips_rising_first_lag", "gt", 1.6, "Độ ngả lưng tăng thêm sau đáy"),
            ("torso_tilt_max", "gt", 1.2, "Góc đổ lưng cực đại"),
            ("knee_angle_at_inflection", "lt", 0.6, "Mức xuống đáy trước khi hông bật"),
        ],
        "heels_lifting": [
            ("shin_tilt_max", "gt", 1.7, "Độ chúi cẳng chân về trước"),
            ("shin_tilt_at_inflection", "gt", 1.3, "Độ nghiêng cẳng chân tại đáy"),
        ],
        "butt_wink": [
            ("spine_compression_min", "lt", 1.4, "Độ co cụp cột sống thắt lưng"),
            ("hip_angle_at_inflection", "lt", 1.1, "Độ gập hông sâu tại đáy"),
            ("body_line_angle_min", "lt", 1.0, "Góc gập gập thân-đùi-cổ chân"),
        ],
        "hips_too_low": [
            ("spine_compression_at_inflection", "lt", 1.6, "Độ nén thân tại đáy"),
            ("knee_angle_at_inflection", "lt", 1.4, "Góc gối gập quá sâu chạm bắp"),
            ("hip_angle_at_inflection", "lt", 1.0, "Góc gập hông quá sâu"),
        ],
        "leaning_forward": [
            ("torso_tilt_max", "gt", 1.5, "Góc đổ thân người về trước"),
            ("hips_rising_first_lag", "gt", 1.4, "Độ gập thân thêm lúc đứng lên"),
            ("hip_angle_min", "lt", 0.9, "Độ gập hông về trước"),
            ("torso_driver_corr", "gt", 0.80, "Độ ngả thân người đồng pha với khớp chủ đạo"),
        ],
        "knee_valgus": [
            ("knee_valgus_ratio_min", "lt", 1.8, "Tỷ lệ độ mở 2 gối / cổ chân"),
            ("knee_valgus_ratio_at_inflection", "lt", 1.4, "Độ chụm gối tại đáy"),
            ("knee_asym_max", "gt", 0.8, "Độ lệch hai đầu gối"),
        ],
    },
    "chest fly machine": {
        "shoulder_protraction": [
            ("shoulder_angle_max", "gt", 1.6, "Góc vai nhô về trước"),
            ("shoulder_angle_at_inflection", "gt", 1.2, "Góc vai tại đỉnh ép"),
            ("neck_to_sh_dist_min", "lt", 1.0, "Độ rụt vai về cổ"),
        ],
        "elbow_lockout": [
            ("elbow_angle_max", "gt", 1.5, "Góc khuỷu tay duỗi khóa thẳng"),
            ("elbow_angle_mean", "gt", 1.3, "Góc khuỷu trung bình cả rep"),
            ("wrist_squeeze_dist_min", "lt", 0.8, "Khoảng cách khép hai tay"),
        ],
        "excessive_elbow_bend": [
            ("elbow_angle_min", "lt", 1.6, "Góc gập khuỷu tay quá nhọn"),
            ("elbow_angle_at_inflection", "lt", 1.3, "Góc khuỷu tại đỉnh ép"),
            ("wrist_squeeze_dist_min", "gt", 0.9, "Độ mở hở cổ tay khi ép"),
        ],
        "elbows_dropping": [
            ("elbow_drop_vs_sh_max", "gt", 1.7, "Độ chúc cùi chỏ thấp hơn vai"),
            ("elbow_drop_vs_sh_at_inflection", "gt", 1.4, "Độ hạ khuỷu lúc ép ngực"),
        ],
        "shoulders_shrugging": [
            ("neck_to_sh_dist_min", "lt", 1.7, "Khoảng cách vai-cổ bị rụt"),
            ("shoulder_y_asym_max", "gt", 0.9, "Lệch nhô bả vai"),
        ],
    },
    "hip thrust": {
        "head_looking_up": [
            ("neck_angle_at_inflection", "gt", 1.7, "Góc cổ ngửa nhìn trần nhà"),
            ("neck_angle_min", "gt", 1.3, "Góc ngửa cổ tối thiểu"),
        ],
        "improper_foot_placement": [
            ("shin_tilt_at_inflection", "dev", 1.7, "Độ lệch cẳng chân khỏi phương đứng"),
            ("knee_angle_at_inflection", "dev", 1.3, "Góc gối lệch chuẩn 90-105°"),
        ],
        "hyperextended_back": [
            ("hip_below_bridge_min", "lt", 1.5, "Độ ưỡn vồng hông vượt cầu"),
            ("shin_tilt_at_inflection", "gt", 1.1, "Độ đẩy trượt người ra sau"),
            ("body_line_angle_max", "gt", 1.0, "Góc ưỡn quá ngưỡng thẳng lưng"),
        ],
        "partial_rom": [
            ("hip_angle_at_inflection", "lt", 1.7, "Góc mở hông tại đỉnh"),
            ("hip_below_bridge_at_inflection", "gt", 1.5, "Độ hụt hông dưới cầu vai-gối"),
        ],
        "using_momentum": [
            ("driver_speed_bouncing", "gt", 1.5, "Gia tốc giật nảy hông"),
            ("driver_peak_speed", "gt", 1.3, "Tốc độ quăng hông cực đại"),
            ("driver_peak_to_mean_speed", "gt", 0.90, "Tỷ lệ giật tốc độ cực đại so với trung bình"),
            ("torso_driver_corr", "gt", 0.85, "Độ lắc thân người ăn gian theo nhịp chuyển động"),
        ],
    },
    "t bar row": {
        "standing_too_upright": [
            ("torso_tilt_min", "lt", 1.7, "Độ ngả lưng tối thiểu"),
            ("torso_tilt_mean", "lt", 1.4, "Góc gập thân trung bình"),
            ("torso_tilt_rom", "lt", 1.0, "Biên độ giữ ổn định thân"),
        ],
        "uncontrolled_eccentric": [
            ("ecc_conc_speed_ratio", "gt", 1.6, "Tỷ lệ tốc độ thả tạ / kéo tạ"),
            ("driver_phase2_speed", "gt", 1.0, "Vận tốc pha hạ tạ"),
            ("torso_tilt_min", "lt", 0.7, "Độ bật thân pha hạ"),
            ("driver_dimless_jerk", "gt", 0.85, "Độ giật thiếu kiểm soát trong chu kỳ"),
        ],
        "using_momentum": [
            ("torso_tilt_rom", "gt", 1.7, "Biên độ giật ngửa thân trên"),
            ("torso_tilt_peak_speed", "gt", 1.4, "Tốc độ giật lưng lấy đà"),
            ("driver_peak_to_mean_speed", "gt", 0.90, "Tỷ lệ giật tốc độ cực đại so với trung bình"),
            ("torso_driver_corr", "gt", 0.85, "Độ lắc thân người ăn gian theo nhịp chuyển động"),
        ],
        "round_back": [
            ("spine_compression_min", "lt", 1.4, "Độ gù rút ngắn cột sống lưng"),
            ("torso_tilt_max", "gt", 1.3, "Độ gập cong thân về trước"),
            ("neck_angle_min", "lt", 1.0, "Độ cúi gập cổ theo lưng"),
        ],
        "straight_legs": [
            ("knee_angle_min", "gt", 1.8, "Góc đầu gối khóa thẳng đơ"),
            ("knee_angle_mean", "gt", 1.4, "Góc gối trung bình cả rep"),
        ],
        "improper_foot_placement": [
            ("shin_tilt_at_inflection", "dev", 1.5, "Độ nghiêng chân trụ lệch chuẩn"),
            ("knee_valgus_ratio_min", "dev", 1.3, "Tỷ lệ độ rộng thế đứng"),
        ],
    },
    "leg raises": {
        "using_momentum": [
            ("driver_peak_speed", "gt", 1.5, "Tốc độ quăng chân cực đại"),
            ("driver_speed_bouncing", "gt", 1.4, "Gia tốc giật nảy khớp hông"),
            ("torso_tilt_rom", "gt", 1.1, "Biên độ đung đưa thân người"),
            ("torso_tilt_peak_speed", "gt", 1.0, "Tốc độ vung đà thân"),
            ("driver_peak_to_mean_speed", "gt", 0.90, "Tỷ lệ giật tốc độ cực đại so với trung bình"),
            ("torso_driver_corr", "gt", 0.85, "Độ lắc thân người ăn gian theo nhịp chuyển động"),
        ],
        "hip_flexor_dominance": [
            ("hip_angle_min", "gt", 1.4, "Hông chưa cuộn hết lên trên"),
            ("torso_tilt_rom", "lt", 1.3, "Xương chậu không cuộn sau"),
            ("driver_peak_speed", "lt", 1.0, "Tốc độ gập chân chậm đều nhưng nông"),
        ],
        "arching_lower_back": [
            ("waist_sag_signed_min", "gt", 1.7, "Độ võng hông rời mặt tựa"),
            ("waist_sag_signed_max", "gt", 1.4, "Độ võng thắt lưng cực đại"),
        ],
        "partial_rom": [
            ("hip_angle_min", "gt", 1.5, "Chân nâng chưa đủ cao"),
            ("hip_angle_rom", "lt", 1.4, "Biên độ nâng chân"),
            ("driver_peak_speed", "lt", 0.8, "Tốc độ nâng chân ngắn"),
        ],
    },
}

EXERCISE_VALID_ERRORS: dict[str, list[str]] = {
    ex: list(rules.keys()) for ex, rules in EXERCISE_ERROR_RULES.items()
}


# ==============================================================================
# 2. TỪ ĐIỂN LỜI KHUYÊN SỬA FORM CỤ THỂ CHO TỪNG MÃ LỖI (32 MÃ + NONE)
# ==============================================================================
ERROR_COACHING_ADVICE_VI: dict[str, str] = {
    "none": "Duy trì nhịp thở đều, kiểm soát tốc độ cả pha lên lẫn pha xuống như hiện tại.",
    "partial_rom": "Giảm nhẹ mức tạ để đi hết biên độ tự nhiên của khớp ở cả điểm đáy và điểm khóa đỉnh.",
    "using_momentum": "Hạ tạ chậm lại, khóa chặt cơ lõi (Core) và không quăng giật thân người để mượn đà.",
    "uncontrolled_eccentric": "Kiểm soát pha hạ tạ kéo dài 2-3 giây thay vì thả rơi tự do.",
    "shoulders_shrugging": "Ghì chặt hai bả vai xuống dưới và ra sau (depress scapulae), giữ cổ dài tách khỏi vai.",
    "leaning_forward": "Giữ ngực mở và trọng tâm dồn thẳng trục, không gập đổ thân trên quá mức về trước.",
    "improper_foot_placement": "Đặt hai bàn chân rộng vừa bằng vai, cẳng chân vuông góc với mặt sàn ở điểm chịu lực.",
    "unequal_pull": "Tập trung kéo đều lực cả hai tay cùng lúc, giữ thanh vai luôn song song với xà.",
    "kipping": "Duỗi thẳng và kẹp nhẹ hai chân, gồng bụng giữ hông cố định thay vì vung chân lấy đà.",
    "neck_extension": "Giữ cổ trung lập thẳng hàng với cột sống, kéo ngực hướng lên xà thay vì ngửa cổ rướn cằm.",
    "waist_sagging": "Siết chặt cơ bụng và cơ mông để giữ vai - hông - cổ chân thành một đường thẳng Plank.",
    "elbow_excessive_flare": "Khép hai khuỷu tay tạo góc 45°-60° so với thân người thay vì bạnh ngang 90° hình chữ T.",
    "hips_too_high": "Hạ hông ngang bằng đường nối vai - mắt cá, tránh chổng mông làm giảm áp lực lên ngực.",
    "snaking_push_up": "Gồng cứng toàn thân như tấm ván để vai và hông rời sàn lên cùng một nhịp.",
    "torso_swinging": "Áp chặt lưng vào đệm tựa và bám chắc tay cầm hai bên ghế để cố định thân trên.",
    "butt_lifting": "Ép chặt mông xuống mặt ghế khi đá tạ lên, giảm bớt tạ nếu hông bị nhấc khỏi ghế.",
    "hands_on_knees": "Nắm chặt hai tay cầm bên hông ghế, tuyệt đối không dùng tay tì đẩy hỗ trợ lên đùi hoặc gối.",
    "hips_rising_first": "Khi thoát đáy Squat, đẩy lực gót chân đồng thời nâng ngực và hông lên cùng một tốc độ.",
    "heels_lifting": "Dồn đều trọng tâm vào giữa bàn chân và gót chân, cải thiện độ dẻo cổ chân trước khi xuống sâu.",
    "butt_wink": "Dừng hạ sâu ở điểm cột sống thắt lưng còn giữ được độ cong sinh lý tự nhiên, mở nhẹ mũi chân.",
    "hips_too_low": "Kiểm soát điểm dừng khi đùi song song hoặc thấp hơn mặt sàn chút ít, tránh thả rơi sập đáy.",
    "knee_valgus": "Chủ động đẩy hai đầu gối mở hướng theo mũi chân, kích hoạt cơ mông nhỡ suốt quá trình đứng lên.",
    "shoulder_protraction": "Ép chặt hai xương bả vai vào đệm lưng ghế suốt bài tập, chỉ chuyển động khớp vai - cánh tay.",
    "elbow_lockout": "Giữ một độ cong rất nhẹ (~10-15°) cố định ở khuỷu tay suốt rep để bảo vệ khớp khuỷu.",
    "excessive_elbow_bend": "Mở rộng góc khuỷu tay ở mức tự nhiên (~135-155°) để dồn lực kéo căng cơ ngực thay vì tay sau.",
    "elbows_dropping": "Giữ cùi chỏ nâng cao ngang bằng đường giữa ngực và mặt phẳng tay cầm suốt biên độ ép.",
    "head_looking_up": "Thu nhẹ cằm về phía ngực (chin tuck) và nhìn cố định vào một điểm ngang tầm mắt.",
    "hyperextended_back": "Kết thúc đỉnh rep bằng cách siết mạnh cơ mông ở tư thế lưng thẳng, không ưỡn võng thắt lưng.",
    "standing_too_upright": "Cúi gập hông (hip hinge) sâu khoảng 45°-60° về trước và giữ góc lưng cố định khi kéo.",
    "round_back": "Ưỡn mở ngực, khóa thẳng cột sống lưng từ xương chậu lên vai trước khi nhấc tạ chèo.",
    "straight_legs": "Chùng nhẹ hai đầu gối (~15-20°) để truyền tải lực an toàn xuống đùi sau và bảo vệ lưng dưới.",
    "hip_flexor_dominance": "Khi chân lên ngang hông, chủ động cuộn cong xương chậu hướng về phía ngực để vào cơ bụng dưới.",
    "arching_lower_back": "Ép chặt lưng dưới vào đệm tựa và dừng hạ chân trước khi thắt lưng bị võng nhấc lên.",
}


# ==============================================================================
# 3. HỆ THỐNG CHẨN ĐOÁN LAI 2 TẦNG (BIOMECHANICAL RULE ENGINE)
# ==============================================================================
class BiomechanicalRuleEngine:
    """
    Bộ chẩn đoán & giải thích 32 mã lỗi kỹ thuật cho 8 bài tập Nhóm A:
      - Tầng 1: Điểm vi phạm chuẩn hóa trên luật vật lý (median + robust IQR của Train Đúng).
      - Tầng 2: Đối chiếu Vector Chữ Ký Độ Lệch (Normalized Z-Deviation Signature)
                của từng mã lỗi thu thập CHỈ trên tập Train của Fold hiện tại.
    """

    def __init__(
        self,
        rule_registry: dict | None = None,
        rule_weight: float = 0.55,
        proto_weight: float = 0.45,
    ):
        self.rules = rule_registry or EXERCISE_ERROR_RULES
        self.rule_weight = float(rule_weight)
        self.proto_weight = float(proto_weight)
        self.stats_by_ex: dict[str, dict[str, np.ndarray]] = {}
        self.err_prototypes: dict[str, dict] = {}
        self.rule_calib_by_ex: dict[str, dict[str, np.ndarray]] = {}
        self.fitted = False

    def fit(self, df_train: pd.DataFrame, X_tab_train: np.ndarray) -> "BiomechanicalRuleEngine":
        """Học tham chiếu khỏe mạnh & chữ ký lỗi thực nghiệm CHỈ trên tập Train (Zero Leakage)."""
        df_reset = df_train.reset_index(drop=True)
        min_stds = np.asarray([_infer_min_std(n) for n in TABULAR_FEATURE_NAMES], dtype=np.float32)

        for ex in GROUP_A_EXERCISES:
            mask_cor = ((df_reset["exercise"] == ex) & (df_reset["form_label"] == "correct")).to_numpy()
            if not np.any(mask_cor):
                mask_cor = (df_reset["exercise"] == ex).to_numpy()
            pool_cor = X_tab_train[mask_cor]

            med = np.median(pool_cor, axis=0).astype(np.float32)
            p25 = np.percentile(pool_cor, 25, axis=0).astype(np.float32)
            p75 = np.percentile(pool_cor, 75, axis=0).astype(np.float32)
            iqr_scale = (p75 - p25) / 1.349
            std_raw = np.std(pool_cor, axis=0).astype(np.float32)
            robust_scale = np.maximum(np.maximum(iqr_scale, std_raw), min_stds).astype(np.float32)

            self.stats_by_ex[ex] = {
                "median": med,
                "scale": robust_scale,
                "p15": np.percentile(pool_cor, 15, axis=0).astype(np.float32),
                "p85": np.percentile(pool_cor, 85, axis=0).astype(np.float32),
            }

            # Tập hợp các cột đặc trưng luật + động học dùng tạo Vector Chữ Ký Lỗi
            feat_cols = sorted({
                _FEAT_IDX[feat]
                for clauses in self.rules[ex].values()
                for feat, _, _, _ in clauses
            })
            for extra in ("golden_angle_dev_mean", "golden_dev_at_inflection"):
                if extra in _FEAT_IDX and _FEAT_IDX[extra] not in feat_cols:
                    feat_cols.append(_FEAT_IDX[extra])

            mask_inc = ((df_reset["exercise"] == ex) & (df_reset["form_label"] == "incorrect")).to_numpy()
            y_err = df_reset.loc[mask_inc, "error_reason"].to_numpy()
            z_inc = np.clip(
                (X_tab_train[mask_inc][:, feat_cols] - med[feat_cols]) / robust_scale[feat_cols],
                -4.0,
                4.0,
            )

            protos: dict[str, tuple[np.ndarray, float]] = {}
            total_inc = max(1, int(mask_inc.sum()))
            for err_code in self.rules[ex]:
                e_mask = y_err == err_code
                cnt = int(e_mask.sum())
                if cnt > 0:
                    centroid = np.median(z_inc[e_mask], axis=0).astype(np.float32)
                    prior = cnt / float(total_inc)
                    protos[err_code] = (centroid, prior)

            self.err_prototypes[ex] = {
                "feat_cols": feat_cols,
                "protos": protos,
            }

            cand_rules = self.rules[ex]
            err_codes_ex = list(cand_rules.keys())
            ex_stats_cur = self.stats_by_ex[ex]
            x_inc_ex = X_tab_train[mask_inc]
            if len(x_inc_ex) > 0:
                mat = np.zeros((len(x_inc_ex), len(err_codes_ex)), dtype=np.float32)
                for si in range(len(x_inc_ex)):
                    for ei, ec in enumerate(err_codes_ex):
                        mat[si, ei], _ = self._score_err_clauses(x_inc_ex[si], cand_rules[ec], ex_stats_cur)
                r_med = np.percentile(mat, 60, axis=0).astype(np.float32)
                r_scale = np.maximum(np.std(mat, axis=0), 0.35).astype(np.float32)
            else:
                r_med = np.ones(len(err_codes_ex), dtype=np.float32)
                r_scale = np.ones(len(err_codes_ex), dtype=np.float32)
            self.rule_calib_by_ex[ex] = {"r_med": r_med, "r_scale": r_scale}

        self.fitted = True
        return self

    def _score_err_clauses(
        self,
        tab_vec: np.ndarray,
        clauses: list[tuple[str, str, float, str]],
        ex_stats: dict[str, np.ndarray],
        biomech_metrics: dict[str, float] | None = None,
    ) -> tuple[float, dict | None]:
        """Tính điểm vi phạm chuẩn hóa trên 1 lỗi và trích điều khoản nổi bật nhất."""
        total_score = 0.0
        total_w = 0.0
        best_clause = None
        best_raw_z = -1e9

        for feat_name, direction, weight, desc_vi in clauses:
            val = float(tab_vec[_FEAT_IDX[feat_name]])
            if biomech_metrics is not None:
                biomech_metrics[feat_name] = round(val, 3)
            v_score, raw_z, ref_val, med_val = self._clause_violation(val, feat_name, direction, ex_stats)
            if feat_name in _SYNERGY_FEATURES:
                total_score += 0.25 * weight * v_score
            else:
                total_score += weight * v_score
                total_w += weight

            if biomech_metrics is not None:
                weighted_z = weight * raw_z
                if weighted_z > best_raw_z:
                    best_raw_z = weighted_z
                    best_clause = {
                        "feature": feat_name,
                        "desc_vi": desc_vi,
                        "direction": direction,
                        "measured": round(val, 2),
                        "reference": round(ref_val, 2),
                        "healthy_median": round(med_val, 2),
                        "unit": _infer_unit(feat_name),
                    }

        return float(total_score / max(total_w, 1e-6)), best_clause

    def _clause_violation(
        self,
        val: float,
        feat_name: str,
        direction: str,
        ex_stats: dict[str, np.ndarray],
    ) -> tuple[float, float, float, float]:
        j = _FEAT_IDX[feat_name]
        med = float(ex_stats["median"][j])
        sc = float(ex_stats["scale"][j])
        if direction == "gt":
            raw_z = (val - med) / sc
            ref = float(ex_stats["p85"][j])
        elif direction == "lt":
            raw_z = (med - val) / sc
            ref = float(ex_stats["p15"][j])
        elif direction == "dev":
            raw_z = abs(val - med) / sc - 0.5
            ref = med
        else:
            raise ValueError(f"Unknown rule direction: {direction}")
        return max(0.0, float(raw_z)), float(raw_z), ref, med

    def explain_rep(
        self,
        tab_vec: np.ndarray,
        exercise_name: str,
        p_incorrect: float | None = None,
        is_incorrect: bool = True,
        top_k: int = 3,
    ) -> dict:
        """
        Chẩn đoán & sinh phản hồi giải thích bằng tiếng Việt cho 1 rep (dùng cho Bước E & Realtime).
        """
        if not self.fitted:
            raise RuntimeError("BiomechanicalRuleEngine must be fitted on Train before calling explain_rep()!")
        if exercise_name not in self.rules:
            raise ValueError(f"Không có bảng luật cho bài tập: {exercise_name}")

        ex_stats = self.stats_by_ex[exercise_name]
        cand_rules = self.rules[exercise_name]
        err_codes = list(cand_rules.keys())

        if not is_incorrect:
            prob_txt = f"[Xác suất Sai: {p_incorrect * 100:.1f}%] " if p_incorrect is not None else ""
            return {
                "exercise": exercise_name,
                "predicted_error_code": "none",
                "predicted_error_reason": "none",
                "vietnamese_error_name": ERROR_REASON_VN_DICT["none"],
                "measured_evidence": "Các góc khớp và nhịp độ chuyển động đều nằm trong vùng chuẩn của bài tập.",
                "coaching_advice": ERROR_COACHING_ADVICE_VI["none"],
                "top3_candidates": [],
                "top_k_errors": ["none"],
                "ranked_details": [],
                "biomech_metrics": {},
                "feedback_vi": f"{prob_txt}{ERROR_REASON_VN_DICT['none']}. {ERROR_COACHING_ADVICE_VI['none']}",
            }

        # Tầng 1: Điểm vi phạm cơ sinh học tường minh (S_rule)
        raw_rule_scores = []
        evidences = {}
        biomech_metrics = {}
        for err_code in err_codes:
            err_score, best_clause = self._score_err_clauses(
                tab_vec,
                cand_rules[err_code],
                ex_stats,
                biomech_metrics=biomech_metrics,
            )
            raw_rule_scores.append(err_score)
            evidences[err_code] = best_clause

        raw_arr = np.asarray(raw_rule_scores, dtype=np.float64)
        calib = self.rule_calib_by_ex.get(exercise_name)
        if calib is not None:
            raw_arr = 0.70 * raw_arr + 0.30 * np.maximum(
                0.0, (raw_arr - calib["r_med"] * 0.5) / calib["r_scale"]
            )

        arr_r = raw_arr * 2.5
        exp_r = np.exp(arr_r - np.max(arr_r))
        rule_probs = exp_r / np.maximum(exp_r.sum(), 1e-8)

        # Tầng 2: Đối chiếu Chữ Ký Lỗi thực nghiệm từ Train (S_proto)
        proto_pack = self.err_prototypes[exercise_name]
        feat_cols = proto_pack["feat_cols"]
        protos = proto_pack["protos"]
        proto_probs = np.zeros_like(rule_probs)

        if protos:
            z_sample = np.clip(
                (tab_vec[feat_cols] - ex_stats["median"][feat_cols]) / ex_stats["scale"][feat_cols],
                -4.0,
                4.0,
            )
            raw_proto_logits = np.full(len(err_codes), -1e9, dtype=np.float64)
            observed_mask = np.zeros(len(err_codes), dtype=bool)

            for i, code in enumerate(err_codes):
                if code in protos:
                    centroid, prior = protos[code]
                    rel_feats = {f for f, _, _, _ in cand_rules[code]}
                    rel_w = np.asarray(
                        [1.0 if TABULAR_FEATURE_NAMES[c] in rel_feats else 0.25 for c in feat_cols],
                        dtype=np.float32,
                    )
                    z_w = z_sample * rel_w
                    c_w = centroid * rel_w
                    dist_mse = float(np.sum(rel_w * ((z_sample - centroid) ** 2)) / np.sum(rel_w))
                    cos_sim = float(
                        np.dot(z_w, c_w) / ((float(np.linalg.norm(z_w)) + 1e-6) * (float(np.linalg.norm(c_w)) + 1e-6))
                    )
                    raw_proto_logits[i] = -0.65 * dist_mse + 1.40 * cos_sim + 0.35 * np.log(max(prior, 1e-3))
                    observed_mask[i] = True

            if np.any(observed_mask):
                obs_logits = raw_proto_logits[observed_mask]
                obs_exp = np.exp(obs_logits - np.max(obs_logits))
                proto_probs[observed_mask] = obs_exp / np.maximum(obs_exp.sum(), 1e-8)

            # Chỉ cho Tầng 2 (Prototype) phát huy tối đa ở các lỗi có dấu hiệu vật lý kích hoạt (rule_z > 0)
            active_gate = (
                np.where(raw_arr > 0.0, 1.0, 0.25)
                if np.max(raw_arr) > 0.02
                else np.ones(len(err_codes), dtype=np.float64)
            )
            w_p = np.where(observed_mask, self.proto_weight * active_gate, 0.0)
            blended_probs = (1.0 - w_p) * rule_probs + w_p * proto_probs
        else:
            blended_probs = rule_probs.copy()

        scored = [
            {
                "error_reason": code,
                "vietnamese_name": ERROR_REASON_VN_DICT[code],
                "score": round(float(blended_probs[i]), 4),
                "rule_score": round(float(rule_probs[i]), 4),
                "proto_score": round(float(proto_probs[i]), 4),
                "rule_z": round(float(raw_arr[i]), 4),
                "evidence": evidences[code],
            }
            for i, code in enumerate(err_codes)
        ]
        scored.sort(key=lambda d: d["score"], reverse=True)

        top_choice = scored[0]
        err_code = top_choice["error_reason"]
        ev = top_choice["evidence"]
        m_val, r_val, h_med, unit = ev["measured"], ev["reference"], ev["healthy_median"], ev["unit"]
        if ev["direction"] == "gt" and m_val > r_val:
            cmp_txt = f"đo được {m_val}{unit} (vượt ngưỡng chuẩn <= {r_val}{unit})"
        elif ev["direction"] == "lt" and m_val < r_val:
            cmp_txt = f"đo được {m_val}{unit} (thấp hơn ngưỡng chuẩn >= {r_val}{unit})"
        elif ev["direction"] == "dev" and abs(m_val - h_med) > 1e-6:
            cmp_txt = f"đo được {m_val}{unit} (lệch khỏi mức chuẩn {h_med}{unit})"
        else:
            cmp_txt = f"đo được {m_val}{unit} (trung vị chuẩn {h_med}{unit})"

        measured_evidence = f"{ev['desc_vi']}: {cmp_txt}"
        advice = ERROR_COACHING_ADVICE_VI[err_code]
        prob_prefix = f"[Xác suất Sai: {p_incorrect * 100:.1f}%] " if p_incorrect is not None else ""
        feedback_vi = (
            f"{prob_prefix}Lỗi chính: {ERROR_REASON_VN_DICT[err_code]} "
            f"[{measured_evidence}]. Gợi ý sửa: {advice}"
        )

        return {
            "exercise": exercise_name,
            "predicted_error_code": err_code,
            "predicted_error_reason": err_code,
            "vietnamese_error_name": ERROR_REASON_VN_DICT[err_code],
            "measured_evidence": measured_evidence,
            "coaching_advice": advice,
            "top3_candidates": scored[:top_k],
            "top_k_errors": [d["error_reason"] for d in scored[:top_k]],
            "ranked_details": scored[:top_k],
            "biomech_metrics": biomech_metrics,
            "feedback_vi": feedback_vi,
        }

    def explain_sample(
        self,
        tab_vec: np.ndarray,
        exercise_name: str,
        p_incorrect: float | None = None,
        top_k: int = 3,
    ) -> dict:
        """Alias tương thích gọi explain_rep()."""
        return self.explain_rep(tab_vec, exercise_name, p_incorrect=p_incorrect, top_k=top_k)

    def explain_batch(
        self,
        X_tab: np.ndarray,
        exercises: list[str] | np.ndarray,
        probs_incorrect: np.ndarray | None = None,
        top_k: int = 3,
    ) -> list[dict]:
        """Giải thích hàng loạt cho cả tập Validation hoặc Test."""
        out = []
        for i in range(len(X_tab)):
            p_i = float(probs_incorrect[i]) if probs_incorrect is not None else None
            out.append(self.explain_rep(X_tab[i], str(exercises[i]), p_incorrect=p_i, top_k=top_k))
        return out


BiomechanicalRuleExplainer = BiomechanicalRuleEngine


# ==============================================================================
# 4. KIỂM ĐỊNH ĐỘ PHỦ 32 MÃ LỖI & CHẨN ĐOÁN 5-FOLD (--verify)
# ==============================================================================
def verify_rules() -> None:
    """Kiểm định độ phủ 32/32 mã lỗi, Top-1/Top-3 Accuracy 5-Fold và mẫu giải thích tiếng Việt."""
    from src.dataset import (
        build_stratified_group_folds,
        load_pose_cache,
        load_task2_metadata,
        prepare_fold_data,
    )

    print("=" * 78)
    print("FORGAI STEP D - KIỂM ĐỊNH HỆ THỐNG LUẬT CƠ SINH HỌC 32 MÃ LỖI (src/rules.py)")
    print("=" * 78)

    # 1. Kiểm tra độ phủ 100% mã lỗi & lời khuyên
    rule_codes = {code for ex in EXERCISE_ERROR_RULES.values() for code in ex}
    dict_codes = set(ERROR_REASON_VN_DICT.keys()) - {"none"}
    advice_codes = set(ERROR_COACHING_ADVICE_VI.keys()) - {"none"}
    assert rule_codes == dict_codes, f"Lệch mã lỗi: thiếu={dict_codes - rule_codes}, thừa={rule_codes - dict_codes}"
    assert dict_codes == advice_codes, f"Thiếu lời khuyên: {dict_codes - advice_codes}"
    total_pairs = sum(len(v) for v in EXERCISE_ERROR_RULES.values())
    print(f"[1/3] Độ phủ từ điển: {len(rule_codes)}/32 mã lỗi, {total_pairs} cặp (bài tập, lỗi), 100% có lời khuyên -> PASS")

    # 2. Đánh giá 5-Fold Zero Data Leakage trên toàn bộ các rep Sai
    df = load_task2_metadata()
    pose_cache = load_pose_cache(df)
    folds = build_stratified_group_folds(df)

    tier1_top1, hybrid_top1, hybrid_top3 = [], [], []
    sample_examples = []

    for f in folds:
        b = prepare_fold_data(df, f, pose_cache=pose_cache, apply_train_aug=False)
        df_tr = df.iloc[f["train_idx"]].reset_index(drop=True)
        df_te = df.iloc[f["test_idx"]].reset_index(drop=True)

        # Kiểm định thuần Tầng 1 (proto_weight=0.0) và Lai 2 Tầng (proto_weight=0.45)
        eng_rule_only = BiomechanicalRuleEngine(rule_weight=1.0, proto_weight=0.0).fit(df_tr, b["train"]["tab_raw"])
        eng_hybrid = BiomechanicalRuleEngine(rule_weight=0.55, proto_weight=0.45).fit(df_tr, b["train"]["tab_raw"])

        m_inc = (df_te["form_label"] == "incorrect").to_numpy()
        x_te_inc = b["test"]["tab_raw"][m_inc]
        ex_te_inc = df_te.loc[m_inc, "exercise"].tolist()
        y_te_inc = df_te.loc[m_inc, "error_reason"].tolist()
        sid_te_inc = df_te.loc[m_inc, "segment_id"].tolist()

        res_rule = eng_rule_only.explain_batch(x_te_inc, ex_te_inc)
        res_hyb = eng_hybrid.explain_batch(x_te_inc, ex_te_inc, probs_incorrect=np.full(len(x_te_inc), 0.92))

        tier1_top1.append(sum(r["predicted_error_code"] == yt for r, yt in zip(res_rule, y_te_inc)) / len(y_te_inc))
        hybrid_top1.append(sum(r["predicted_error_code"] == yt for r, yt in zip(res_hyb, y_te_inc)) / len(y_te_inc))
        hybrid_top3.append(sum(yt in r["top_k_errors"] for r, yt in zip(res_hyb, y_te_inc)) / len(y_te_inc))

        if f["fold"] == 0:
            picked_ex = set()
            for sid, yt, out in zip(sid_te_inc, y_te_inc, res_hyb):
                if out["exercise"] in ("squat", "push-up", "pull up") and out["exercise"] not in picked_ex:
                    sample_examples.append((sid, yt, out["feedback_vi"]))
                    picked_ex.add(out["exercise"])

    print(
        f"[2/3] Độ chính xác chẩn đoán lỗi 5-Fold (ngẫu nhiên ~1/6 = 16.7%):\n"
        f"      - Chỉ Tầng 1 (Luật vật lý 0-shot): Top-1 = {np.mean(tier1_top1) * 100:.1f}%\n"
        f"      - Lai 2 Tầng (Luật + Chữ ký Train): Top-1 = {np.mean(hybrid_top1) * 100:.1f}%, "
        f"Top-3 = {np.mean(hybrid_top3) * 100:.1f}%"
    )
    print("[3/3] Ví dụ phản hồi tiếng Việt thực tế trên Fold 0 Test:")
    for sid, yt, fb in sample_examples:
        print(f"      - [{sid} | nhãn thật: {yt}] -> {fb}")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    parser = argparse.ArgumentParser(description="ForgAI Step D Biomechanical Rules Verification")
    parser.add_argument("--verify", action="store_true", help="Kiểm định độ phủ 32 mã lỗi & 5-Fold CV")
    args = parser.parse_args()
    if args.verify:
        verify_rules()
    else:
        print("Chạy `python src/rules.py --verify` để kiểm định hệ thống luật 32 mã lỗi.")
