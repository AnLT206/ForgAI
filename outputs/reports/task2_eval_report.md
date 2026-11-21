# Báo cáo Đánh giá Task 2 (5-Fold Stratified Group Cross-Validation)

## 1. Hậu kiểm Zero Data Leakage
- Tổng số mẫu Out-of-Fold: **561** (`330 correct` / `231 incorrect`)
- Số `segment_id` duy nhất: **561**
- Số nhóm `video_id` độc lập: **405**
- Số `video_id` rò rỉ giữa các Fold: **0**

## 2. So sánh 4 Mô hình qua 5 Fold (Mean ± Std) & Pooled OOF (N = 561)
| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC | Pooled Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BiomechanicalRF (114-d E.3+) | 76.47 ± 2.58% | 76.16 ± 2.58% | 75.91 ± 2.61% | 74.44 ± 3.37% | 0.843 ± 0.027 | **75.92%** |
| CNN-BiLSTM + FiLM | 72.38 ± 4.39% | 72.93 ± 3.96% | 72.08 ± 4.28% | 76.17 ± 4.18% | 0.810 ± 0.026 | **72.11%** |
| Transformer + FiLM | 70.59 ± 3.61% | 71.24 ± 3.51% | 70.33 ± 3.55% | 74.90 ± 4.67% | 0.792 ± 0.037 | **70.35%** |
| Hybrid Ensemble (0.50/0.25/0.25) | 77.01 ± 3.02% | 76.94 ± 2.75% | 76.55 ± 2.94% | 76.61 ± 2.47% | 0.835 ± 0.030 | **76.55%** |

### Đối chiếu cặp RF vs Hybrid Ensemble (McNemar Exact)
- Cả 2 cùng đúng: **403** | Hybrid sửa đúng ca RF sai: **29** | RF đúng ca Hybrid sai: **26** | Cả 2 cùng sai: **103** (p-value = `0.7877`).

## 3. Phân rã theo 8 Bài tập Nhóm A (Pooled OOF)
| Bài tập | Tổng (Đúng/Sai) | RF Acc | RF Macro-F1 | CNN-BiLSTM MF1 | Transformer MF1 | Hybrid Acc | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pull up | 81 (41/40) | 82.7% | 82.7% | 75.1% | 79.0% | 81.5% | **81.5%** | 82.5% |
| push-up | 80 (41/39) | 75.0% | 75.0% | 73.5% | 75.0% | 76.2% | **76.2%** | 82.1% |
| leg extension | 69 (40/29) | 81.2% | 81.1% | 79.7% | 79.7% | 87.0% | **86.9%** | 93.1% |
| squat | 76 (43/33) | 75.0% | 74.6% | 70.9% | 70.7% | 77.6% | **77.3%** | 75.8% |
| chest fly machine | 63 (41/22) | 69.8% | 67.8% | 64.0% | 65.8% | 71.4% | **69.7%** | 68.2% |
| hip thrust | 65 (41/24) | 70.8% | 68.9% | 61.5% | 55.0% | 63.1% | **61.5%** | 58.3% |
| t bar row | 64 (42/22) | 76.6% | 73.7% | 71.4% | 57.9% | 75.0% | **72.3%** | 63.6% |
| leg raises | 63 (41/22) | 79.4% | 77.1% | 74.1% | 73.1% | 82.5% | **81.0%** | 77.3% |

## 4. Phân rã theo Nguồn Dữ liệu (EFA / GYMED / YouTube)
| Nguồn | Số mẫu (Đúng/Sai) | RF Acc | RF Spec (Đúng) | RF Macro-F1 | Hybrid Acc | Hybrid Spec (Đúng) | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EFA | 76 (52/24) | 75.0% | 78.8% | 72.0% | 72.4% | 76.9% | 69.0% | 62.5% |
| GYMED | 142 (141/1) | 78.2% | 78.0% | 46.9% | 78.2% | 78.7% | 43.9% | 0.0% |
| YouTube | 343 (137/206) | 76.1% | 77.4% | 75.6% | 77.6% | 75.9% | 76.9% | 78.6% |

## 5. Đánh giá BiomechanicalRuleEngine trên 231 Mẫu Sai
- **Chẩn đoán thuần (is_incorrect=True):** Top-1 Recall = **42.4%** | Top-2 Recall = **63.2%** | Top-3 Recall = **82.7%**
- **Đầu cuối (qua bộ lọc Hybrid):** Tỷ lệ rep Đúng không bị báo lỗi nhầm = **77.3%** | Tỷ lệ rep Sai vừa bắt đúng Sai vừa gọi đúng lỗi Top-1 = **34.6%**

## 6. Top-10 Đặc trưng Cơ sinh học Quan trọng nhất (RF + ExtraTrees)
- `rep_duration_sec`: `0.0515`
- `ecc_duration_sec`: `0.0413`
- `golden_angle_dev_mean`: `0.0280`
- `golden_dev_max`: `0.0256`
- `golden_dev_at_inflection`: `0.0252`
- `golden_dev_std`: `0.0216`
- `ank_vs_wrist_2d_y_rom`: `0.0205`
- `hip_vs_wrist_2d_x_speed`: `0.0191`
- `rule_top1_score`: `0.0182`
- `rule_top1_rule_score`: `0.0177`

## 7. Các Ca Sai Tự tin Cao nhất của Hybrid (Error Analysis)
| segment_id | Bài tập | Nguồn | Nhãn thật | error_reason | P(incorrect) | Dự đoán Hybrid |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| leg_raises_correct_022 | leg raises | YouTube | correct | none | 88.6% | incorrect |
| squat_incorrect_005 | squat | EFA | incorrect | hips_too_low (Thả rơi mông quá sâu chạm bắp chân làm mất độ căng kiểm soát ở đáy) | 12.9% | correct |
| push_up_correct_023 | push-up | GYMED | correct | none | 86.8% | incorrect |
| pull_up_incorrect_021 | pull up | YouTube | incorrect | shoulders_shrugging (Nhún hai bả vai lên sát tai (Dùng cơ cầu vai thay vì ép chặt bả vai)) | 14.0% | correct |
| pull_up_incorrect_036 | pull up | YouTube | incorrect | neck_extension (Ngửa cổ rướn cằm quá mức để cố qua xà thay vì kéo ngực lên) | 16.6% | correct |
| t_bar_row_incorrect_012 | t bar row | YouTube | incorrect | using_momentum (Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp) | 16.8% | correct |
| t_bar_row_correct_042 | t bar row | YouTube | correct | none | 83.2% | incorrect |
| squat_incorrect_001 | squat | EFA | incorrect | butt_wink (Cụp xương chậu / cuộn tròn lưng dưới ở điểm đáy Squat (Butt Wink)) | 16.9% | correct |