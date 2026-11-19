# Báo cáo Đánh giá Task 2 (5-Fold Stratified Group Cross-Validation)

## 1. Hậu kiểm Zero Data Leakage
- Tổng số mẫu Out-of-Fold: **561** (`330 correct` / `231 incorrect`)
- Số `segment_id` duy nhất: **561**
- Số nhóm `video_id` độc lập: **405**
- Số `video_id` rò rỉ giữa các Fold: **0**

## 2. So sánh 4 Mô hình qua 5 Fold (Mean ± Std) & Pooled OOF (N = 561)
| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC | Pooled Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BiomechanicalRF (106-d E.3+) | 72.02 ± 4.75% | 71.15 ± 4.79% | 71.09 ± 4.81% | 66.23 ± 6.96% | 0.774 ± 0.047 | **71.13%** |
| CNN-BiLSTM + FiLM | 67.91 ± 4.01% | 67.86 ± 4.51% | 67.36 ± 4.24% | 67.53 ± 8.28% | 0.763 ± 0.042 | **67.42%** |
| Transformer + FiLM | 68.99 ± 4.09% | 68.83 ± 4.13% | 68.41 ± 4.07% | 67.96 ± 6.46% | 0.750 ± 0.043 | **68.45%** |
| Hybrid Ensemble (0.50/0.25/0.25) | 71.13 ± 3.78% | 70.46 ± 4.03% | 70.33 ± 3.95% | 66.67 ± 5.64% | 0.780 ± 0.043 | **70.34%** |

### Đối chiếu cặp RF vs Hybrid Ensemble (McNemar Exact)
- Cả 2 cùng đúng: **364** | Hybrid sửa đúng ca RF sai: **35** | RF đúng ca Hybrid sai: **40** | Cả 2 cùng sai: **122** (p-value = `0.6445`).

## 3. Phân rã theo 8 Bài tập Nhóm A (Pooled OOF)
| Bài tập | Tổng (Đúng/Sai) | RF Acc | RF Macro-F1 | CNN-BiLSTM MF1 | Transformer MF1 | Hybrid Acc | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pull up | 81 (41/40) | 76.5% | 76.5% | 71.5% | 74.1% | 76.5% | **76.5%** | 80.0% |
| push-up | 80 (41/39) | 78.8% | 78.7% | 79.9% | 78.7% | 76.2% | **76.2%** | 82.1% |
| leg extension | 69 (40/29) | 69.6% | 68.6% | 70.5% | 73.0% | 76.8% | **76.0%** | 69.0% |
| squat | 76 (43/33) | 81.6% | 80.9% | 75.9% | 77.3% | 80.3% | **79.7%** | 72.7% |
| chest fly machine | 63 (41/22) | 58.7% | 54.6% | 65.1% | 63.0% | 68.3% | **65.1%** | 54.5% |
| hip thrust | 65 (41/24) | 72.3% | 69.1% | 51.6% | 57.8% | 61.5% | **59.1%** | 50.0% |
| t bar row | 64 (42/22) | 67.2% | 65.3% | 62.0% | 59.3% | 71.9% | **69.5%** | 63.6% |
| leg raises | 63 (41/22) | 66.7% | 61.0% | 50.8% | 53.6% | 52.4% | **48.6%** | 36.4% |

## 4. Phân rã theo Nguồn Dữ liệu (EFA / GYMED / YouTube)
| Nguồn | Số mẫu (Đúng/Sai) | RF Acc | RF Spec (Đúng) | RF Macro-F1 | Hybrid Acc | Hybrid Spec (Đúng) | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EFA | 76 (52/24) | 77.6% | 82.7% | 74.4% | 78.9% | 84.6% | 75.6% | 66.7% |
| GYMED | 142 (141/1) | 71.8% | 71.6% | 44.1% | 70.4% | 70.2% | 43.5% | 100.0% |
| YouTube | 343 (137/206) | 70.8% | 78.1% | 70.6% | 69.7% | 74.5% | 69.4% | 66.5% |

## 5. Đánh giá BiomechanicalRuleEngine trên 231 Mẫu Sai
- **Chẩn đoán thuần (is_incorrect=True):** Top-1 Recall = **43.3%** | Top-2 Recall = **62.3%** | Top-3 Recall = **79.7%**
- **Đầu cuối (qua bộ lọc Hybrid):** Tỷ lệ rep Đúng không bị báo lỗi nhầm = **74.2%** | Tỷ lệ rep Sai vừa bắt đúng Sai vừa gọi đúng lỗi Top-1 = **32.0%**

## 6. Top-10 Đặc trưng Cơ sinh học Quan trọng nhất (RF + ExtraTrees)
- `golden_angle_dev_mean`: `0.0343`
- `golden_dev_max`: `0.0320`
- `golden_dev_at_inflection`: `0.0298`
- `golden_dev_std`: `0.0261`
- `rule_top1_score`: `0.0221`
- `rule_top1_rule_score`: `0.0211`
- `shoulder_angle_std`: `0.0185`
- `driver_phase1_speed`: `0.0166`
- `chin_above_wrist_at_inflection`: `0.0165`
- `driver_peak_speed`: `0.0155`

## 7. Các Ca Sai Tự tin Cao nhất của Hybrid (Error Analysis)
| segment_id | Bài tập | Nguồn | Nhãn thật | error_reason | P(incorrect) | Dự đoán Hybrid |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| leg_raises_incorrect_011 | leg raises | YouTube | incorrect | using_momentum (Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp) | 10.1% | correct |
| push_up_correct_023 | push-up | GYMED | correct | none | 86.4% | incorrect |
| pull_up_incorrect_036 | pull up | YouTube | incorrect | neck_extension (Ngửa cổ rướn cằm quá mức để cố qua xà thay vì kéo ngực lên) | 15.3% | correct |
| leg_extension_incorrect_013 | leg extension | YouTube | incorrect | torso_swinging (Đung đưa / giật ngả thân trên theo từng nhịp đá để mượn đà nâng tạ) | 15.4% | correct |
| push_up_correct_010 | push-up | EFA | correct | none | 84.5% | incorrect |
| leg_raises_incorrect_018 | leg raises | YouTube | incorrect | using_momentum (Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp) | 16.0% | correct |
| leg_raises_correct_022 | leg raises | YouTube | correct | none | 83.3% | incorrect |
| squat_incorrect_005 | squat | EFA | incorrect | hips_too_low (Thả rơi mông quá sâu chạm bắp chân làm mất độ căng kiểm soát ở đáy) | 17.2% | correct |