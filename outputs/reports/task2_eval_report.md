# Báo cáo Đánh giá Task 2 (5-Fold Stratified Group Cross-Validation)

## 1. Hậu kiểm Zero Data Leakage
- Tổng số mẫu Out-of-Fold: **521** (`330 correct` / `191 incorrect`)
- Số `segment_id` duy nhất: **521**
- Số nhóm `video_id` độc lập: **371**
- Số `video_id` rò rỉ giữa các Fold: **0**

## 2. So sánh 4 Mô hình qua 5 Fold (Mean ± Std) & Pooled OOF (N = 521)
| Mô hình | CV Accuracy | CV Balanced Acc | CV Macro-F1 | CV Recall (Sai) | CV ROC-AUC | Pooled Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| BiomechanicalRF (106-d E.3+) | 71.99 ± 3.23% | 69.64 ± 3.60% | 69.60 ± 3.46% | 60.81 ± 8.72% | 0.769 ± 0.045 | **69.69%** |
| CNN-BiLSTM + FiLM | 69.87 ± 2.00% | 70.13 ± 1.95% | 68.87 ± 1.90% | 71.18 ± 4.27% | 0.762 ± 0.024 | **68.90%** |
| Transformer + FiLM | 67.79 ± 5.09% | 67.39 ± 4.43% | 66.51 ± 4.81% | 65.97 ± 4.08% | 0.747 ± 0.037 | **66.50%** |
| Hybrid Ensemble (0.50/0.25/0.25) | 73.52 ± 2.73% | 72.27 ± 2.38% | 71.90 ± 2.57% | 67.58 ± 3.48% | 0.777 ± 0.035 | **71.89%** |

### Đối chiếu cặp RF vs Hybrid Ensemble (McNemar Exact)
- Cả 2 cùng đúng: **351** | Hybrid sửa đúng ca RF sai: **32** | RF đúng ca Hybrid sai: **24** | Cả 2 cùng sai: **114** (p-value = `0.3497`).

## 3. Phân rã theo 8 Bài tập Nhóm A (Pooled OOF)
| Bài tập | Tổng (Đúng/Sai) | RF Acc | RF Macro-F1 | CNN-BiLSTM MF1 | Transformer MF1 | Hybrid Acc | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| pull up | 71 (41/30) | 77.5% | 77.2% | 71.7% | 73.2% | 77.5% | **77.2%** | 80.0% |
| push-up | 80 (41/39) | 71.2% | 71.2% | 68.1% | 73.7% | 70.0% | **69.9%** | 76.9% |
| leg extension | 65 (40/25) | 75.4% | 74.0% | 69.0% | 68.3% | 76.9% | **76.4%** | 80.0% |
| squat | 68 (43/25) | 77.9% | 76.1% | 74.3% | 71.3% | 79.4% | **78.2%** | 76.0% |
| chest fly machine | 61 (41/20) | 60.7% | 51.1% | 64.5% | 57.3% | 63.9% | **55.1%** | 30.0% |
| hip thrust | 60 (41/19) | 70.0% | 63.1% | 66.8% | 57.4% | 71.7% | **66.8%** | 52.6% |
| t bar row | 59 (42/17) | 83.1% | 79.3% | 66.5% | 57.0% | 76.3% | **72.0%** | 64.7% |
| leg raises | 57 (41/16) | 57.9% | 47.9% | 58.0% | 56.1% | 71.9% | **66.5%** | 56.2% |

## 4. Phân rã theo Nguồn Dữ liệu (EFA / GYMED / YouTube)
| Nguồn | Số mẫu (Đúng/Sai) | RF Acc | RF Spec (Đúng) | RF Macro-F1 | Hybrid Acc | Hybrid Spec (Đúng) | Hybrid Macro-F1 | Hybrid Recall (Sai) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| EFA | 76 (52/24) | 75.0% | 76.9% | 72.5% | 81.6% | 86.5% | 78.7% | 70.8% |
| GYMED | 142 (141/1) | 78.2% | 78.0% | 46.9% | 73.2% | 73.0% | 44.7% | 100.0% |
| YouTube | 303 (137/166) | 68.3% | 79.6% | 68.3% | 71.6% | 77.4% | 71.6% | 66.9% |

## 5. Đánh giá BiomechanicalRuleEngine trên 191 Mẫu Sai
- **Chẩn đoán thuần (is_incorrect=True):** Top-1 Recall = **47.6%** | Top-2 Recall = **66.0%** | Top-3 Recall = **81.2%**
- **Đầu cuối (qua bộ lọc Hybrid):** Tỷ lệ rep Đúng không bị báo lỗi nhầm = **77.0%** | Tỷ lệ rep Sai vừa bắt đúng Sai vừa gọi đúng lỗi Top-1 = **36.6%**

## 6. Top-10 Đặc trưng Cơ sinh học Quan trọng nhất (RF + ExtraTrees)
- `golden_dev_max`: `0.0356`
- `golden_dev_std`: `0.0343`
- `golden_angle_dev_mean`: `0.0336`
- `golden_dev_at_inflection`: `0.0321`
- `rule_top1_score`: `0.0216`
- `driver_phase1_speed`: `0.0205`
- `rule_top1_rule_score`: `0.0199`
- `wrist_to_knee_dist_min`: `0.0173`
- `shoulder_angle_std`: `0.0167`
- `chin_above_wrist_max`: `0.0159`

## 7. Các Ca Sai Tự tin Cao nhất của Hybrid (Error Analysis)
| segment_id | Bài tập | Nguồn | Nhãn thật | error_reason | P(incorrect) | Dự đoán Hybrid |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| leg_raises_incorrect_008 | leg raises | YouTube | incorrect | using_momentum (Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp) | 11.3% | correct |
| squat_incorrect_001 | squat | EFA | incorrect | butt_wink (Cụp xương chậu / cuộn tròn lưng dưới ở điểm đáy Squat (Butt Wink)) | 13.9% | correct |
| t_bar_row_correct_042 | t bar row | YouTube | correct | none | 85.5% | incorrect |
| leg_raises_incorrect_011 | leg raises | YouTube | incorrect | using_momentum (Dùng đà quán tính quăng giật người thay vì kiểm soát cơ bắp) | 15.8% | correct |
| t_bar_row_incorrect_013 | t bar row | YouTube | incorrect | standing_too_upright (Đứng thân trên quá thẳng (> 60° so với sàn) làm lực chuyển sang cầu vai thay vì lưng giữa) | 16.1% | correct |
| squat_incorrect_005 | squat | EFA | incorrect | hips_too_low (Thả rơi mông quá sâu chạm bắp chân làm mất độ căng kiểm soát ở đáy) | 16.8% | correct |
| chest_fly_machine_incorrect_014 | chest fly machine | YouTube | incorrect | elbow_lockout (Duỗi khóa thẳng đơ khớp khuỷu tay 180° thay vì giữ độ cong nhẹ tự nhiên) | 19.0% | correct |
| t_bar_row_incorrect_015 | t bar row | YouTube | incorrect | uncontrolled_eccentric (Hạ tạ rơi tự do quá nhanh, thiếu kiểm soát ở pha âm (Eccentric)) | 19.9% | correct |