# Thử Nghiệm: Đánh Giá Hiệu Năng Mô Hình Logistic Khi Loại Bỏ `sub_grade`

Thư mục này chứa mã nguồn và báo cáo thử nghiệm độc lập, nhằm trả lời câu hỏi:
> **"Nếu loại bỏ hoàn toàn `sub_grade` ra khỏi tập đặc trưng, mô hình Hồi quy Logistic có bị tệ hơn mốc Baseline (`sub_grade` rule) hay không?"**

---

## 1. Thiết Kế Thử Nghiệm

* **Baseline (`sub_grade` rule):** Mốc quy tắc xếp hạng tín dụng nội bộ của Lending Club (gán xác suất vỡ nợ lịch sử theo 35 hạng `sub_grade` từ A1 đến G5 trên tập train).
* **Mô hình Logistic Đầy đủ (+`sub_grade`):** Không gian gốc 153 đặc trưng (gồm 45 biến số, 5 biến phân loại mã hóa One-Hot với 35 dummy variables của `sub_grade`).
* **Mô hình Logistic BỎ `sub_grade`:** Chỉ sử dụng **118 đặc trưng thuần túy** về thông tin người vay và lịch sử tín dụng bureau (FICO, DTI, thu nhập, số tài khoản tín dụng, mục đích vay, loại nhà ở,...), loại bỏ triệt để toàn bộ 35 dummy variables của `sub_grade`.

---

## 2. Bảng Kết Quả Chi Tiết

### Đánh giá trên Tập Validation (92.631 khoản vay):

| Mô hình | Số đặc trưng | ROC-AUC Train | ROC-AUC Val | Chênh lệch (Train - Val) | Gini Val | KS Val |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline `sub_grade`** | 1 (sub_grade) | 0.6565 | 0.6815 | -0.0250 | 36.31% | 27.02% |
| **2. Logistic L2 (+sub_grade)** | 153 biến | 0.6873 | **0.7040** | -0.0167 | **40.79%** | **29.94%** |
| **3. Logistic L2 (BỎ sub_grade)** | 118 biến | 0.6775 | **0.6890** | -0.0115 | **37.80%** | **27.68%** |

---

### Đánh giá trên Tập Test Out-of-Time (88.669 khoản vay - Duy nhất 1 lần):

| Mô hình | Test ROC-AUC | Chênh lệch so với Baseline ($\Delta \text{AUC}$) | Test Gini | Test KS |
|---|:---:|:---:|:---:|:---:|
| **1. Baseline `sub_grade`** | 0.6818 | +0.0000 | 36.36% | 26.45% |
| **2. Logistic L2 (+sub_grade)** | **0.7039** | **+0.0221** | **40.79%** | **29.87%** |
| **3. Logistic L2 (BỎ sub_grade)** | **0.6879** | **+0.0061** | **37.58%** | **27.34%** |

---

## 3. Kiểm Định Thống Kê Bootstrap 95% CI ($B = 1.000$)

1. **So sánh Logistic (Không `sub_grade`) vs Baseline `sub_grade`:**
   * $\Delta \text{AUC} = \mathbf{+0.0075}$
   * 95% Bootstrap Confidence Interval: **`[+0.0033, +0.0117]`**
   * **Kết luận:** Khoảng tin cậy hoàn toàn mang dấu dương (không chứa 0). Logistic khi không có `sub_grade` **vẫn vượt trội hơn Baseline có ý nghĩa thống kê** trên cả tập Validation và tập Test.
2. **Giá trị gia tăng của `sub_grade` đối với Logistic:**
   * $\Delta \text{AUC} = \mathbf{+0.0150}$
   * 95% Bootstrap Confidence Interval: **`[+0.0133, +0.0167]`**
   * **Kết luận:** `sub_grade` đóng góp thêm **+0.0150 AUC** khi kết hợp cùng 45 đặc trưng tài chính người vay.

---

## 4. Biểu Đồ Trực Quan Hóa

Biểu đồ so sánh đường cong ROC và các thước đo rủi ro được lưu tại:  
`experiences/figures/exp_no_subgrade_comparison.png`

---

## 5. Cách Tái Lập Thử Nghiệm

Chạy lệnh sau từ thư mục gốc của project:
```bash
python3 experiences/exp_logistic_no_subgrade.py
```
