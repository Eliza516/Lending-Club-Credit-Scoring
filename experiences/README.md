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

## 4. Phân Tích Dịch Chuyển Hệ Số Hồi Quy (Coefficients Shift Analysis)

Khi so sánh chi tiết giữa mô hình **Logistic CÓ `sub_grade`** (153 biến) và **Logistic BỎ `sub_grade`** (118 biến):

* **Hiện tượng Hấp thụ tín hiệu (Signal Absorption):** Khi có mặt `sub_grade`, 35 dummy variables của biến này đóng vai trò "chiếc ô thâu tóm" phần lớn rủi ro đã được Lending Club chấm điểm trước, khiến độ lớn hệ số của các biến gốc bị nén nhỏ lại.
* **Hiện tượng Tái phân bổ trọng số (Weight Re-allocation):** Khi gỡ bỏ `sub_grade`, mô hình tự động "giải phóng" và gia tăng trọng số mạnh mẽ cho các biến tài chính gốc của người vay:
  * **Điểm tín dụng FICO (`fico_avg`):** Hệ số nhảy vọt từ `-0.1067` lên **`-0.2374`** (tăng **2.22 lần** độ mạnh trong việc giảm thiểu rủi ro nợ xấu).
  * **Thu nhập (`log_annual_inc`):** Hệ số tăng từ `-0.0386` lên **`-0.1021`** (tăng **2.64 lần** tác động giảm rủi ro).
  * **Tỷ lệ dùng thẻ tín dụng (`revol_util`):** Hệ số tăng từ `+0.0521` lên **`+0.0978`** (tăng **1.88 lần** mức độ cảnh báo rủi ro).
  * **Số lần tra cứu tín dụng 6 tháng (`inq_last_6mths`):** Hệ số tăng từ `+0.0494` lên **`+0.0951`** (tăng **1.92 lần** mức độ rủi ro).
  * **Tỷ lệ nợ trên thu nhập (`dti`):** Hệ số tăng từ `+0.1277` lên **`+0.1583`** (tăng **1.24 lần**).

> Toàn bộ 153 hệ số được xuất chi tiết tại: [`experiences/coefficients_comparison.csv`](coefficients_comparison.csv).

---

## 5. Biểu Đồ Trực Quan Hóa

1. Biểu đồ so sánh đường cong ROC và các thước đo rủi ro:  
   [`experiences/figures/exp_no_subgrade_comparison.png`](figures/exp_no_subgrade_comparison.png)
2. Biểu đồ phân tích dịch chuyển hệ số hồi quy:  
   [`experiences/figures/exp_coefficients_shift.png`](figures/exp_coefficients_shift.png)

---

## 6. Cách Tái Lập Thử Nghiệm

Chạy các lệnh sau từ thư mục gốc của project:
```bash
# 1. Chạy đánh giá hiệu năng & Bootstrap CI
python3 experiences/exp_logistic_no_subgrade.py

# 2. Chạy so sánh chi tiết hệ số hồi quy & xuất bảng CSV
python3 experiences/exp_compare_coefficients.py
```
