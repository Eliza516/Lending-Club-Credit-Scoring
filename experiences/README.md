# Thử Nghiệm Mở Rộng Trong Thư Mục `experiences/`

Thư mục này chứa mã nguồn và báo cáo thử nghiệm độc lập, nhằm giải quyết hai câu hỏi nghiên cứu mở rộng:
1. **Thử nghiệm 1:** Nếu loại bỏ hoàn toàn `sub_grade` ra khỏi tập đặc trưng, mô hình Hồi quy Logistic có bị tệ hơn mốc Baseline (`sub_grade` rule) hay không và các trọng số rủi ro dịch chuyển ra sao?
2. **Thử nghiệm 2:** Nếu chỉ sử dụng một tập hợp nhỏ 12 trường thông tin cốt lõi (không dùng `sub_grade`) và quy đổi sang thang điểm dạng điểm số (Points-to-Double-Odds / PDO), hiệu năng phân loại và chính sách phê duyệt sẽ như thế nào?

> [!NOTE]
> **Lưu ý về tập Test Out-of-Time:** Tập Test (88.669 khoản vay) được dùng thêm ở đây cho mục đích phân tích học thuật bổ sung, hoàn toàn không được dùng để lựa chọn hay tinh chỉnh mô hình.

---

## 1. Thử Nghiệm 1: Đánh Giá Vai Trò `sub_grade` & Phân Tích Dịch Chuyển Hệ Số

Notebook thực thi đầy đủ: [`experiences/exp_logistic_subgrade.ipynb`](exp_logistic_subgrade.ipynb)

### 1.1. Bảng Kết Quả Đánh Giá Hiệu Năng

#### Đánh giá trên Tập Validation (92.631 khoản vay):

| Mô hình | Số đặc trưng | ROC-AUC Train | ROC-AUC Val | Chênh lệch (Train - Val) | Gini Val | KS Val |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline `sub_grade`** | 1 (sub_grade) | 0.6565 | 0.6815 | -0.0250 | 36.31% | 27.02% |
| **2. Logistic L2 (+sub_grade)** | 153 biến | 0.6873 | **0.7040** | -0.0167 | **40.79%** | **29.94%** |
| **3. Logistic L2 (BỎ sub_grade)** | 118 biến | 0.6775 | **0.6890** | -0.0115 | **37.80%** | **27.68%** |

#### Đánh giá bổ sung trên Tập Test Out-of-Time (88.669 khoản vay):

| Mô hình | Test ROC-AUC | Chênh lệch so với Baseline ($\Delta \text{AUC}$) | Test Gini | Test KS |
|---|:---:|:---:|:---:|:---:|
| **1. Baseline `sub_grade`** | 0.6818 | +0.0000 | 36.36% | 26.45% |
| **2. Logistic L2 (+sub_grade)** | **0.7039** | **+0.0221** | **40.79%** | **29.87%** |
| **3. Logistic L2 (BỎ sub_grade)** | **0.6879** | **+0.0061** | **37.58%** | **27.34%** |

### 1.2. Kiểm Định Thống Kê Bootstrap 95% CI ($B = 1.000$ trên tập Validation)

1. **So sánh Logistic (Không `sub_grade`) vs Baseline `sub_grade` trên Validation:**
   * $\Delta \text{AUC} = \mathbf{+0.0075}$
   * 95% Bootstrap Confidence Interval: **`[+0.0033, +0.0117]`**
   * **Kết luận:** Khoảng tin cậy hoàn toàn mang dấu dương (không chứa 0). Logistic khi không có `sub_grade` **vượt trội hơn Baseline có ý nghĩa thống kê trên tập Validation**. Trên tập Test, mô hình đạt mức chênh lệch dương $+0.0061$ (phép đo điểm duy nhất trên Test, không tính CI).
2. **Đóng góp của `sub_grade` đối với Logistic trên Validation:**
   * $\Delta \text{AUC} = \mathbf{+0.0150}$, 95% CI: `[+0.0133, +0.0167]`.

### 1.3. Hiện Tượng Dịch Chuyển Hệ Số Hồi Quy (Coefficients Shift Analysis)

Khi so sánh giữa mô hình có `sub_grade` (153 biến) và bỏ `sub_grade` (118 biến):
* **Hấp thụ tín hiệu (Signal Absorption):** 35 dummy variables của `sub_grade` thâu tóm phần lớn tín hiệu rủi ro, nén nhỏ độ lớn hệ số của các biến đặc trưng tài chính gốc.
* **Tái phân bổ trọng số (Weight Re-allocation):** Khi gỡ bỏ `sub_grade`, mô hình tự động tăng mạnh trọng số cho các biến gốc của người vay:
  * Điểm FICO (`fico_avg`): Hệ số tăng độ lớn từ `-0.1067` lên **`-0.2374`** (tăng 2.22 lần).
  * Thu nhập (`log_annual_inc`): Hệ số tăng từ `-0.0386` lên **`-0.1021`** (tăng 2.64 lần).
  * Tỷ lệ dùng thẻ (`revol_util`): Hệ số tăng từ `+0.0521` lên **`+0.0978`** (tăng 1.88 lần).
  * Lượt tra cứu tín dụng 6 tháng (`inq_last_6mths`): Hệ số tăng từ `+0.0494` lên **`+0.0951`** (tăng 1.92 lần).
  * Tỷ lệ nợ trên thu nhập (`dti`): Hệ số tăng từ `+0.1277` lên **`+0.1583`** (tăng 1.24 lần).

> Toàn bộ 153 hệ số được lưu tại: [`coefficients_comparison.csv`](coefficients_comparison.csv).  
> Biểu đồ ROC: [`figures/exp_no_subgrade_comparison.png`](figures/exp_no_subgrade_comparison.png).  
> Biểu đồ dịch chuyển hệ số: [`figures/exp_coefficients_shift.png`](figures/exp_coefficients_shift.png).

---

## 2. Thử Nghiệm 2: Mô Hình Tinh Gọn 12 Đặc Trưng & Thẻ Điểm Tín Dụng

Notebook thực thi đầy đủ: [`experiences/exp_compact_12_features.ipynb`](exp_compact_12_features.ipynb)

Mô hình gồm **12 trường thông tin cốt lõi** (10 biến số + 2 biến phân loại One-Hot thành 30 cột, không dùng `sub_grade`):
* **10 biến số:** `fico_avg`, `dti`, `log_annual_inc`, `loan_to_income`, `revol_util`, `inq_last_6mths`, `acc_open_past_24mths`, `total_bal_ex_mort`, `delinq_2yrs`, `loan_amnt`.
* **2 biến phân loại:** `home_ownership`, `purpose`.

### 2.1. So Sánh Hiệu Năng & Kết Quả Thẳng Thắn

| Mô hình | Số biến One-Hot | Val ROC-AUC | Val Gini | Test ROC-AUC | Test Gini |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline `sub_grade`** | 1 (rule) | **0.6815** | **36.31%** | **0.6818** | **36.36%** |
| **2. Logistic L2 Đầy đủ (+sub_grade)** | 153 cột | **0.7040** | **40.79%** | **0.7039** | **40.79%** |
| **3. Mô hình Tinh gọn (12 đặc trưng)** | 30 cột | 0.6750 | 34.99% | 0.6730 | 34.60% |

* **Ghi nhận thẳng thắn:** Mô hình 12 biến **kém hơn mốc Baseline `sub_grade`** (Val AUC `0.6750` so với `0.6815`; Test AUC `0.6730` so với `0.6818`).
* **Sức mạnh phân loại (theo Gini):** Giữ được khoảng **84.8%** sức mạnh phân loại của mô hình đầy đủ (Test Gini $34.60\% / 40.79\%$), trong khi cắt giảm 80% số chiều dữ liệu (từ 153 xuống 30 cột).
* **Quy đổi Thang điểm (Points-to-Double-Odds / PDO):** Thiết lập $PDO=20$, $\text{Base Score}=600$ tại Odds 5:1 (phương pháp thang điểm phổ biến trong ngành, *chưa đối chiếu quy định chuẩn hóa quốc tế*). Ngưỡng phân tầng được xác định từ phân phối điểm của tập Validation và kiểm chứng trên tập Test.

> Biểu đồ phân bố điểm số: [`figures/exp_compact_scorecard_distribution.png`](figures/exp_compact_scorecard_distribution.png).

---

## 3. Cách Tái Lập Thử Nghiệm

Mở và chạy trực tiếp các notebook (theo nguyên tắc không tạo file .py riêng của repo):
1. [`experiences/exp_logistic_subgrade.ipynb`](exp_logistic_subgrade.ipynb): Tái lập toàn bộ Thử nghiệm 1.
2. [`experiences/exp_compact_12_features.ipynb`](exp_compact_12_features.ipynb): Tái lập toàn bộ Thử nghiệm 2.
