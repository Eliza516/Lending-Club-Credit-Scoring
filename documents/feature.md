# Tài liệu Xử lý Dữ liệu & Kỹ thuật Đặc trưng (Feature Engineering & Preprocessing)

## 1. Định nghĩa nhãn mục tiêu (Choose labels)

Dữ liệu khoản vay Lending Club có 7 trạng thái ban đầu (`loan_status`):
- `Fully Paid`: Trả đủ gốc và lãi -> **Lớp 0 (Tốt / Không vỡ nợ)**
- `Charged Off`: Vỡ nợ / Xóa nợ không thu hồi được -> **Lớp 1 (Xấu / Vỡ nợ)**
- Các trạng thái khác (`Current`, `Late`, `In Grace Period`): Bị loại bỏ vì khoản vay chưa kết thúc chu kỳ thanh toán.

Nguồn tham khảo:
- *Financial Data Analysis with Lending Club* (Medium)
- *Risk assessment in social lending via random forests* (Milad Malekipirbazari, Vural Aksakalli, 2015)
- *Managing Dataset Shift by Adversarial Validation for Credit Scoring* (PRICAI 2022)

---

## 2. Quy tắc loại bỏ đặc trưng (Feature Selection)

Từ 151 cột gốc trong dữ liệu Lending Club, quy trình lọc áp dụng 7 tiêu chí khách quan:
1. **Lọc nhãn & phụ trợ:** Bỏ `loan_status` (nguồn tạo nhãn) và `issue_d` (đã chuẩn hóa sang datetime).
2. **Lọc định danh (ID):** Bỏ `id` và `url` (độ phân mảnh $\ge 99\%$).
3. **Lọc trống 100%:** Bỏ 32 cột không có dòng dữ liệu nào trong tập train.
4. **Lọc rò rỉ thông tin (Data Leakage):** Bỏ 42 cột phát sinh **sau khi cho vay** (`total_pymnt`, `recoveries`, `last_pymnt_d`...).
5. **Lọc chữ quá phân mảnh:** Bỏ các cột văn bản tự do có $> 100$ loại giá trị (`emp_title`, `desc`, `title`, `zip_code`).
6. **Lọc gần như hằng số:** Bỏ 7 cột có một giá trị chiếm $\ge 99\%$ số dòng (`policy_code`, `pymnt_plan`, `delinq_amnt`...).
7. **Lọc thiếu trầm trọng:** Bỏ 5 cột khuyết thiếu $> 30\%$ trên tập train.
$\to$ **Chốt 57 cột ứng viên (`kept_candidates`).**

---

## 3. Kỹ thuật tạo đặc trưng mới (Feature Engineering)

Thực hiện trong Pipeline thông qua hàm `add_features`:
- **`fico_avg`:** `(fico_range_low + fico_range_high) / 2` (ước lượng điểm tâm FICO không chệch, bỏ 2 cột gốc có tương quan $r \approx 1.0$).
- **`credit_history_years`:** `(issue_date - earliest_cr_line).dt.days / 365.25` (thâm niên tín dụng tính bằng năm).
- **`loan_to_income` (LTI):** `(loan_amnt / annual_inc).clip(upper=0.515)` (tỷ lệ khoản vay trên thu nhập năm, cắt trần tại phân vị 99.9%).
- **`log_annual_inc`:** `np.log1p(annual_inc)` (logarit hóa thu nhập, giảm độ lệch phải từ 36.7 xuống 0.19).
- **`log_revol_bal`:** `np.log1p(revol_bal)` (logarit hóa số dư nợ xoay vòng).
- **`emp_length_years`:** Chuyển thâm niên chuỗi text (`"< 1 year"`, `"10+ years"`) thành số từ `0` đến `10`.

---

## 4. Xử lý đa cộng tuyến & Loại bỏ biến trùng lặp nghiệp vụ

Sau khi tạo đặc trưng mới, tiếp tục lọc:
- Bỏ các cặp biến số có tương quan $|r| \ge 0.90$ (chỉ giữ lại biến có ROC-AUC đơn lẻ cao hơn).
- **Đồng nhất sản phẩm 36 tháng:** Toàn bộ dữ liệu được lọc chuẩn hóa chỉ giữ các khoản vay kỳ hạn 36 tháng ngay từ Notebook 1 Mục 3.4 (loại bỏ kỳ hạn 60 tháng do chưa kịp đáo hạn). Cột `term` trở thành hằng số (100% giá trị `" 36 months"`) nên được loại bỏ hoàn toàn khỏi mô hình.
- **Loại bỏ `grade`:** Vì `sub_grade` đã chứa 35 hạng (A1 -> G5) chi tiết hơn 7 hạng của `grade` (A -> G). Giữ cả hai gây đa cộng tuyến hoàn hảo.
- **Loại bỏ `int_rate`:** Trong Lending Club, lãi suất được áp cố định theo từng `sub_grade`. Thực nghiệm cho thấy bỏ `int_rate` hiệu năng ROC-AUC giữ nguyên 0.7010.

---

## 5. Kết quả bộ dữ liệu đầu ra cho Mô hình

- **Phạm vi bài toán:** Chuyên biệt dự đoán rủi ro cho các khoản vay cá nhân kỳ hạn 36 tháng phát hành từ năm 2012.
- **Số dòng train:** 408.188 dòng (tỷ lệ vỡ nợ 13.70%).
- **Cột số cuối cùng (`NUMERIC_FINAL`):** 45 cột (điền trung vị `SimpleImputer` + chuẩn hóa `StandardScaler`).
- **Cột phân loại cuối cùng (`CATEGORICAL_FINAL`):** 5 cột (`sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`).
- **Sau One-Hot Encoding:** 5 biến phân loại nở thành 108 cột nhị phân (0/1).
- **Tổng số chiều ma trận `X_train`:** **153 đặc trưng**, sẵn sàng cho việc huấn luyện mô hình.

---

## 6. Mô hình Tinh gọn 12 Đặc trưng & Thẻ điểm Tín dụng (Compact Scorecard Model)

Bên cạnh mô hình đầy đủ 153 biến, dự án phát triển mô hình thu gọn gồm **12 trường thông tin then chốt** (10 số + 2 phân loại, nở thành 30 cột One-Hot), loại bỏ biến trung gian `sub_grade`:
* **10 biến số:** `fico_avg`, `dti`, `log_annual_inc`, `loan_to_income`, `revol_util`, `inq_last_6mths`, `acc_open_past_24mths`, `total_bal_ex_mort`, `delinq_2yrs`, `loan_amnt`.
* **2 biến phân loại:** `home_ownership`, `purpose`.
* **Hiệu năng & So sánh thẳng thắn:** Test ROC-AUC = `0.6730`, Gini = `34.60%`, KS = `25.07%` (kém hơn Baseline `sub_grade` có AUC `0.6818`, giữ khoảng 84.8% sức mạnh theo Gini so với mô hình 153 biến).
* **Giá trị thực tiễn:** Giảm 80% số chiều biến và quy đổi sang thang điểm số theo phương pháp PDO với $\text{PDO}=20$, $\text{Base}=600$ tại Odds 5:1 (*chưa đối chiếu quy chuẩn chính thức*) (xem chi tiết tại [`experiences/exp_compact_12_features.ipynb`](../experiences/exp_compact_12_features.ipynb)).
