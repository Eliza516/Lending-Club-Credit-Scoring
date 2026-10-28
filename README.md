# Dự đoán vỡ nợ Lending Club (nhánh `simple`)

Dự đoán một khoản vay có bị **xóa nợ (Charged Off)** hay không, chỉ từ thông tin biết được lúc xét duyệt.
Project môn *Học thống kê* (DS66A / DS66B). Nhánh `simple` là bản làm lại **đơn giản, theo từng bước**:
mỗi hàm và mỗi kết quả nằm ở một ô code riêng, và mọi con số/nhận xét đều do code in ra từ dữ liệu thật.

> Bản cũ (4 notebook, nhiều kỹ thuật hơn) nằm trong `archive/`.

## Dữ liệu

Kaggle: [wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club).
Đặt `accepted_2007_to_2018Q4.csv.gz` vào `data/raw/` (file bị git bỏ qua). Không có dữ liệu giả lập:
thiếu file thì notebook dừng và báo rõ.

## Cách chạy

```bash
pip install -r requirements.txt
jupyter lab        # chạy notebook 1 rồi notebook 2, mỗi cái từ trên xuống dưới
```

Notebook 1 mất vài phút (nạp file nén lớn, đỉnh khoảng 10 GB RAM). Notebook 2 mất khoảng 10 phút.

## Các bước đã làm

| Notebook | Bước | Nội dung |
|---|---|---|
| `01_hieu_du_lieu_va_chia_tap` | 1 | Tóm tắt đề bài; từ điển giải thích đủ 151 cột (có `assert` khớp header) |
| | 2 | Nạp **toàn bộ dữ liệu gốc**, đối chiếu từ điển với dữ liệu thật, định nghĩa nhãn bằng bằng chứng |
| | 3 | EDA về cấu trúc thời gian; biểu đồ lý giải chọn mốc 2012; luật "đã đáo hạn"; lọc kỳ hạn 36 tháng; chia **70 / 15 / 15 theo thời gian** (kiểu Arturo-GA) |
| `02_lam_sach_va_feature` | 4 | Làm sạch; pipeline học từ train, áp sang val/test; ô chứng minh không rò rỉ |
| | 5 | Gán 151 cột vào nhóm (mỗi nhóm có bằng chứng), feature engineering, loại đa cộng tuyến, biến hằng số (`term_months`) & biến trùng lặp (`grade`, `int_rate`), 4 biểu đồ lý giải điền trung vị & chuẩn hóa StandardScaler, bộ đặc trưng WoE nâng cao (50 features) |
| `03_logistic` | 6.1 | Huấn luyện & đánh giá Hồi quy Logistic (One-Hot 153 features vs WoE 50 features); kiểm tra quá khớp & giả thuyết chất lượng dữ liệu bureau 2012; ma trận nhầm lẫn & trade-off theo ngưỡng quyết định; phân tích hệ số $\beta$ và Odds Ratio $e^\beta$ |
| `04_random_forest` | 6.2 | Huấn luyện & đánh giá Random Forest; thực nghiệm chứng minh bất biến với co giãn (Scaling Invariance); thử nghiệm lỗi raw string; so sánh 3 phương pháp mã hóa (One-Hot 153f vs Ordinal 50f vs WoE 50f); phân tích quá khớp & Gini Feature Importance |
| `05_gradient_boosting` | 6.3 & Tổng kết | Huấn luyện HistGradientBoosting (Baseline & Tinh chỉnh siêu tham số bằng TimeSeriesSplit trong train); Permutation Importance trên validation; **Bảng tổng hợp đối đầu Benchmark cả 3 mô hình**; Biểu đồ ROC so sánh trực diện; Khuyến nghị triển khai thực tế |

## Kết quả của bước 1-6 (đọc từ output notebook)

**Chia dữ liệu** (notebook 1):

| | Số dòng | Giai đoạn phát hành | Tỷ lệ vỡ nợ |
|---|---|---|---|
| train | 408.188 | 2012-01 đến 2015-05 | 0,137 |
| val | 92.631 | 2015-06 đến 2015-09 | 0,147 |
| test | 88.669 | 2015-10 đến 2015-12 | 0,148 |

- File gốc 2.260.701 dòng, 151 cột; 1.345.310 dòng có nhãn; sau 4 bộ lọc (từ 2012, đã đáo hạn và chỉ lấy kỳ hạn 36 tháng) còn 589.488 dòng.
- Mốc bắt đầu 2012 được chọn dựa trên cấu trúc dữ liệu: 44 cột lịch sử tín dụng bureau bắt đầu xuất hiện từ năm 2012 (tránh để 44 cột này bị khuyết 100% như ở giai đoạn 2010–2011; các cột này vẫn còn tỷ lệ thiếu 5–30% trong giai đoạn 2012–2015 và được điền trung vị ở Bước 5), đồng thời giai đoạn trước 2012 chỉ chiếm dưới 3% số khoản vay có nhãn.
- 100% các khoản vay ở cả 3 tập đều có kỳ hạn 36 tháng, triệt tiêu hoàn toàn lệch kỳ hạn (term shift). Độ lệch tỷ lệ vỡ nợ theo thời gian vẫn tồn tại (train 0,137 vs val 0,147 vs test 0,148, chênh lệch 0,011) do xu hướng nợ xấu tăng dần theo thời gian của Lending Club.

**Chọn feature & Tiền xử lý** (notebook 2):
- 151 cột gốc -> dữ liệu đầu vào đã chuẩn hóa 100% kỳ hạn 36 tháng từ notebook 1 -> loại biến hằng số `term_months`, đa cộng tuyến và 2 biến trùng lặp (`grade` và `int_rate`) -> **45 feature số + 5 feature phân loại**, tức **153 cột** sau mã hóa one-hot.
- **Lý giải Điền trung vị (Median Imputation):** Kiểm chứng trên toàn bộ 31 cột số bị thiếu trong tập train cho thấy đa số đều lệch phải (Skewness từ 0.03 đến 606.4). Giá trị Mean bị ngoại lai kéo cao hơn Median từ +10% đến +138% (ví dụ `bc_open_to_buy` Mean cao hơn Median +138.2%, `avg_cur_bal` cao hơn +100.7%), do đó điền bằng Median giúp bảo toàn tâm phân phối trung thực của đa số người vay.
- **Lý giải Chuẩn hóa (StandardScaler):** Thang đo gốc giữa các biến chênh lệch hơn 100.000 lần (từ 0.3 của `delinq_2yrs` đến 127.000 USD của `tot_cur_bal`). Đưa về $\mu = 0, \sigma = 1$ giúp giải thuật Gradient hội tụ ổn định, phạt điều chuẩn $L_2$ công bằng và cho phép so sánh trực tiếp độ lớn hệ số hồi quy.
- **Bộ đặc trưng Weight of Evidence (WoE - Mục 5.15):** Mã hóa 5 biến phân loại thành 5 biến số log-odds liên tục, tạo ra bộ dữ liệu song song **50 đặc trưng** (`X_train_woe.npy`). Giảm 67.3% số chiều dữ liệu, tăng tốc độ huấn luyện 3–5 lần, ROC-AUC của HistGB đạt **0.7097** và Random Forest đạt **0.7040**.

**Huấn luyện & Đánh giá mô hình Hồi quy Logistic** (notebook 3):
- **Phạm vi bài toán:** Dự đoán rủi ro cho các khoản vay cá nhân kỳ hạn 36 tháng (đồng nhất 100% giữa train, val, test).
- **Hiệu năng trên tập Validation (2015-06 đến 2015-09):**
  - Không gian One-Hot (153f): ROC-AUC `0.7031` (Gini = `40.62%`, KS = `29.83%`).
  - Không gian WoE (50f): ROC-AUC `0.7033` (Gini = `40.66%`, KS = `29.75%`).
- **Kiểm tra quá khớp & diễn biến theo thời gian:** Không có dấu hiệu quá khớp (Train AUC `0.6970` vs Val AUC `0.7031`). Hiện tượng Val cao hơn Train một phần do chất lượng dữ liệu: kiểm tra trên năm 2012 cho thấy nhóm đủ cột bureau đạt AUC `0.6658` cao hơn mức chung `0.6630` (+0.0028).
- **Phân loại theo ngưỡng & Ma trận nhầm lẫn:** Khảo sát trade-off giữa tỷ lệ từ chối hồ sơ (Reject Rate) và khả năng bắt đúng nợ xấu (Recall) qua các ngưỡng từ 0.10 đến 0.30.
- **Phân tích hệ số $\beta$ và Odds Ratio $e^\beta$:** Nhận diện các yếu tố rủi ro hàng đầu (lãi suất `int_rate`, tỷ lệ nợ `dti`) và các yếu tố bảo vệ (hạng tín dụng tốt, điểm FICO cao, thu nhập hàng năm lớn).

**Huấn luyện & Đánh giá mô hình Random Forest** (notebook 4):
- **Bản chất tiền xử lý đối với mô hình cây:**
  - *Tính bất biến với phép co giãn (Scaling Invariance):* Chứng minh lý thuyết và thực nghiệm rằng co giãn dữ liệu (`StandardScaler`) có chênh lệch AUC $< 0.0001$ so với dữ liệu chưa scale. Random Forest hoàn toàn không cần chuẩn hóa scale.
  - *Dữ liệu chuỗi thô (Raw String):* Scikit-learn ném ngoại lệ `ValueError: could not convert string to float` nếu đưa chuỗi thô vào `RandomForestClassifier`. Bắt buộc phải mã hóa thành số.
- **So sánh 3 kỹ thuật mã hóa biến phân loại trên tập Validation:**
  - `One-Hot Encoding (153 đặc trưng)`: ROC-AUC = **0.6892** (108 cột dummy thưa thớt làm loãng việc bốc ngẫu nhiên `max_features = sqrt(p)` tại mỗi phân nhánh).
  - `Ordinal Encoding (50 đặc trưng, unscaled)`: ROC-AUC = **0.7033** (giữ nguyên 50 chiều, cây tìm ngưỡng phân nhánh số nguyên hiệu quả, tăng vọt +0.0141 AUC).
  - `Weight of Evidence (50 đặc trưng)`: ROC-AUC = **0.7040** (đơn điệu hóa rủi ro, tăng tốc độ và độ chính xác phân hạng).
- **Đặc trưng quá khớp:** Train AUC $\approx 0.81$ vs Val AUC $\approx 0.7040$ do các cây phát triển sâu (`max_depth=12`), nhưng 100 cây kết hợp (ensemble) giúp kiểm soát phương sai rất tốt trên tập kiểm định độc lập.
- **Gini Feature Importance (MDI):** Xác định Top 20 đặc trưng chi phối khả năng phân loại (lãi suất, điểm FICO, thời gian trả nợ gần nhất).

**Huấn luyện HistGradientBoosting & Tổng kết Benchmark** (notebook 5):
- **Cơ chế HistGradientBoosting:** Phân khoảng 256 bins nguyên (`uint8`) giúp tăng tốc độ huấn luyện gấp hàng chục lần; xây dựng cây tuần tự sửa lỗi gradient của các cây trước; hỗ trợ giá trị khuyết bản địa.
- **Tinh chỉnh siêu tham số (Hyperparameter Tuning):** Sử dụng `TimeSeriesSplit(n_splits=3)` hoàn toàn bên trong tập `train` (2012-01 đến 2015-05) với `refit=True`, đảm bảo tuyệt đối không có rò rỉ dữ liệu tương lai từ tập `val`.
- **Permutation Importance:** Đo mức sụt giảm ROC-AUC trên 5,000 mẫu Validation khi xáo trộn từng đặc trưng, khắc phục nhược điểm thiên vị số lượng giá trị duy nhất của Gini MDI.
- **Bảng Tổng Hợp Đối Đầu (Head-to-head Benchmark) Cả 3 Họ Mô Hình trên Tập Validation:**

| Họ mô hình | Cấu hình / Không gian | ROC-AUC | Gini (%) | KS (%) | Brier | LogLoss |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting** | **WoE Tuned (50f)** | **0.7097** | **41.94%** | **31.01%** | **0.11675** | **0.38296** |
| HistGradientBoosting | WoE Baseline (50f) | 0.7091 | 41.83% | 30.97% | 0.11685 | 0.38325 |
| HistGradientBoosting | One-Hot Baseline (153f) | 0.7070 | 41.40% | 30.51% | 0.11710 | 0.38402 |
| HistGradientBoosting | One-Hot Tuned (153f) | 0.7056 | 41.13% | 30.45% | 0.11729 | 0.38473 |
| **Random Forest** | **WoE (50 features)** | **0.7040** | **40.80%** | **30.10%** | 0.11785 | 0.38644 |
| Random Forest | Ordinal (50 features) | 0.7033 | 40.66% | 30.12% | 0.14580 | 0.46450 |
| **Logistic Regression** | **WoE (50 features)** | **0.7033** | **40.66%** | **29.75%** | 0.11749 | 0.38546 |
| Logistic Regression | One-Hot (153 features) | 0.7031 | 40.62% | 29.83% | 0.11751 | 0.38555 |
| Random Forest | One-Hot (153 features) | 0.6892 | 37.84% | 27.85% | 0.14720 | 0.46820 |

- **Kết luận Kỹ thuật & Khuyến nghị Lựa chọn Mô hình:**
  1. *Năng lực phân loại cao nhất:* `HistGradientBoosting (WoE Tuned)` dẫn đầu toàn diện với ROC-AUC `0.7097`, Gini `41.94%`, KS `31.01%`. Phù hợp triển khai khi mục tiêu là tối đa hóa tỷ suất sinh lời và hạn chế nợ xấu.
  2. *Tính minh bạch và tuân thủ quy chuẩn tín dụng:* `Logistic Regression (WoE)` đạt ROC-AUC `0.7033`, có hệ số $\beta$ cố định, dễ dàng chuyển đổi thành bảng điểm Scorecard truyền thống theo điểm chuẩn (Scorecard Points).
  3. *Tác động của không gian đặc trưng:* Bộ đặc trưng WoE (50 chiều) vượt trội toàn diện so với One-Hot (153 chiều) trên mọi mô hình cây (tăng từ 0.6892 lên 0.7040 trên Random Forest, giảm thời gian huấn luyện 3-5 lần).

## Hạn chế đã biết (đều do notebook đo được)

1. **Phạm vi mô hình chuyên biệt cho khoản vay 36 tháng** (100% đồng nhất giữa train, val, test). Khoản vay 60 tháng đã được chủ động loại bỏ ngay từ Mục 3.4 do không đủ thời gian đáo hạn ở giai đoạn 2015. Test chỉ gồm 3 tháng cuối 2015.
2. **32 cột trống hoàn toàn trong train**, trong đó 17 cột chỉ bắt đầu có dữ liệu ở giai đoạn val/test, nên không dùng được.
3. **Các cột dạng "số tháng kể từ sự kiện"** bị loại vì thiếu trên 30% trong train, dù giá trị thiếu có thể có nghĩa là "chưa từng xảy ra"; chưa kiểm tra bằng code xem việc đó có mất tín hiệu không.
4. `issue_d` (ngày giải ngân) được dùng thay cho ngày nộp đơn; chỉ có khoản vay đã được chấp nhận trong dữ liệu.
5. Cách chia này **thay cho mốc chung 2016-01-01** của bản cũ; các nhóm khác dùng cùng dataset cần đối chiếu `md5_id_train` trong `artifacts/split_manifest.json`.
6. **Tỷ lệ các khoản vay đã có kết quả (resolved rate) ở các tháng 10–12/2015 (giai đoạn của tập test)** có xu hướng giảm nhẹ (từ xấp xỉ 100% ở các tháng trước xuống 99.68% vào tháng 12/2015, tương ứng khoảng 0.32% khoản vay chưa đáo hạn hoàn toàn do nằm sát mốc snapshot 12/2018).

## Ghi chú về repo

- `CLAUDE.md`: quy tắc làm việc cho các phiên Claude Code (nguyên tắc "mọi kết luận có code suy ra").
- `docs/`: tài liệu của bản cũ, có số liệu từ nguồn ngoài và các bước không còn trong notebook mới.
  Coi là lỗi thời cho đến khi được viết lại.
