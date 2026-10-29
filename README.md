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
| `03_logistic` | 6 | Huấn luyện & so sánh **5 mô hình** (Baseline sub_grade, Logistic L2, Decision Tree nông, Random Forest nông, HistGradientBoosting); thước đo ROC-AUC, Gini, KS; kiểm tra quá khớp (Train vs Val AUC); khoảng tin cậy Bootstrap 95% ($B=1,000$); đánh giá tập Test duy nhất 1 lần, kết luận bằng f-string & hạn chế |

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

**Huấn luyện & So sánh 5 Mô hình Phân loại Nợ Xấu** (notebook 3):

Thiết kế đơn giản, minh bạch với đúng 9 nội dung cốt lõi trên cùng không gian 153 đặc trưng One-Hot:

1. **Baseline `sub_grade`**: Mốc quy tắc điểm tín dụng nội bộ có sẵn của Lending Club (tỷ lệ nợ xấu lịch sử theo từng sub-grade trên train), bắt buộc có để chứng minh mô hình Machine Learning "hơn gì cái đã có".
2. **Logistic L2 ($C=1$)**: Mô hình cơ bản, dễ giải thích, chuẩn mực đối chiếu trong thẩm định tín dụng.
3. **Decision Tree nông (`max_depth=5`, `min_samples_leaf=50`)**: Mô hình phi tuyến đơn giản nhất, nắm bắt tương tác mà không bị quá khớp.
4. **Random Forest (cây nông, 100 cây, `max_depth=6`, `min_samples_leaf=50`)**: Đại diện kỹ thuật Bagging (nhiều cây độc lập).
5. **HistGradientBoosting (100 cây, `max_depth=5`, `min_samples_leaf=50`)**: Đại diện kỹ thuật Boosting (nhiều cây nối tiếp sửa sai gradient).
6. **Thước đo chuẩn mực:** ROC-AUC, Gini ($2 \times \text{AUC} - 1$), KS (Kolmogorov-Smirnov). Không cần cắt ngưỡng nhị phân.
7. **Kiểm tra quá khớp (Overfitting Check):** Đo chênh lệch Train AUC vs Val AUC.
8. **Khoảng tin cậy Bootstrap 95% ($B=1,000$):** Kiểm định ý nghĩa thống kê cho $\Delta \text{AUC}$ so với Baseline `sub_grade`.
9. **Đánh giá duy nhất 1 lần trên tập Test:** Khẳng định hiệu năng thực tế, in kết luận bằng `f-string` và nêu rõ hạn chế.

**Bảng So Sánh Hiệu Năng và Kiểm Tra Quá Khớp trên Tập Validation (92.631 khoản vay):**

| Mô hình | ROC-AUC Train | ROC-AUC Val | Chênh lệch (Train - Val) | Gini Val | KS Val |
|---|:---:|:---:|:---:|:---:|:---:|
| 1. Baseline `sub_grade` | 0.6565 | 0.6815 | -0.0250 | 36.31% | 27.02% |
| 2. **Logistic L2 ($C=1$)** | 0.6873 | **0.7040** | -0.0167 | **40.79%** | **29.94%** |
| 3. Decision Tree nông (depth=7) | 0.6541 | 0.6536 | +0.0006 | 30.71% | 22.25% |
| 4. Random Forest (cây nông, depth=8) | 0.6853 | 0.6836 | +0.0017 | 36.71% | 26.42% |
| 5. **HistGradientBoosting** | 0.7125 | **0.7034** | +0.0091 | **40.68%** | **29.99%** |

*Ghi chú về hiện tượng Val AUC cao hơn Train AUC (chênh lệch âm ở Baseline, Logistic):* Chưa rõ nguyên nhân chính xác (có thể do dịch chuyển cơ cấu danh mục vay hoặc tỷ lệ báo cáo đầy đủ hơn ở giai đoạn 2015), nhưng khẳng định **không có dấu hiệu quá khớp**.

**Khoảng Tin Cậy Bootstrap 95% cho Chênh Lệch so với Baseline (Tập Validation):**

| Mô hình | Delta AUC | 95% CI (Bootstrap) | Hơn baseline có thật không? |
|---|:---:|:---:|:---:|
| 2. **Logistic L2 ($C=1$)** | **+0.0225** | **[+0.0196, +0.0253]** | **Có hơn thật ($p < 0.05$)** |
| 3. Decision Tree nông | -0.0280 | [-0.0329, -0.0233] | Không hơn (kém hơn) |
| 4. Random Forest (cây nông) | +0.0020 | [-0.0019, +0.0060] | Không hơn (khoảng tin cậy chứa 0) |
| 5. **HistGradientBoosting** | **+0.0219** | **[+0.0186, +0.0251]** | **Có hơn thật ($p < 0.05$)** |

**Kết Quả Đánh Giá Cuối Cùng trên Tập Test (88.669 khoản vay, 2015-10 đến 2015-12 - Duy Nhất 1 Lần):**
- Baseline `sub_grade`: ROC-AUC = **0.6818** (Gini = `36.36%`, KS = `26.45%`).
- **Logistic L2 ($C=1$)**: ROC-AUC = **0.7039** (Gini = `40.79%`, KS = `29.87%`) $\implies$ Cải thiện **+0.0221** so với Baseline.
- **HistGradientBoosting**: ROC-AUC = **0.7038** (Gini = `40.76%`, KS = `29.54%`) $\implies$ Cải thiện **+0.0220** so với Baseline.

*Kết luận thực tiễn:* Mô hình **Hồi quy Logistic L2 ($C=1$)** là lựa chọn tối ưu nhất để triển khai trong thực tế tín dụng: hiệu năng đứng đầu, hoàn toàn không quá khớp, huấn luyện nhanh và có hệ số rủi ro minh bạch, dễ dàng giải thích cho cơ quan thẩm định.

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
