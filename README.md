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
| | 5 | Gán 151 cột vào nhóm (mỗi nhóm có bằng chứng), feature engineering, loại đa cộng tuyến, biến hằng số (`term_months`) & biến trùng lặp (`grade`, `int_rate`), pipeline hoàn chỉnh |
| `03_logistic` | 6 | Huấn luyện mô hình Hồi quy Logistic; hệ thống metric tín dụng chuyên sâu (ROC-AUC, Gini, KS, PR-AUC, Brier, Log-Loss); kiểm tra quá khớp; phân tích hệ số |

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

**Chọn feature** (notebook 2): 151 cột gốc -> dữ liệu đầu vào đã chuẩn hóa 100% kỳ hạn 36 tháng từ notebook 1 -> loại biến hằng số `term_months`, đa cộng tuyến và 2 biến trùng lặp (`grade` và `int_rate`) -> **45 feature số + 5 feature phân loại**, tức **153 cột** sau mã hóa one-hot.

**Huấn luyện & Đánh giá mô hình Hồi quy Logistic** (notebook 3):
- **Phạm vi bài toán:** Chuyên biệt dự đoán rủi ro cho các khoản vay cá nhân kỳ hạn 36 tháng (đồng nhất 100% giữa train, val, test).
- **Hiệu năng trên tập Validation:**
  - ROC-AUC: `0.7040` (tương ứng Gini = `40.79%`)
  - KS Statistic: `29.94%` (chuẩn phân tách tốt trong thẩm định tín dụng, tiệm cận 30%)
  - PR-AUC: `0.2776` (gần gấp đôi tỷ lệ ngẫu nhiên 0.1474)
  - Brier Score: `0.1174` | Log-Loss: `0.3852`
- **Kiểm tra quá khớp & trôi dạt phân phối:** ROC-AUC train `0.6873` vs val `0.7040` (chênh lệch `-0.0167`; không bị quá khớp; val cao hơn train do năng lực phân tách rủi ro tăng dần theo thời gian: train 2012 là `0.6630` -> 2013 là `0.6810` -> 2014 là `0.6904` -> val 2015 đạt `0.7040`).
- **Phân loại theo ngưỡng:** Tại ngưỡng tỷ lệ nợ xấu nền `0.1370`, mô hình đạt Recall `68.4%` (bắt đúng 9.333 / 13.653 khoản vay vỡ nợ) với Accuracy `62.4%`.

## Hạn chế đã biết (đều do notebook đo được)

1. **Phạm vi mô hình chuyên biệt cho khoản vay 36 tháng** (100% đồng nhất giữa train, val, test). Khoản vay 60 tháng đã được chủ động loại bỏ ngay từ Mục 3.4 do không đủ thời gian đáo hạn ở giai đoạn 2015. Test chỉ gồm 3 tháng cuối 2015.
2. **32 cột trống hoàn toàn trong train**, trong đó 17 cột chỉ bắt đầu có dữ liệu ở giai đoạn val/test, nên không dùng được.
3. **Các cột dạng "số tháng kể từ sự kiện"** bị loại vì thiếu trên 30% trong train, dù giá trị thiếu có thể có nghĩa là "chưa từng xảy ra"; chưa kiểm tra bằng code xem việc đó có mất tín hiệu không.
4. `issue_d` (ngày giải ngân) được dùng thay cho ngày nộp đơn; chỉ có khoản vay đã được chấp nhận trong dữ liệu.
5. Cách chia này **thay cho mốc chung 2016-01-01** của bản cũ; các nhóm khác dùng cùng dataset cần đối chiếu
   `md5_id_train` trong `artifacts/split_manifest.json`.

## Ghi chú về repo

- `CLAUDE.md`: quy tắc làm việc cho các phiên Claude Code (nguyên tắc "mọi kết luận có code suy ra").
- `docs/`: tài liệu của bản cũ, có số liệu từ nguồn ngoài và các bước không còn trong notebook mới.
  Coi là lỗi thời cho đến khi được viết lại.
