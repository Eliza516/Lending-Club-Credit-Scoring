# CLAUDE.md - Lending Club Credit Scoring (nhánh `simple`)

Hướng dẫn cho các phiên Claude Code làm việc trong repo này.

## Dự án

Dự đoán **vỡ nợ** trên bộ dữ liệu khoản vay Lending Club (Kaggle: `wordsforthewise/lending-club`)
chỉ từ **thông tin biết được lúc xét duyệt**. Phân loại nhị phân: `1 = Charged Off`, `0 = Fully Paid`.

Đây là project giữa/cuối kỳ môn *Học thống kê* (DS66A / DS66B), do sinh viên làm và phải tự giải thích được
từng dòng code. Nhánh `simple` là bản làm lại **đơn giản**, theo từng bước. Bản cũ nằm trong `archive/`.

## Nguyên tắc số một: mọi kết luận đều có code suy ra

- **Mọi con số và nhận xét về dữ liệu phải do một ô code in ra.** Không gõ tay thống kê, tỷ lệ, AUC hay xếp hạng
  vào markdown; không dùng số liệu từ nguồn ngoài (bài báo, blog, repo khác) để kết luận về dữ liệu của mình.
  Markdown chỉ giải thích *làm gì* và *vì sao*; phần nhận xét là ô code in câu bằng f-string từ biến đã tính.
- Khi tham khảo project/paper khác: chỉ **nêu tên + link nguồn gốc + tên kỹ thuật**, không trích số, và ghi rõ
  chưa đối chiếu nội dung gốc. Sau đó phải có ô code kiểm chứng lựa chọn đó trên dữ liệu của mình.
- Nếu kết quả không như kỳ vọng thì notebook phải nói đúng như vậy, không sửa lời cho khớp.
- Mỗi hàm một ô code, mỗi kết quả (bảng, in số, biểu đồ) một ô code. Code đơn giản, dùng pandas/scikit-learn
  thuần, không cần kỹ thuật cao siêu. Một kỹ thuật cần giải thích dài mới hiểu thì bỏ.
- **Viết bằng tiếng Việt cho phần diễn giải** (markdown, chú thích, câu nhận xét in ra bằng f-string). Tên biến, tên hàm, tên cột dataframe và nhãn kỹ thuật (như `loan_count`, `num_rows`, `Yes`/`No`/`Uncertain`) giữ tiếng Anh chuẩn.
- Notebook được commit **kèm output** (output là bằng chứng). Mọi thay đổi phải chạy lại từ đầu đến cuối.

## Bố cục

```
notebooks/01_hieu_du_lieu_va_chia_tap.ipynb   Bước 1-3: đề bài, từ điển 151 cột, nạp dữ liệu gốc, EDA thời gian, chia 70/15/15
notebooks/02_lam_sach_va_feature.ipynb        Bước 4-5: làm sạch, pipeline, chọn/lọc feature theo nhóm, feature engineering
notebooks/03_logistic.ipynb                  Bước 6: Huấn luyện & So sánh 5 Mô hình (Baseline sub_grade, Logistic L2, Decision Tree, Random Forest, HistGradientBoosting), Kiểm tra quá khớp, Bootstrap CI 95%, Đánh giá tập Test duy nhất 1 lần
archive/                                      Các notebook bản cũ (01-04), giữ để tham khảo
docs/                                         Tài liệu của bản cũ (xem lưu ý cuối file)

data/raw/accepted_2007_to_2018Q4.csv.gz       File Kaggle (git-ignored, bắt buộc)
data/processed/{train,val,test}.parquet       Notebook 1 -> notebook 2 (đủ 151 cột gốc + vài cột phụ), data_dictionary.csv
artifacts/                                    split_manifest.json, preprocess_pipeline.joblib, feature_info.json,
                                              column_groups.csv, figures/  (git-ignored)
```

Các bước sau (mô hình hóa, đánh giá...) sẽ được thêm vào sau khi bước 1-5 hoàn tất.

Không tạo `src/` hay file `.py` riêng: code nằm trong notebook, hàm định nghĩa ngay gần chỗ dùng đầu tiên.
Không có dữ liệu giả lập: file thật là bắt buộc để mọi con số in ra là kết quả thật.

## Cách chia dữ liệu (kiểu Arturo-GA, theo thời gian)

1. Chỉ giữ khoản vay **có nhãn** (`Fully Paid` / `Charged Off`), **phát hành từ 2012** (mốc 2012 được chọn vì 44 cột lịch sử tín dụng bureau bắt đầu có dữ liệu từ năm này và giai đoạn trước đó chỉ chiếm dưới 3% số khoản vay có nhãn), **đã đáo hạn** (tháng giải ngân + kỳ hạn không muộn hơn tháng cuối của dữ liệu), và **kỳ hạn 36 tháng** (loại bỏ kỳ hạn 60 tháng vì chưa đủ thời gian đáo hạn ở giai đoạn 2015).
2. Sắp theo `issue_d` (rồi `id`), cắt hai **ngày** tại vị trí 70% và 85%: train < val < test.
3. `assert` thứ tự thời gian giữ trong code. Manifest `split_manifest.json` có số dòng, ngày cắt, MD5 id train.

Hệ quả đã được notebook 1 đo bằng code: 100% các khoản vay ở cả train, val và test đều là 36 tháng (đồng nhất về kỳ hạn). Biến `term` trở thành hằng số nên được loại bỏ ở Bước 5 cùng với `grade` và `int_rate`. Vẫn tồn tại sự thay đổi tỷ lệ nợ xấu theo thời gian (train 13,7% vs val 14,7% vs test 14,8%).

**Cách chia này thay cho mốc chung 2016-01-01 của bản cũ.** Nếu các nhóm khác cùng dùng dataset này, phải báo
và đối chiếu `md5_id_train` trong manifest.

## Quy tắc chống rò rỉ

- Chỉ dùng cột biết được lúc nộp đơn. Từ điển (notebook 1, mục 1.3) ghi *giả thuyết* cho từng cột; notebook 2
  mục 5.3 kiểm chứng bằng code (ROC-AUC từng cột đứng một mình trên train, so với cột lúc nộp đơn tốt nhất).
  Cột "không rõ thời điểm" bị loại cho an toàn. Cột mới phải được thêm vào từ điển kèm lý do trước khi dùng.
- **Chia trước, biến đổi sau.** Mọi thứ *học một con số từ dữ liệu* (trung vị, trung bình, độ lệch chuẩn, danh
  sách giá trị one-hot, mọi ngưỡng như `REVOL_UTIL_CAP`, `LOAN_TO_INCOME_CAP`, `MAX_MISSING`) chỉ tính trên `train`.
- Quy tắc làm sạch và feature engineering là công thức cố định theo từng dòng. Phần học nằm trong `Pipeline`
  (`ColumnTransformer`), chỉ `fit` trên train; val/test chỉ `transform`. Có ô chứng minh điều đó trong notebook 2.
- EDA/chọn feature chỉ dùng train. Ngoại lệ có ghi rõ: EDA thời gian ở notebook 1 (bước 3) dùng toàn bộ khoản
  vay có nhãn nhưng chỉ hỏi về cấu trúc thời gian, không hỏi quan hệ feature-nhãn.
- Khi làm bước mô hình: validation theo thời gian (`TimeSeriesSplit`), không dùng K-fold xáo trộn.

## Quy ước

- Python >= 3.10, `SEED = 42`, lưu hình bằng `plt.savefig(..., dpi=150, bbox_inches="tight")` vào `artifacts/figures/`.
- Không commit `data/` và `artifacts/` (git-ignored).
- `docs/*` mô tả bản cũ (có số liệu từ nguồn ngoài và các bước không còn tồn tại như SHAP/PSI). Không trích làm
  kết quả của notebook hiện tại; cần viết lại hoặc xóa khi người dùng quyết định.
