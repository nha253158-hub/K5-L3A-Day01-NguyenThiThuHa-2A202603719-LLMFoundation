# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603719
- **Họ và tên:** Nguyễn Thị Thu Hà
- **SEED:** 7574

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

LLM chạy qua **Groq**: model nhỏ `openai/gpt-oss-20b`, model lớn `openai/gpt-oss-120b`. Hàng LLM chỉ đo trên SUB (40 tin) nên cột F1 test bằng F1 SUB.

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.899 | 0.889 | 0.054 | 0 |
| 2. BoW + MLP | 0.938 | 0.947 | 0.148 | 0 |
| 3a. MiniLM + kNN | 0.926 | 1.000 | 71.11 | 0 |
| 3b. MiniLM + LogReg | 0.963 | 1.000 | 30.36 | 0 |
| 3c. MiniLM + MLP head | 0.957 | 1.000 | 29.23 | 0 |
| 4a. LLM nhỏ zero-shot | 1.000* | 1.000 | 2405.93 | 0.00515 |
| 4b. LLM lớn zero-shot | 1.000* | 1.000 | 2401.45 | 0.01030 |
| 4c. LLM nhỏ few-shot | 0.974* | 0.974 | 3196.03 | 0.01425 |

\* chỉ đo trên SUB 40 tin.

- Trên đường Pareto: 1 (BoW + LogReg), 2 (BoW + MLP), 3c (MiniLM + MLP head).
- Đạt SLA p95 < 50 ms: 1, 2, 3b, 3c.

## 3. Dự đoán và thực tế

**(1) `ml_f1` — đoán F1 ≈ 0.95, thực tế 0.899.**
Precision = 1.000 nhưng recall chỉ 0.817: model không chặn nhầm tin ham nào nhưng bỏ sót gần 1/5 tin spam. Lý do lệch: BoW chỉ khớp đúng từng chữ. Các tin spam không có từ khoá quen ("You will be receiving this week's Triple Echo ringtone shortly") hoặc dùng biến thể của từ khoá ("texting" so với "text") đều bị đoán là ham. Tôi đã đánh giá quá cao khả năng của việc chỉ khớp từ khoá.

**(2) `llm_models` — đoán model lớn hơn ~5 điểm F1, đắt ~10 lần, LLM ngang hoặc kém 3b. Thực tế: hai model cùng F1 = 1.000, model lớn chỉ đắt gấp 2 lần (0.0103 so với 0.00515 USD/1k), LLM ngang 3b (đều 1.000 trên SUB).**
Lý do lệch: (a) Tôi lấy tỉ lệ giá của cặp OpenAI (terra/luna ≈ 10 lần), trong khi Groq định giá gpt-oss-120b chỉ gấp đôi gpt-oss-20b. (b) SUB chỉ có 40 tin điển hình, quá dễ, nên cả hai model đều chạm trần 1.000 và không còn chỗ để thấy khác biệt. Tôi rút ra rằng muốn so model thì phải dùng tập khó hơn (như 50 case khó), không nên dùng tập nhỏ và dễ.

**(3) `hard_eval` — đoán BoW còn 55–65% accuracy, và LLM khó nhất ở prompt injection. Thực tế: BoW chỉ còn khoảng 50%, còn LLM khó nhất ở nhóm *ham trông như spam* (70% đúng) và *nói về spam* (75%). Ở nhóm prompt injection, 4a vẫn đúng 83%.**
Lý do lệch: gpt-oss-20b ít bị câu "Ignore all previous instructions" lừa hơn tôi nghĩ. Ngược lại, nó rất nhạy với các từ như *free / prize / won*, nên tin bình thường của bạn bè hoặc tin kể lại một vụ lừa đảo lại bị gán nhầm là spam. Điểm yếu của LLM ở đây là bị bối cảnh đánh lừa chứ không phải bị injection.

## 4. Phân tích 50 case khó

| Model | F1 SUB | F1 50 case khó | Mức tụt |
| --- | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.889 | 0.333 | −0.56 |
| 2. BoW + MLP | 0.947 | 0.368 | **−0.58** |
| 3a. Emb + kNN | 1.000 | 0.741 | −0.26 |
| 3b/3c. Emb + LogReg / MLP | 1.000 | 0.653 | −0.35 |
| 4a. LLM nhỏ zero-shot | 1.000 | 0.918 | −0.08 |
| 4d. LLM nhỏ + prompt phòng thủ | 1.000 | 0.935 | −0.07 |

*(Số liệu của lần chạy 50 case gốc. Output cuối cùng trong notebook là lần chạy với 58 case, gồm 50 case gốc và 8 case tôi tự viết, nên các con số hơi khác.)*

**BoW + MLP tụt nhiều nhất** (0.947 → 0.368). Ở nhóm *spam giọng tự nhiên* BoW chỉ đúng 10%, ở nhóm *né từ khoá* chỉ 0–17%. Nhóm ham thì đúng 100%, nhưng đó là vì BoW có xu hướng đoán ham, không phải vì nó hiểu tin nhắn.

**Case 1 — BoW sai, nhóm né từ khoá:** `"C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4r..."` (spam → BoW đoán ham).
Cơ chế: `tokenize` tách ra các token `c0ngr4tulat10ns`, `y0u`, `h4ve`, `w0n`, `g1ft`, `c4r`. Không token nào nằm trong vocab 2000 từ, nên các cột tương ứng đều bằng 0 và bị bỏ qua. Vector của tin này gần như rỗng, `w·x + b ≈ b`. Vì b âm (ham chiếm đa số) nên p(spam) < 0.5.

**Case 2 — LLM sai, nhóm nói về spam:** `"lol I just got a text saying I won £1000 and to call some 0906 number, people still fall for th..."` (ham → 4d đoán spam).
Cơ chế: tin nhắn chứa gần đủ dấu hiệu của spam (*won £1000*, *call*, đầu số *0906*). LLM nhận ra mẫu spam nhưng không phân biệt được ai đang nói: người gửi đang *kể lại / chê* một tin lừa đảo, chứ không phải đang gửi tin lừa đảo. Cả 4 lỗi của 4d đều thuộc kiểu này, tức là tin ham bị đoán thành spam vì có từ khoá.

## 5. Thử thách cá nhân

**Đề #3 (SEED 7574 → 7574 % 6 = 2):** Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?

**Cách làm:** lặp qua `h ∈ {4, 512}`, gọi `build_mlp(len(vocab), hidden=h)`, train với cùng `SEED`, rồi đo F1 trên test và thời gian train.

**Kết quả:**

| Hidden | Số tham số | F1 test | Train (s) |
| ---: | ---: | ---: | ---: |
| 4 | 8,009 | [điền] | [điền] |
| 64 (Phần 2) | 128,129 | 0.938 | 2.86 |
| 512 | 1,025,025 | [điền] | [điền] |

**Giải thích:** [điền]

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (trung bình khoảng 116 tin/s) và p95 < 50 ms/tin.

**Chọn: 3c MiniLM + MLP head (hoặc 3b MiniLM + LogReg).**
- **Đạt SLA:** p95 là 29.2 ms (3c) và 30.4 ms (3b), dưới 50 ms. Kết hợp được cả độ trễ và chất lượng: F1 test 0.957–0.963, F1 SUB 1.000, nằm trên đường Pareto (3c).
- **Bền hơn BoW:** F1 trên 50 case khó là 0.653, so với 0.33–0.37 của BoW.
- **Chi phí API bằng 0:** tự host được. Throughput đo trên 1 CPU Colab khoảng 46 tin/s, nên cần khoảng 3 instance cho tải trung bình (nhiều hơn vào giờ cao điểm), hoặc chạy GPU với batch.

**Loại bỏ:**
- **LLM:** p95 khoảng 2,400 ms, gấp gần 50 lần SLA. Chi phí với giá đo được (Groq 20b, 0.00515 USD/1k) là khoảng 51.5 USD/ngày, tức khoảng 1,545 USD/tháng. Nếu dùng gpt-6-luna theo bài tính tay thì khoảng 10.5 USD/ngày cho 1 triệu tin, tức khoảng 105 USD/ngày (khoảng 3,150 USD/tháng) cho 10 triệu tin.
- **kNN:** p95 là 71 ms, vượt SLA, và càng chậm khi bộ nhớ lớn lên.
- **BoW:** nhanh nhất (0.05 ms) nhưng sụp ở case khó.

**Thiết kế lai (router):** BoW + LogReg chạy trước vì gần như miễn phí. Tin nào có p ∈ [0.3, 0.7] thì chuyển sang Embedding + head. LLM chỉ chạy **bất đồng bộ**, ngoài luồng chính, trên số ít tin vẫn còn lưỡng lự hoặc bị người dùng báo cáo, và dùng để gán nhãn dữ liệu mới cho model nhỏ.

**Đánh đổi chấp nhận:**
- Vẫn sai ở nhóm thao túng và spam giọng tự nhiên (Embedding chỉ đúng 25–40%).
- Cần dữ liệu có nhãn và phải train lại định kỳ khi spammer đổi chiêu.
- Tốn công vận hành hạ tầng encoder thay vì gọi API.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 4 | [điền prompt bạn đã dùng cho call_openai / parse_label / llm_classify_batch] | [điền] | [điền, ví dụ: thêm `max_retries=5, base_delay=4` cho retry vì Groq báo 429; kiểm tra bằng bộ test API giả lập ✅] |


**Điều tôi hiểu ra hôm nay:**
- Điểm cao trên tập test cùng phân phối với train chưa chứng minh model "hiểu" bài toán. BoW đạt F1 0.94 nhưng chỉ còn khoảng 0.35 trên 50 case khó, vì nó chỉ học thuộc từ khoá.
- Với dữ liệu lệch lớp, accuracy gây hiểu nhầm: chỉ cần luôn đoán ham là đã được 0.87. Vì vậy phải nhìn precision, recall và F1.
- Không có model nào tốt nhất ở mọi mặt. LLM chính xác và bền nhất nhưng có p95 gấp khoảng 50 lần SLA và tốn tiền theo token. Lựa chọn thực tế phải dựa trên đường Pareto và ràng buộc của sản phẩm, thường là một hệ thống lai.
