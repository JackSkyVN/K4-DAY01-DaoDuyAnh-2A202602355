# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** GPU (CUDA)

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cu128 / Ultralytics 8.4.145

**Checkpoint sử dụng:** `yolo11n-cls.pt` · `yolo11n.pt` · `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không — chạy đúng thứ tự từ trên xuống dưới, không thay đổi checkpoint, threshold hay mã nguồn.

> ZIP do notebook tạo có tên `K4-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo hoặc output.

---

## 1. Phân loại ảnh – Prediction cấp ảnh

**Nguồn evidence:** `classification_predictions.json` · sample `traffic` · coco_image_id `210273` · ảnh 640×428 px

### Record hạng 1

| Trường | Giá trị |
|---|---|
| `sample_id` | `traffic` |
| `rank` | `1` |
| `class_id` | `468` |
| `class_name` | `cab` |
| `score` | `0.510915` |
| `taxonomy_name` | `ImageNet-1K` |
| `model_file` | `yolo11n-cls.pt` |

### Top-5 dự đoán cho ảnh `traffic`

| rank | class_id | class_name | score |
|---|---|---|---|
| 1 | 468 | cab | 0.5109 |
| 2 | 654 | minibus | 0.1643 |
| 3 | 734 | police_van | 0.0858 |
| 4 | 757 | recreational_vehicle | 0.0541 |
| 5 | 829 | streetcar | 0.0482 |

### Diễn giải và phân tích

**Record hạng 1 mô tả toàn ảnh như thế nào?**
Record hạng 1 cho thấy model đánh giá toàn bộ ảnh `traffic` có khả năng cao nhất thuộc lớp `cab` (taxi/xe buýt nhỏ) với score 0.51. Đây là **một nhãn duy nhất cho cả ảnh**, không phân biệt từng vật thể riêng lẻ. Ảnh giao thông thực tế có nhiều loại phương tiện (xe buýt lớn, ô tô con, người đi bộ), nhưng phân loại ảnh chỉ chọn ra *một lớp* mô tả "chủ đề tổng quát" nhất — cho thấy giới hạn rõ ràng của task phân loại ảnh khi so với detection hay segmentation.

**Ai định nghĩa class list?**
Class list (1.000 lớp) được định nghĩa bởi bộ dữ liệu **ImageNet-1K**, không phải do model tự tạo ra. Checkpoint `yolo11n-cls.pt` được huấn luyện trên taxonomy này, vì vậy nó chỉ có thể dự đoán trong phạm vi 1.000 lớp ImageNet, dù ảnh có chứa vật thể ngoài danh sách đó.

**Vì sao cần giữ cả `class_id`, `class_name` và `taxonomy_name`?**
- `class_id = 468` là mã số cố định trong taxonomy ImageNet-1K — không thay đổi dù tên hiển thị được dịch sang ngôn ngữ khác.
- `class_name = "cab"` là tên mà con người đọc được — thuận tiện nhưng có thể thay đổi theo phiên bản taxonomy hoặc bị dịch sai.
- `taxonomy_name = "ImageNet-1K"` xác định *nguồn gốc* của class list — cần thiết khi ánh xạ sang taxonomy khác (ví dụ COCO-80).
Giữ cả ba trường giúp tránh nhầm lẫn khi mapping nhãn sang hệ thống khác.

**Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?**
Guideline cần đặt ra một **quy tắc ưu tiên** rõ ràng, ví dụ: *"Chọn lớp mô tả vật thể chiếm diện tích lớn nhất hoặc nằm trung tâm ảnh; nếu nhiều vật thể cùng kích thước, chọn lớp phù hợp với mục đích bộ dữ liệu."* Không thể chỉ chép lại lớp có score cao nhất vì model không biết quy tắc ưu tiên của dự án.

**Vì sao model score không phải ground truth?**
`score = 0.510915` chỉ phản ánh mức độ tin cậy của model với dự đoán `cab` — không chứng minh rằng `cab` là nhãn đúng theo guideline gán nhãn. Ground truth do **người annotator** tạo dựa trên guideline cụ thể của dự án, có thể là `bus` hoặc `traffic_scene` tùy quy tắc. Model có thể đúng, có thể sai, và reviewer cần đối chiếu trực tiếp với ảnh.

---

## 2. Phát hiện vật thể – Lớp và box cho từng object

**Nguồn evidence:** `detection_predictions.json` · `visuals/detection_predictions.png` · sample `traffic` · ngưỡng `0.35` · tổng **53 vật thể**

### Một record được phân tích chi tiết

Tôi chọn record đầu tiên — một chiếc bus lớn nổi bật (score cao nhất = 0.91):

| Trường | Giá trị |
|---|---|
| `sample_id` | `traffic` |
| `class_id` | `5` |
| `class_name` | `bus` |
| `score` | `0.912558` |
| `bbox_xyxy` | `[93.17, 187.95, 223.01, 320.91]` |
| `bbox_width` | `129.84 px` |
| `bbox_height` | `132.96 px` |
| `score_threshold` | `0.35` |

**Diễn giải vị trí box bằng lời:**
Hộp giới hạn bắt đầu từ góc trên-trái tại tọa độ `(93, 188)` — cách cạnh trái ảnh 93 pixel và cách cạnh trên 188 pixel — và kết thúc ở góc dưới-phải `(223, 321)`. Hộp có chiều rộng ~130 px và chiều cao ~133 px, bao quanh toàn bộ thân xe buýt. Với ảnh gốc 640×428 px, hộp này nằm ở phần dưới-trái, chiếm khoảng 20% chiều ngang và 31% chiều dọc.

### Thí nghiệm ngưỡng confidence

| Ngưỡng (threshold) | Số vật thể phát hiện | Nhận xét |
|---|---|---|
| `0.25` (ngưỡng thấp) | nhiều hơn 53 | Giữ lại cả dự đoán không chắc chắn; tăng độ bao phủ nhưng tăng nhiễu |
| `0.35` (ngưỡng thực thi) | **53 objects** | Điểm cân bằng giữa bao phủ và độ tin cậy |

**Điều gì thay đổi khi thay đổi ngưỡng?**
Khi giảm threshold xuống dưới 0.35, số prediction tăng lên — model giữ thêm các trường hợp nó ít tự tin hơn, nghĩa là **độ bao phủ cao hơn** nhưng reviewer phải kiểm tra nhiều hộp hơn, trong đó có cả dự đoán sai. Khi tăng threshold lên cao hơn 0.35, số prediction giảm — chỉ giữ vật thể model tự tin cao, **giảm khối lượng review** nhưng có thể bỏ sót vật thể thật. Quan trọng: threshold chỉ là bộ lọc cho *prediction của model* — annotator vẫn phải gán nhãn mọi vật thể theo guideline, kể cả khi model bỏ sót.

### Quy tắc gán nhãn đề xuất

**Quy tắc 1 — Hộp chặt (tight bounding box):**
Hộp giới hạn phải bao sát thân vật thể: khoảng cách từ cạnh hộp đến biên nhìn thấy của vật thể không vượt quá 5 pixel ở mỗi phía. Không mở rộng hộp vào nền trống hay vật thể kề bên.

**Quy tắc 2 — Vật thể bị che khuất hoặc cắt mép:**
Nếu vật thể bị che khuất dưới 50% diện tích nhưng vẫn nhận dạng được → gán nhãn và vẽ hộp bao phần nhìn thấy. Nếu vật thể bị cắt ở mép ảnh → hộp dừng tại biên ảnh, không kéo ra ngoài. Trường hợp che khuất > 50% hoặc không đủ thông tin để xác định lớp → escalate cho lead annotator.

---

## 3. Phân đoạn theo từng đối tượng – Polygon cho mỗi instance

**Nguồn evidence:** `segmentation_predictions.json` · `visuals/segmentation_prediction.png` · sample `traffic` · ngưỡng `0.35` · tổng **49 đối tượng**

### Record được phân tích chi tiết

Tôi chọn `instance_id = "traffic-001"` — chiếc bus có score cao nhất (0.926):

| Trường | Giá trị |
|---|---|
| `instance_id` | `traffic-001` |
| `class_id` | `5` |
| `class_name` | `bus` |
| `score` | `0.925745` |
| `bbox_xyxy` | `[95.3, 188.72, 224.08, 319.96]` |
| `polygon_point_count` | `120 điểm` |
| Điểm đầu polygon (ví dụ) | `[148.0, 189.0]` → `[147.0, 190.0]` → `[145.0, 190.0]` ... |

**Mô tả polygon bằng lời:**
Đa giác gồm 120 cặp tọa độ `[x, y]` tạo thành đường viền khép kín bao quanh thân xe buýt. Các điểm bắt đầu từ khu vực mái xe phía trên (y ≈ 189), đi theo đường dốc xuống bên trái (x giảm từ 148 xuống 96), kéo thẳng dọc thân trái xe đến đáy (y ≈ 312–319), đi ngang qua phần đáy xe rồi leo dọc thân bên phải, và khép kín về điểm xuất phát. Đây là biên của chiếc bus lớn ở phần dưới-trái ảnh giao thông.

### Polygon bổ sung chi tiết gì so với box?

So sánh với detection record cùng vật thể (`bbox_xyxy = [93.17, 187.95, 223.01, 320.91]`):

| Đặc điểm | Hộp giới hạn (Detection) | Đa giác (Segmentation) |
|---|---|---|
| Hình dạng | Hình chữ nhật 4 cạnh thẳng | Đường viền bám theo hình dạng thực của xe |
| Vùng bao phủ | Bao gồm cả nền trống bên trong hộp | Chỉ bao phần thân xe, loại bỏ nền |
| Số điểm mô tả | 4 tọa độ | 120 cặp tọa độ |
| Độ chính xác biên | Thấp | Cao — theo sát biên nhìn thấy của vật thể |

**Ví dụ cụ thể:** Phần mái xe buýt có hình cong, hộp giới hạn vẫn bao trùm cả khoảng không phía trên. Polygon đi theo đường cong mái từ `(148, 189)` xuống `(96, 222)`, phản ánh chính xác hơn hình dạng thực của xe.

### `instance_id` dùng để làm gì?

`instance_id = "traffic-001"` là mã **phân biệt từng đối tượng riêng lẻ** trong output bài lab. Nó **không phải** `class_id` (class_id=5 là lớp bus), **không phải** tracking ID theo frame. Trong cùng một ảnh, nếu có hai chiếc bus thì chúng có hai `instance_id` khác nhau (e.g., `traffic-001` và `traffic-003`) nhưng cùng `class_id = 5`.

### Quy tắc gán nhãn đề xuất

**Quy tắc biên mask:**
Biên đa giác phải bám theo cạnh nhìn thấy của vật thể với sai số không quá 3 pixel. Không cắt vào thân vật thể và không bao nền quá 5 pixel liên tục tại bất kỳ đoạn nào. Với đoạn thẳng dài (e.g., thân xe buýt), cho phép giảm mật độ điểm nhưng vẫn giữ đúng hình dạng tổng thể.

**Quy tắc xử lý vùng mờ/tiếp xúc/che khuất:**
Khi hai vật thể chạm nhau (e.g., xe buýt che một phần ô tô phía sau), biên mỗi đối tượng chạy đến điểm giao nhau và dừng lại; không kéo biên vào vùng bị che. Nếu vùng tiếp xúc không xác định được biên rõ ràng (rộng > 10 px), escalate cho lead để thống nhất quy ước trước khi gán nhãn hàng loạt.

---

## 4. Vòng đời và kiểm tra chất lượng

```
ảnh thô → guideline → ground truth (annotator) → huấn luyện model → prediction → QC/rework
```

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
|---|---|---|---|---|
| Phân loại ảnh | Một nhãn lớp (class_id + class_name) cho toàn ảnh | Ảnh `traffic` chứa nhiều loại xe — model chọn `cab` nhưng có thể ưu tiên `bus` theo guideline | Gán một nhãn theo quy tắc ưu tiên chủ thể của guideline | Kiểm tra nhãn có phù hợp với chủ thể chính; đối chiếu với quy tắc ưu tiên |
| Phát hiện vật thể | Cặp (class_id, bbox_xyxy) cho mỗi vật thể | Ngưỡng 0.35 cho 53 objects — một số hộp có thể chồng lên nhau hoặc cắt mép ảnh | Vẽ hộp sát biên vật thể; gán đúng lớp; đánh dấu vật thể bị cắt/che | Kiểm tra hộp không quá rộng/hẹp; phát hiện nhãn lớp sai; xác nhận xử lý vật thể cắt mép |
| Instance segmentation | Cặp (class_id, polygon_xy) cho mỗi instance riêng lẻ | Biên polygon có thể không bám sát nếu xe bị che khuất một phần | Vẽ đa giác bám biên nhìn thấy; tạo instance riêng cho mỗi đối tượng dù cùng lớp | Kiểm tra biên không cắt vào thân vật thể; xác nhận instance_id không lặp; đánh giá nhất quán biên vùng mờ |

---

## 5. An toàn dữ liệu

**Quy tắc bảo vệ dữ liệu:**
Chỉ sử dụng ảnh công khai từ bộ COCO đã được notebook cố định và xác thực checksum. Không tải ảnh cá nhân, khuôn mặt, biển số xe, ảnh khách hàng hay bất kỳ dữ liệu nội bộ/nhạy cảm nào lên Colab hoặc repository công khai. Mọi output (JSON, PNG) chỉ chứa dữ liệu từ ảnh mẫu công khai đã kiểm tra.

**Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**
Lab Coach hoặc GV phụ trách qua kênh hỗ trợ chính thức của lớp ngay lập tức, không tiếp tục xử lý hoặc chia sẻ dữ liệu đó.

---

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json` — đã tạo và xác thực checksum PASS
- [x] `detection_predictions.json` — đã tạo và xác thực checksum PASS
- [x] `segmentation_predictions.json` — đã tạo và xác thực checksum PASS
- [x] `IMAGE_ATTRIBUTION.md` — đã tạo cùng output
- [x] `visuals/classification_top5.png` — biểu đồ top-5 ảnh `traffic`
- [x] `visuals/detection_predictions.png` — hộp phát hiện vật thể đã vẽ
- [x] `visuals/segmentation_prediction.png` — mask đa giác đã vẽ
- [x] Ô validation cuối notebook báo `PASS` cho cả 3 checkpoint và 3 sample.
- [x] Không có họ tên, MSSV hoặc dữ liệu cá nhân trong báo cáo/output.

---

## 7. Tóm tắt điểm học được

| Tác vụ | Model checkpoint | Taxonomy | Đơn vị prediction | Khác biệt với ground truth |
|---|---|---|---|---|
| Phân loại ảnh | `yolo11n-cls.pt` | ImageNet-1K (1000 lớp) | 1 lớp cho toàn ảnh | GT do annotator chọn theo guideline ưu tiên chủ thể |
| Phát hiện vật thể | `yolo11n.pt` | COCO-80 | 1 lớp + 1 box cho mỗi vật thể | GT cần box chặt, đúng lớp, xử lý che khuất |
| Instance segmentation | `yolo11n-seg.pt` | COCO-80 | 1 lớp + 1 box + polygon biên cho mỗi instance | GT cần biên bám sát, nhất quán tại vùng mơ hồ |

> **Lưu ý quan trọng:** Prediction của model (`score`, `bbox_xyxy`, `polygon_xy`) là *đầu ra tự động* phục vụ quan sát định dạng. Ground truth là nhãn do **con người** tạo theo guideline và được reviewer kiểm tra. Hai khái niệm này không thể hoán đổi cho nhau.
