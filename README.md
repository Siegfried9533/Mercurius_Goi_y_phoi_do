# Mercurius_Goi_y_phoi_do
Dự án này xây dựng một hệ thống AI có khả năng "chấm điểm" và đánh giá phong cách thời trang của người dùng. Người dùng tải lên một bức ảnh, và hệ thống sẽ phân tích, so sánh nó với một "Cơ sở Dữ liệu Chuẩn mực Thời trang" (Lookbooks) đã được định nghĩa trước để đưa ra nhận xét.

# 💡 Ý tưởng Cốt lõi
Hệ thống hoạt động dựa trên nguyên lý Tìm kiếm Sự tương đồng Trực quan (Visual Similarity Search).

CLIP (Contrastive Language–Image Pre-Training): Chúng ta dùng mô hình này để "vector hóa" hình ảnh—biến mỗi bức ảnh thành một "dấu vân tay số" (ví dụ: một mảng 512 con số) đại diện cho nội dung và phong cách của nó.

FAISS (Facebook AI Similarity Search): Sau khi có hàng ngàn "dấu vân tay" từ CSDL "chuẩn", chúng ta dùng FAISS để tạo một "mục lục" (Index) siêu nhanh. Nó cho phép, từ một "dấu vân tay" của người dùng, tìm ra "dấu vân tay" giống nhất trong CSDL chuẩn chỉ trong vài mili giây.

🛠️ Công nghệ Sử dụng
Python 3.9+

FastAPI: Để xây dựng API server, nhận ảnh và trả về kết quả JSON.

sentence-transformers: Thư viện Python để dễ dàng sử dụng các mô hình CLIP.

faiss-cpu: Thư viện của Facebook AI để xây dựng Index và tìm kiếm vector siêu nhanh.

Pillow (PIL): Để xử lý và mở file ảnh.

NumPy: Để xử lý các mảng (vector).
```
📂 Cấu trúc Thư mục
/StyleSense-AI
│
├── data/
│   └── fashion_standards/
│       ├── look_001.jpg
│       ├── look_002.jpg
│       └── ... (Hàng ngàn ảnh lookbook 'chuẩn')
│
├── assets/
│   ├── fashion_standards.index   (File Index FAISS, tạo ra ở Bước 1.3)
│   ├── standards_vectors.npy     (File npy chứa vector, tạo ra ở Bước 1.2)
│   └── standards_map.json      (File map ID-tên file, tạo ra ở Bước 1.2)
│
├── src/
│   ├── 01_vectorize_standards.py (Script cho Bước 1.2)
│   ├── 02_build_faiss_index.py (Script cho Bước 1.3)
│   └── main.py                 (API server cho Giai đoạn 2)
│
├── requirements.txt
└── README.md
```
## 🏛️ Giai đoạn 1: Xây dựng "CSDL Chuẩn mực Thời trang" (Offline)
Đây là các bước chuẩn bị dữ liệu. Bạn chỉ cần chạy các script này một lần (hoặc mỗi khi cập nhật lookbook).

###Bước 1.1: Thu thập Dữ liệu "Chuẩn"
Hành động: Tự thu thập ảnh và đưa vào thư mục data/fashion_standards/.

Lý giải: Đây là "cuốn sách giáo khoa" về thời trang của bạn. Chất lượng và sự đa dạng của CSDL này sẽ quyết định 100% chất lượng đánh giá.

### Bước 1.2: Vector hóa "Chuẩn mực"
Hành động: Chạy script src/01_vectorize_standards.py.

Nhiệm vụ: Script này sẽ tải mô hình CLIP, duyệt qua từng ảnh trong data/fashion_standards/, tạo vector, và lưu ra 2 file:

assets/standards_vectors.npy (Tất cả vector)
assets/standards_map.json (Ánh xạ vị trí vector với tên file ảnh)

###Bước 1.3: Xây dựng Index Tìm kiếm (FAISS)
Hành động: Chạy script src/02_build_faiss_index.py.

Nhiệm vụ: Script này sẽ tải file .npy (ở Bước 1.2) và nạp toàn bộ vector vào một Index FAISS. Sau đó, lưu Index đã "huấn luyện" ra file: assets/fashion_standards.index.

## ⚙️ Giai đoạn 2: API Đánh giá "Đẹp/Xấu" (Online)
Đây là hệ thống API server (src/main.py) mà người dùng cuối (ví dụ: ứng dụng di động) sẽ tương tác.

###Bước 2.1: Khởi động API
Hành động: Chạy server FastAPI (ví dụ: uvicorn src.main:app --reload).

Nhiệm vụ: Khi server khởi động, nó sẽ tải 1 lần duy nhất các tài sản sau vào bộ nhớ (RAM):

Model CLIP

Index FAISS (assets/fashion_standards.index)

File Map JSON (assets/standards_map.json)

### Bước 2.2 - 2.5: Endpoint /assess-style
Endpoint: POST /assess-style

Đầu vào: Một file ảnh (multipart/form-data).

#### Quy trình xử lý:

Nhận ảnh: Lấy ảnh người dùng tải lên.

Vector hóa: Dùng chính model CLIP đã tải ở Bước 2.1 để tạo query_vector cho ảnh người dùng.

So sánh: Dùng query_vector tìm kiếm trên Index FAISS (với k=1) để tìm 1 ảnh "chuẩn" giống nhất.

Lấy kết quả: FAISS trả về D (điểm tương đồng, ví dụ: 0.95) và I (ID của ảnh giống nhất, ví dụ: 120).

Quyết định: Dựa trên điểm tương đồng và một ngưỡng (ví dụ: THRESHOLD = 0.85) để quyết định là "phù hợp" hay "chưa phù hợp".

#### 📥 Ví dụ Phản hồi (Response)
Nếu ảnh người dùng có độ tương thích là 95% (vượt ngưỡng 85%):
```
JSON

{
  "is_good_style": true,
  "compatibility_percent": 95.0,
  "message": "Đánh giá: Rất phù hợp (Tương thích 95.0%)",
  "most_similar_look": "data/fashion_standards/look_0120.jpg"
}
Nếu ảnh người dùng có độ tương thích là 72% (dưới ngưỡng 85%):

JSON

{
  "is_good_style": false,
  "compatibility_percent": 72.0,
  "message": "Kiến nghị: Chưa phù hợp (Tương thích chỉ 72.0%)",
  "most_similar_look": "data/fashion_standards/look_0451.jpg"
}
```
