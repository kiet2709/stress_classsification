
# 🧠 Stress Detection Web App

Ứng dụng web phân loại **Căng thẳng tâm lý (Stress Detection)** từ văn bản người dùng nhập vào, sử dụng mô hình **RoBERTa-base fine-tuned** trên tập dữ liệu **Dreaddit**.  
Hệ thống được triển khai theo kiến trúc **MVC** bằng **Flask**, tích hợp **LIME** để giải thích mô hình và container hóa với **Docker**.

---

## 🚀 1. Tính năng chính

- 🔍 **Phân loại văn bản:** Phát hiện “Stress” / “Not Stress” từ nội dung người dùng nhập.  
- 📊 **Hiển thị xác suất:** Thanh confidence động, hiển thị tỉ lệ phần trăm của từng lớp.  
- 🧩 **Giải thích bằng LIME:** Làm nổi bật các từ khóa ảnh hưởng nhất đến kết quả.  
- 💾 **Lưu lịch sử phân tích:** Tự động lưu 5 kết quả gần nhất trên trình duyệt (localStorage).  
- ⚙️ **Triển khai dễ dàng:** Toàn bộ dự án chạy trong Docker container chỉ với 1 lệnh.

---

## 🧱 2. Kiến trúc hệ thống

| Thành phần             | Công nghệ / Mô tả                      |
| ---------------------- | -------------------------------------- |
| **Backend**            | Python, Flask                          |
| **Deep Learning**      | PyTorch, Transformers (`roberta-base`) |
| **Giải thích mô hình** | LIME (`lime-text`)                     |
| **Frontend**           | HTML, CSS, Vanilla JavaScript          |
| **Triển khai**         | Docker (`python:3.9-slim`)             |
| **Lưu trữ cục bộ**     | `localStorage`                         |

---

## 📦 3. Cài đặt và chạy ứng dụng

### 🔹 Bước 1: Tải mô hình

1. Truy cập link Google Drive:  
   👉 [**Tải model_1.zip tại đây**](https://drive.google.com/drive/folders/1ahQf0Aoychr_I5NrzOu29vWVM-916dcW)

2. Tải file `model_1.zip` về.

3. Giải nén và **đặt thư mục kết quả vào cùng thư mục dự án**, đổi tên thành **`model`**  
   → Đảm bảo cấu trúc:  
   ./model/
├── pytorch_model.bin
├── config.json
├── tokenizer_config.json
├── vocab.json
└── merges.txt

### 🔹 Bước 2: Build Docker image

Mở terminal tại **thư mục gốc của dự án**, sau đó chạy lệnh:

```bash
docker build -t stress-detector .
🔹 Bước 3: Chạy ứng dụng
bashdocker run -p 5000:5000 stress-detector

Ứng dụng sẽ chạy tại: http://localhost:5000