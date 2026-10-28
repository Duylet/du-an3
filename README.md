# PyTorch Demo — Phân loại dữ liệu bằng mạng nơ-ron

Dự án demo nhỏ sử dụng **PyTorch** để huấn luyện mạng nơ-ron phân loại hai vòng tròn dữ liệu (inside/outside). Mục tiêu là minh họa Tensor, `torch.nn`, hàm mất mát, `autograd`, optimizer và lưu/tải mô hình.

> Đây là dự án học tập độc lập lấy cảm hứng từ thư viện PyTorch, không phải bản sao mã nguồn PyTorch.

## Tính năng
- Tạo dữ liệu mẫu bằng PyTorch, không cần tải dataset.
- Huấn luyện MLP với `nn.Module`, `BCEWithLogitsLoss` và `Adam`.
- Tự chọn CUDA nếu có, nếu không dùng CPU.
- Lưu mô hình vào `artifacts/circle_model.pt`.
- Dự đoán lớp cho một điểm `x, y` từ dòng lệnh.

## 1. Yêu cầu
- Python 3.10 trở lên.
- PyTorch. Máy không có GPU vẫn chạy được bằng CPU.

## 2. Cài đặt trên Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu kích hoạt môi trường bị chặn, có thể chạy trực tiếp bằng `.venv\Scripts\python.exe`.

## 3. Huấn luyện
```powershell
python train.py
```
Chương trình in loss và accuracy, sau đó lưu checkpoint vào thư mục `artifacts`.

## 4. Dự đoán
Sau khi huấn luyện xong:
```powershell
python predict.py --x 0.1 --y 0.2
python predict.py --x 1.2 --y 0.8
```
Do mô hình học từ dữ liệu ngẫu nhiên, kết quả có thể hơi khác giữa các lần huấn luyện.

## Cấu trúc
```text
du-an3/
├── train.py          # Tạo dữ liệu và huấn luyện
├── predict.py        # Tải checkpoint và dự đoán
├── requirements.txt  # Thư viện cần cài
├── .gitignore
└── artifacts/        # Checkpoint tạo khi chạy (không commit)
```

## Kiến thức PyTorch được minh họa
1. **Tensor:** biểu diễn dữ liệu đầu vào và nhãn.
2. **`nn.Module`:** định nghĩa mạng MLP.
3. **Forward pass:** tính logits từ dữ liệu đầu vào.
4. **Loss:** đo sai lệch giữa dự đoán và nhãn.
5. **Autograd + optimizer:** tính gradient và cập nhật trọng số.
6. **`state_dict`:** lưu và tải trọng số mô hình.

Tham khảo: [PyTorch](https://github.com/pytorch/pytorch) · [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html).
