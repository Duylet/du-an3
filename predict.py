"""Dự đoán lớp của một điểm (x, y) bằng checkpoint đã huấn luyện."""
import argparse
from pathlib import Path

import torch
from torch import nn


class CircleClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán điểm bằng mô hình PyTorch.")
    parser.add_argument("--x", type=float, required=True, help="Tọa độ x")
    parser.add_argument("--y", type=float, required=True, help="Tọa độ y")
    parser.add_argument("--model", default="artifacts/circle_model.pt", help="Đường dẫn checkpoint")
    args = parser.parse_args()

    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise SystemExit(f"Không tìm thấy {checkpoint_path}. Hãy chạy 'python train.py' trước.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CircleClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    point = torch.tensor([[args.x, args.y]], dtype=torch.float32)
    with torch.no_grad():
        probability = torch.sigmoid(model(point)).item()
    label = int(probability >= 0.5)
    meaning = "vòng ngoài (lớp 1)" if label else "vòng trong (lớp 0)"
    print(f"Điểm: ({args.x:.3f}, {args.y:.3f})")
    print(f"Xác suất thuộc lớp 1: {probability:.1%}")
    print(f"Dự đoán: {meaning}")


if __name__ == "__main__":
    main()
