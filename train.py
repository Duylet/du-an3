"""Huấn luyện MLP phân loại dữ liệu hai vòng tròn bằng PyTorch."""
from pathlib import Path
import random

import torch
from torch import nn

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


class CircleClassifier(nn.Module):
    """Mạng MLP nhận tọa độ (x, y), trả về một logit."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_dataset(samples: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Tạo dữ liệu 2D: lớp 0 ở vòng trong, lớp 1 ở vòng ngoài."""
    points = torch.randn(samples, 2)
    points = points / points.norm(dim=1, keepdim=True).clamp_min(1e-8)
    radii = torch.empty(samples, 1).uniform_(0.0, 1.5)
    points = points * radii
    labels = (radii[:, 0] > 0.72).float().unsqueeze(1)
    points += 0.04 * torch.randn_like(points)
    return points, labels


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")
    x, y = make_dataset()
    split = int(0.8 * len(x))
    x_train, y_train = x[:split].to(device), y[:split].to(device)
    x_test, y_test = x[split:].to(device), y[split:].to(device)

    model = CircleClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs, batch_size = 150, 64

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        for indices in order.split(batch_size):
            loss = criterion(model(x_train[indices]), y_train[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 15 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(x_test)
                test_loss = criterion(logits, y_test).item()
                accuracy = ((logits >= 0).float() == y_test).float().mean().item()
            print(f"Epoch {epoch:03d}/{epochs} | test_loss={test_loss:.4f} | accuracy={accuracy:.1%}")

    output = Path("artifacts/circle_model.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "description": "MLP classifier trained on synthetic concentric-circle data",
    }, output)
    print(f"\nĐã lưu mô hình: {output.resolve()}")


if __name__ == "__main__":
    main()
