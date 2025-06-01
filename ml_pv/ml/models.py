import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_90(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [2]).transpose(2, 3)


def rotate_180(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [2, 3])


def rotate_270(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [3]).transpose(2, 3)


def x_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [3])


def y_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, [2])


class ParityNet(nn.Module):
    """Convolutional network with parity‐averaging."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # average the four parity‐transformed passes
        # Note here, extra rotations could be added,
        # But datagen automatically randomly rotates images
        out = (
            self._forward_once(x)
            + self._forward_once(rotate_180(x))
            - self._forward_once(x_flip(x))
            - self._forward_once(y_flip(x))
        ) / 4.0
        return out

    def _forward_once(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape (batch, 1, 1)? or (batch, 1)
        return x.sum(dim=1)  # make (batch,)
