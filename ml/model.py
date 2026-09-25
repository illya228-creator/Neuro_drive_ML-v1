import torch.nn as nn


class DrivingModel(nn.Module):
    """Нейросеть для управления автомобилем."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 3)
        )

    def forward(self, sensors):
        return self.network(sensors)