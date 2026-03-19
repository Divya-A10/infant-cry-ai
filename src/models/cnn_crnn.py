import torch
import torch.nn as nn


class CryCNNCRNN(nn.Module):

    def __init__(self, num_classes=5):
        super(CryCNNCRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gru = nn.GRU(
            input_size=32 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.cnn(x)

        b, c, h, w = x.size()

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)

        x, _ = self.gru(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x