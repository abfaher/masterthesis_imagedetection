import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, split_size=7, num_boxes=2, num_classes=1):
        super(SimpleYOLO, self).__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        # Simple CNN with 3 conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (448 // 8) * (448 // 8), 496),  # 448/8 = 56
            nn.LeakyReLU(0.1),
            nn.Linear(496, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x.view(-1, self.S, self.S, self.C + self.B * 5)
