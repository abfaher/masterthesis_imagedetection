import torch
import torch.nn as nn

class SimpleYOLO(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(SimpleYOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.output_dim = B * 5 + C  # (x, y, w, h, confidence) * B + classes

        self.conv = nn.Conv2d(
            in_channels=3,  # Assuming RGB input; adjust if using different modalities
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        out = self.conv(x)  # [batch_size, output_dim, S, S]
        out = out.permute(0, 2, 3, 1)  # Reshape to [batch_size, S, S, output_dim]
        return out
