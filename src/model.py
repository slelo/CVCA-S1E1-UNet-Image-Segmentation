import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Convolutional layers where self.net is a sequential container of layers
        self.net = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    # Forward pass
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolutional layers
        self.conv = DoubleConv(in_channels, out_channels)
        # Max Pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # x is the input feature map
        x = self.conv(x)
        # p is the downsampled feature map
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Upsample by transposed convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # Convolutional layers
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # skip is the feature map from the encoder. The method torch.cat concatenates the feature maps.
        # This concatenation is done along the channel dimension, and needed because the decoder
        # needs the higher resolution feature maps from the encoder to generate the output.
        x = torch.cat([x, skip], dim=1)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = EncoderBlock(3, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.dec1 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    # Forward pass. This method does the following:
    # 1. Pass the input through the encoder blocks
    # 2. Pass the output of the last encoder block through the bottleneck
    # 3. Pass the output of the bottleneck through the decoder blocks
    # 4. Pass the output of the last decoder block through the output layer
    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)

        b = self.bottleneck(p3)

        # d2 and d3 take x as parameter, because:
        d1 = self.dec1(b, s3)
        d2 = self.dec2(x, s2)
        d3 = self.dec3(x, s1)

        return self.out()
