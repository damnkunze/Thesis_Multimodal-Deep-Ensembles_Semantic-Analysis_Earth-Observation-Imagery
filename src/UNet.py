from src.base import *

# ## Model

# *U Net In PyTorch*
# https://medium.com/@fernandopalominocobo/mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # default: stride=1
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Was genau macht transpose in ConvTranspose2d() ?
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)


# big model: 5 Layers
class UNet_big_5layers(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.down_convolution_5 = DownSample(512, 1024)

        self.bottleneck = DoubleConv(1024, 2048)

        self.up_convolution_0 = UpSample(2048, 1024)
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        down_5, p5 = self.down_convolution_5(p4)

        b = self.bottleneck(p5)

        up_0 = self.up_convolution_0(b, down_5)
        up_1 = self.up_convolution_1(up_0, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


# normal model: 4 Layers
#  just like in original UNet paper
class UNet_standard_4layers(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottleneck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


# -> DEFAULT UNet
UNet = UNet_standard_4layers


class UNet_small_3layers(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up_convolution_1 = UpSample(512, 256)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)

        b = self.bottleneck(p3)

        up_1 = self.up_convolution_1(b, down_3)
        up_2 = self.up_convolution_2(up_1, down_2)
        up_3 = self.up_convolution_3(up_2, down_1)

        out = self.out(up_3)
        return out


# Huge model: 6 Layers
class UNet_huge_6layers(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.down_convolution_5 = DownSample(512, 1024)
        self.down_convolution_6 = DownSample(1024, 2048)

        self.bottleneck = DoubleConv(2048, 4096)

        self.up_convolution_1 = UpSample(4096, 2048)
        self.up_convolution_2 = UpSample(2048, 1024)
        self.up_convolution_3 = UpSample(1024, 512)
        self.up_convolution_4 = UpSample(512, 256)
        self.up_convolution_5 = UpSample(256, 128)
        self.up_convolution_6 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        down_5, p5 = self.down_convolution_5(p4)
        down_6, p6 = self.down_convolution_6(p5)

        b = self.bottleneck(p6)

        up_1 = self.up_convolution_1(b, down_6)
        up_2 = self.up_convolution_2(up_1, down_5)
        up_3 = self.up_convolution_3(up_2, down_4)
        up_4 = self.up_convolution_4(up_3, down_3)
        up_5 = self.up_convolution_5(up_4, down_2)
        up_6 = self.up_convolution_6(up_5, down_1)

        out = self.out(up_6)
        return out

# bad!! scale Up model
''' class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, scaleUpBy=1):
        super().__init__()
        # Increase model size if wanted
        # -> Inserted layer has same param count as * 2 existing layers
        k = scaleUpBy

        self.down_convolution_1 = DownSample(in_channels, 64 * k)
        self.down_convolution_2 = DownSample(64 * k, 128 * k)
        self.down_convolution_3 = DownSample(128 * k, 256 * k)
        self.down_convolution_4 = DownSample(256 * k, 512 * k)

        self.bottleneck = DoubleConv(512 * k, 1024 * k)

        self.up_convolution_1 = UpSample(1024 * k, 512 * k)
        self.up_convolution_2 = UpSample(512 * k, 256 * k)
        self.up_convolution_3 = UpSample(256 * k, 128 * k)
        self.up_convolution_4 = UpSample(128 * k, 64 * k)

        self.out = nn.Conv2d(in_channels=64 * k, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottleneck(p4)
        up_1 = self.up_convolution_1(b, down_4)

        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out '''

# --> Model params:
# normal model                     31.033.045
# Add layer 5 at bottleneck        124.363.477 -> params * 4
# Add layer 6 at bottleneck
# -- tried, bad:
# Add layer before & after         31.095.157  -> meh
# Add layer between p2, p3         32.822.293  -> meh
# scale all layer in- and out * 2: 124.112.277 -> params * 4
# scale all layer in- and out * 3: 279.237.717 -> params * 8
