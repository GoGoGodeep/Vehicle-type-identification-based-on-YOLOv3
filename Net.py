import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义卷积层
class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionLayer, self).__init__()

        self.sub_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)

# 定义残差结构，残差必须满足输出特征图大小和通道数都相同
class ResidualLayer(nn.Module):
    def __init__(self, channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = nn.Sequential(
            # stride=1，padding=0 遍历采样
            # 64——>32   见具体的网络图
            ConvolutionLayer(channels, channels // 2, kernel_size=1, stride=1, padding=0),
            # 32——>64
            ConvolutionLayer(channels // 2, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.sub_module(x) + x


# 定义上采样模块，上采样方式采用临近插值法
class UpsampleLayer(nn.Module):
    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        # 上采样倍数为2倍
        return F.interpolate(x, scale_factor=2, mode="nearest")

# 定义下采样层，我们使用步长为2的卷积来实现下采样
# 大小减半（3*3 ——> 3*3/2），通道翻倍（128——>256）
class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleLayer, self).__init__()

        self.sub_module = nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.sub_module(x)

# 定义卷积块
class ConvolutionalSet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            ConvolutionLayer(out_channels, in_channels, kernel_size=3, stride=1, padding=1),

            ConvolutionLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)

# 定义主网络，主网络采用darknet53，对应网络写
class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()

# —————————————————————————————darknet53——————————————————————————————————————
        self.trunk_52 = nn.Sequential(
            ConvolutionLayer(3, 32, 3, 1, 1),
            DownsampleLayer(32, 64),
            # 1次残差
            ResidualLayer(64),

            DownsampleLayer(64, 128),
            # 2次残差
            ResidualLayer(128),
            ResidualLayer(128),

            DownsampleLayer(128, 256),
            # 8次残差
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )   # 输出52*52

        self.trunk_26 = nn.Sequential(
            DownsampleLayer(256, 512),
            # 8次残差
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )   # 输出26*26

        self.trunk_13 = nn.Sequential(
            DownsampleLayer(512, 1024),
            # 4次残差
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
        )   # 输出13*13

# —————————————————————————————Predict One——————————————————————————————————————
        self.conv_set_13 = nn.Sequential(
            ConvolutionalSet(1024, 512),
        )

        self.detection_13 = nn.Sequential(
            ConvolutionLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 30, 1, 1, 0)
        )   # 输出13*13               # 3个种类 1个置信度 4个坐标 5个分类
            # out_channels = 30 = 3 * 10 = 3 * (1 + 4 + 5)，根据需求改变

        self.up_26 = nn.Sequential(
            ConvolutionLayer(512, 256, 1, 1, 0),
            UpsampleLayer()     # 256 ——> 512
        )
# —————————————————————————————Predict Two———————————————————————————————————————
        self.conv_set_26 = nn.Sequential(
            ConvolutionalSet(512+256, 256)  # 进行了拼接操作
        )

        self.detection_26 = nn.Sequential(
            ConvolutionLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 30, 1, 1, 0)
        )

        self.up_52 = nn.Sequential(
            ConvolutionLayer(256, 128, 1, 1, 0),
            UpsampleLayer()     # 128 ——> 256
        )
# —————————————————————————————Predict Three——————————————————————————————————————
        self.conv_set_52 = nn.Sequential(
            ConvolutionalSet(256+128, 128)
        )

        self.detection_52 = nn.Sequential(
            ConvolutionLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 30, 1, 1, 0)
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)

        conv_set_out_13 = self.conv_set_13(h_13)
        detection_out_13 = self.detection_13(conv_set_out_13)
        up_out_26 = self.up_26(conv_set_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)  # 数据拼接，dim=1表示在通道上进行拼接

        conv_set_out_26 = self.conv_set_26(route_out_26)
        detection_out_26 = self.detection_26(conv_set_out_26)
        up_out_52 = self.up_52(conv_set_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)

        conv_set_out_52 = self.conv_set_52(route_out_52)
        detection_out_52 = self.detection_52(conv_set_out_52)

        # 返回三种特征图大小的输出结果
        return detection_out_13, detection_out_26, detection_out_52

if __name__ == '__main__':
    net = MainNet()
    x = torch.randn([1, 3, 416, 416], dtype=torch.float32)
    # 测试网络
    y_13, y_26, y_52 = net(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)

    x = torch.randn([2, 3, 608, 608], dtype=torch.float32)
    y_19, y_38, y_76 = net(x)
    print(y_19.shape)
    print(y_38.shape)
    print(y_76.shape)
