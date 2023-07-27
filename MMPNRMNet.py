from .layers import *
from models import TRO_Net


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class BasicConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        self.conv(x)
        return x


import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 2

    def __init__(self, in_feature, out_feature, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_feature)
        self.conv2 = nn.Conv2d(out_feature, out_feature, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_feature)
        self.conv3 = nn.Conv2d(out_feature, self.expansion * out_feature, kernel_size=1, padding=0, stride=stride)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_feature)
        if stride != 1 or in_feature != self.expansion * out_feature:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_feature, self.expansion * out_feature, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * out_feature)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class FM(nn.Module):
    def __init__(self, channel):
        super(FM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):  # z,z2
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class SCM(nn.Module):
    # out_plane  =32
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),

        )
        self.fm = FM(out_plane // 2)
        self.conv1 = BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x, y):
        x1 = self.main(x)
        x1 = self.fm(y, x1)
        x1 = self.conv1(x1)
        x = torch.cat([x, x1], dim=1)
        return self.conv2(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):  # z,z2
        x = x1 * x2
        out = x1 + self.merge(x)
        return out


class Feature_extraction(nn.Module):
    def __init__(self, num_res=8):
        super(Feature_extraction, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel * 2, num_res),
            EBlock(base_channel * 4, num_res),
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        self.FAM1 = FAM(base_channel * 4)
        self.FM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.FM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res11 = F.interpolate(res1, scale_factor=0.5)

        z = self.feat_extract[1](res1)
        z2 = self.SCM2(x_2, res11)
        z_a = self.FM2(z2, z)
        z_b = self.FAM2(z, z2)
        z = z_a + z_b
        res2 = self.Encoder[1](z)
        res22 = F.interpolate(res2, scale_factor=0.5)
        z = self.feat_extract[2](res2)
        z4 = self.SCM1(x_4, res22)
        z_c = self.FM1(z4, z)
        z_d = self.FAM1(z, z4)
        z = z_c + z_d
        z = self.Encoder[2](z)

        return z, x_, x_2, x_4, res1, res2


class MMPNRMNet(nn.Module):
    def __init__(self, num_res=8):
        super(MMPNRMNet, self).__init__()

        base_channel = 32
        self.Feature_extraction = nn.ModuleList([
            Feature_extraction(num_res),
            Feature_extraction(num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(262, 128, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel * 2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(16, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(262, num_res),
            DBlock(64, num_res),
            DBlock(16, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(134, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(38, 16, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel * 1),
            AFF(base_channel * 7, base_channel * 2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x, y):
        x_z, x_, x_2, x_4, x_res1, x_res2 = self.Feature_extraction[0](x)
        y_z, y_, y_2, y_4, y_res1, y_res2 = self.Feature_extraction[1](y)
        double = torch.cat((x_z, y_z, x_4, y_4), dim=1)

        double = self.Decoder[0](double)
        double = self.feat_extract[3](double)
        double = torch.cat((y_2, double, x_2), dim=1)

        double = self.Convs[0](double)
        double = self.Decoder[1](double)
        double = self.feat_extract[4](double)
        double = torch.cat((y, double, x), dim=1)

        double = self.Convs[1](double)
        double = self.Decoder[2](double)
        double = self.feat_extract[5](double)

        return double


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MMPNRMNet":
        return MMPNRMNet()
    if model_name == "TRO_Net":
        return TRO_Net.TRO_Net()
