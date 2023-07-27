import torch

from .layers import *
import torch.nn.functional as F
import torch.nn as nn


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


class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 2

    def __init__(self, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, self.expansion * 8, kernel_size=1, padding=0, stride=stride)
        self.bn3 = nn.BatchNorm2d(self.expansion * 8)
        if stride != 1 or 3 != self.expansion * 8:
            self.shortcut = nn.Sequential(
                nn.Conv2d(1, self.expansion * 8, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * 8)
            )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.maxpool(out)
        return out


class Bottleneck1(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 2

    def __init__(self, stride=1):
        super(Bottleneck1, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, self.expansion * 32, kernel_size=1, padding=0, stride=stride)
        self.bn3 = nn.BatchNorm2d(self.expansion * 32)
        if stride != 1 or 16 != self.expansion * 32:
            self.shortcut = nn.Sequential(
                nn.Conv2d(16, self.expansion * 32, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * 32)
            )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.maxpool(out)
        return out


class Bottleneck2(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 2

    def __init__(self, stride=1):
        super(Bottleneck2, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, self.expansion * 64, kernel_size=1, padding=0, stride=stride)
        self.bn3 = nn.BatchNorm2d(self.expansion * 64)
        if stride != 1 or 64 != self.expansion * 64:
            self.shortcut = nn.Sequential(
                nn.Conv2d(64, self.expansion * 64, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion * 64)
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
        self.main1 = nn.Sequential(
            # BasicConv(32, out_plane // 4, kernel_size=3, stride=1, relu=True),
            # BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(32, out_plane // 2, kernel_size=3, stride=1, relu=True),

        )
        self.fm = FM(out_plane // 2)
        self.conv1 = BasicConv(out_plane // 2, out_plane - 3, kernel_size=1, stride=1, relu=True)
        self.conv2 = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x, y):
        x1 = self.main(x)
        y = self.main1(y)
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


class Features_extraction(nn.Module):
    def __init__(self, num_res=8):
        super(Features_extraction, self).__init__()
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

    def forward(self, x, y):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        # y_2 = F.interpolate(y, scale_factor=0.5)
        # y_4 = F.interpolate(y_2, scale_factor=0.5)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res11 = F.interpolate(res1, scale_factor=0.25)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z4 = self.SCM1(x_4, res11)
        z_c = self.FM1(z4, z)
        z_d = self.FAM1(z, z4)
        z = z_c + z_d
        z = self.Encoder[2](z)

        return z, x_, x_2, x_4, res1, res2


class texture_feat_extract(nn.Module):
    def __init__(self):
        super(texture_feat_extract, self).__init__()

    def forward(self):
        return 1


class TWRO_Net(nn.Module):
    def __init__(self, num_res=8):
        super(TWRO_Net, self).__init__()

        base_channel = 32
        self.Feature_extraction = nn.ModuleList([
            Features_extraction(num_res),
            Features_extraction(num_res),
        ])
        self.bottleneck = nn.ModuleList([
            Bottleneck(),
            Bottleneck1(),
            Bottleneck2()
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(128, 64, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(64, 32, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(32, 16, kernel_size=3, relu=False, stride=1),
            BasicConv(16, 3, kernel_size=3, relu=False, stride=1),

        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 4, base_channel * 4, kernel_size=1, relu=True, stride=1),
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

    def forward(self, x):

        x_z, x_, x_2, x_4, x_res1, x_res2 = self.Feature_extraction[0](x, x)
        # y_z, y_, y_2, y_4, y_res1, y_res2 = self.Feature_extraction[1](y, y)
        # w = self.bottleneck[0](w)
        # w = self.bottleneck[1](w)
        # w = self.bottleneck[2](w)
        # double = torch.cat((x_z, y_z, x_4, y_4), dim=1)
        # double = torch.cat((x_z, y_z), dim=1)
        double = self.Convs[3](x_z)
        # double1 = double * w
        # double1 = self.Convs[2](double1)
        # double = x_z + double1
        double = self.Decoder[0](double)
        double = self.feat_extract[3](double)
        double = torch.cat((double, x_res2), dim=1)

        double = self.Convs[0](double)
        double = self.Decoder[1](double)
        double = self.feat_extract[4](double)
        double = torch.cat((double, x_res1), dim=1)

        double = self.Convs[1](double)
        double = self.Decoder[2](double)
        double = self.feat_extract[5](double)

        double = self.feat_extract[6](double)
        double = double + x
        return double
