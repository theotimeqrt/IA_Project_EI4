import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


# 没有激活函数的bn块
def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

# SSH:分别对fpn产生的三个特征层进行ssh,得到三个ssh后的深度为64的特征图用于最后的三项任务
# 对于输入ssh的图像,三种不同深度的特征层最后在深度方向拼接
class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
            # 这out_channel == 64
        # conv3X3 filters==32
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        # 写作5x5实际也是3*3  filters==16
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        # 写作7*7实际也是3*3 filters==16
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, inputs):
        conv3X3 = self.conv3X3(inputs)

        conv5X5_1 = self.conv5X5_1(inputs)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        # 堆叠,deep == 32+16+16 == 64
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

# fpn块调用需要指定输入通道列表(3个有效特征层)然后指定输出通道数(64)作为初始化
# 先把三个特征层的深度对齐为64,再上采样把图像长宽对齐
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        # 这里3个output输出     out_channels == 64
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, inputs):
        # names = list(inputs.keys())
        # thislist = list(("apple", "banana", "cherry"))
        # print(thislist) list函数创建列表
        # input = {1: C3, 2: C4, 3: C5}
        inputs = list(inputs.values())
        # 这时inputs是一个含有三个有效特征层输出的列表
        # 进行通道数调整: out_channel = 64
        output1 = self.output1(inputs[0])
        output2 = self.output2(inputs[1])
        output3 = self.output3(inputs[2])
        # output3是第一个输出20,20,64
        # 进行上采样：
        # 先把C5的特征层20,20,64 => 40,40,64
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)
        # output2是第二个输出40,40,64
        #叠加后继续上采样 40,40,64 => 80,80,64
        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        # 三个输出后面分别ssh
        out = [output1, output2, output3]
        return out
