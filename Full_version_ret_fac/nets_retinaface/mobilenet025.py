import torch.nn as nn


# bn层
def conv_bn(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    # depthwise


def conv_dw(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        # 设置group使得每个卷积核只处理一个输入的深度层
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        # 1*1卷积,把上一步产生的深度特征图像合并
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        # Python3.x 和 Python2.x 的一个区别是: Python 3 可以使用直接使用 super().xxx 代替 super(Class, self).xxx :
        # class A(object):  # Python2.x 记得继承 object
        #     def add(self, x):
        #         y = x + 1
        #         print(y)
        #
        # class B(A):
        #     def add(self, x):
        #         super(B, self).add(x)
        #
        # b = B()
        # b.add(2)  # 3
        # 所以这里是从nn.Module继承初始化
        super(MobileNetV1, self).__init__()
        # 定义stage123,属于nn.Sequential
        # 这里拆成三个stage是为了后面fpn提取特征层方便
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8 ,8卷积核的conv2d
            conv_bn(3, 8, 2, leaky=0.1),
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),  # C1

            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),
            conv_dw(32, 32, 1),  # C2

            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),
            conv_dw(64, 64, 1),  # C3:FPN
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),  # C4:FPN
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),  # C5:FPN
        )
        # 自适应平均池化 输出尺寸1,1,256
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)  # 拉平近fc层为1000输出

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    model = MobileNetV1()
    print(model)
