import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torchvision import models

from nets_retinaface.layers import FPN, SSH
from nets_retinaface.mobilenet025 import MobileNetV1

# 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 2，用于代表每个先验框内部包含人脸的概率
# num_anchors=2网格点上先验框的数量
class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        # 分类预测结果用于判断先验框内部是否包含物体，原版的Retinaface使用的是softmax进行判断。
        # 此时我们可以利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 2，用于代表每个先验框内部包含人脸的概率。
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        # 把深度(通道数)调整到最后一个维度上
        out = out.permute(0,2,3,1).contiguous()
        # batchsize,先验框,每个先验框有人脸的概率
        return out.view(out.shape[0], -1, 2)
# 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 4，用于代表每个先验框的调整参数
class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)
# 利用一个1x1的卷积，将SSH的通道数调整成num_anchors x 10（num_anchors x 5 x 2），用于代表每个先验框的每个人脸关键点的调整
class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=2):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    # cfg_mnet = {
    #     'name': 'mobilenet0.25',
    #     'min_sizes': [[16, 32], [64, 128], [256, 512]],
    #     'steps': [8, 16, 32],
    #     'variance': [0.1, 0.2],
    #     'clip': False,
    #     'loc_weight': 2.0,
    #     # ------------------------------------------------------------------#
    #     #   视频上看到的训练图片大小为640，为了提高大图状态下的困难样本
    #     #   的识别能力，我将训练图片进行调大
    #     # ------------------------------------------------------------------#
    #     'train_image_size': 840,
    #     'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    #     'in_channel': 32,
    #     'out_channel': 64
    # }

    def __init__(self, cfg = None, pre_train = False, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        # 这里选backbone,根据utils的config.py
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if pre_train:
                checkpoint = torch.load("./model_data/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                # 有序字典有更好的性能,但是py3.7字典也是有序了
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        #这部分不用也行,不用resnet
        elif cfg['name'] == 'Resnet50':
            backbone = models.resnet50(pretrained=pre_train)
            # 获取一个Model中你指定要获取的哪些层的输出，然后这些层的输出会在一个有序的字典中，
            # 字典中的key就是刚开始初始化这个类传进去的，value就是feature经过指定需要层的输出。
            #'return_layers'     : {'stage1': 1, 'stage2': 2, 'stage3': 3},
            # key是指定要获取Model中哪些层的输出，
            # value是这些层的输出会放在一个OrderedDict中，
            # 其中这个有序字典中的key就是前面的value，就相当于可以按照自己喜好来设置输出的key
            # 所以最后得到的是{1: C3, 2: C4, 3: C5}
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        # 'in_channel'        : 32
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        # 'out_channel'       : 64
        self.fpn = FPN(in_channels_list,out_channels)

        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        # 因为fpn层一共有三组,每组都要单独head一次,所以写_make_class_head,设置fpn_num=3,用来生成head
    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        # self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)
        # fpn的输出 = [output1, output2, output3]

        # SSH
        # self.ssh1 = SSH(out_channels, out_channels)
        # self.ssh2 = SSH(out_channels, out_channels)
        # self.ssh3 = SSH(out_channels, out_channels)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        # SSH1的shape为(80, 80, 64)；
        # SSH2的shape为(40, 40, 64)；
        # SSH3的shape为(20, 20, 64)
        # SSH1就表示将原图像划分成80x80的网格；
        # SSH2就表示将原图像划分成40x40的网格；
        # SSH3就表示将原图像划分成20x20的网格，
        # 每个网格上有两个先验框
        features = [feature1, feature2, feature3]
        # 这里分别有3组每组64个特征图,head把他们重做为三个head需要的深度,三个重做之后的图要拼接的
        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output
