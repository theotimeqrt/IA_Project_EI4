from itertools import product as product
from math import ceil

import torch
# 生成先验框,每个点位生成两个框

class Anchors(object):
    def __init__(self, cfg, image_size=None):
        super(Anchors, self).__init__()
        # 先验框尺寸
        self.min_sizes  = cfg['min_sizes']
        # 原始图像压缩次数
        # 'steps'             : [8, 16, 32],
        self.steps      = cfg['steps']
        # 是否缩放到0-1之间
        self.clip       = cfg['clip']

        #   输入进来的图片的尺寸
        self.image_size = image_size

        #   三个有效特征层高和宽
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            # 取出每一个特征层的先验框
            min_sizes = self.min_sizes[k]
            #-----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            #-----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    # 先验框映射到网格点
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
