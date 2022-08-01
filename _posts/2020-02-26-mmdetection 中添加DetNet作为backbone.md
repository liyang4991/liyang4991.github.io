---
layout:     post
title:      mmdetection 踩坑日记2
subtitle:   mmdetection 添加backbone
date:       2020-02-26
author:     liyang
header-img: img/bg_007.jpg
catalog: true
tags:
    - mmdetection
    - 目标检测
---

# mmdetection 中添加DetNet作为backbone

1. 在mmdet/model/backbones文件夹下新建detnet.py， 添加以下代码：

   ~~~python
   from __future__ import absolute_import
   from __future__ import division
   from __future__ import print_function
   
   from ..registry import BACKBONES
   import torch
   import torch.nn as nn
   import math
   from mmcv.runner import load_checkpoint
   import logging
   
   
   
   __all__ = ['DetNet',]
   
   
   def conv3x3(in_planes, out_planes, stride=1):
       "3x3 convolution with padding"
       return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)
   
   
   class BasicBlock(nn.Module):
       expansion = 1
   
       def __init__(self, inplanes, planes, stride=1, downsample=None):
           super(BasicBlock, self).__init__()
           self.conv1 = conv3x3(inplanes, planes, stride)
           self.bn1 = nn.BatchNorm2d(planes)
           self.relu = nn.ReLU(inplace=True)
           self.conv2 = conv3x3(planes, planes)
           self.bn2 = nn.BatchNorm2d(planes)
           self.downsample = downsample
           self.stride = stride
   
       def forward(self, x):
           residual = x
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
   
           if self.downsample is not None:
               residual = self.downsample(x)
   
           out += residual
           out = self.relu(out)
   
           return out
   
   
   class Bottleneck(nn.Module):
       expansion = 4
   
       def __init__(self, inplanes, planes, stride=1, downsample=None):
           super(Bottleneck, self).__init__()
           self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
           self.bn1 = nn.BatchNorm2d(planes)
           self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                  padding=1, bias=False)
           self.bn2 = nn.BatchNorm2d(planes)
           self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
           self.bn3 = nn.BatchNorm2d(planes * 4)
           self.relu = nn.ReLU(inplace=True)
           self.downsample = downsample
           self.stride = stride
   
       def forward(self, x):
           residual = x
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)
   
           out = self.conv3(out)
           out = self.bn3(out)
   
           if self.downsample is not None:
               residual = self.downsample(x)
   
           out += residual
           out = self.relu(out)
   
           return out
   
   
   class BottleneckA(nn.Module):
       expansion = 4
   
       def __init__(self, inplanes, planes, stride=1, downsample=None):
           super(BottleneckA, self).__init__()
           assert inplanes == (planes * 4), 'inplanes != planes * 4'
           assert stride == 1, 'stride != 1'
           assert downsample is None, 'downsample is not None'
           self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
           self.bn1 = nn.BatchNorm2d(planes)
           self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                                  padding=2, bias=False)  # stride = 1, dilation = 2
           self.bn2 = nn.BatchNorm2d(planes)
           self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
           self.bn3 = nn.BatchNorm2d(planes * 4)
           self.relu = nn.ReLU(inplace=True)
           self.downsample = downsample
           self.stride = stride
   
       def forward(self, x):
           residual = x
   
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)
   
           out = self.conv3(out)
           out = self.bn3(out)
   
           if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
               residual = self.downsample(x)
   
           out += residual
           out = self.relu(out)
   
           return out
   
   
   class BottleneckB(nn.Module):
       expansion = 4
   
       def __init__(self, inplanes, planes, stride=1, downsample=None):
           super(BottleneckB, self).__init__()
           assert inplanes == (planes * 4), 'inplanes != planes * 4'
           assert stride == 1, 'stride != 1'
           assert downsample is None, 'downsample is not None'
           self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # inplanes = 1024, planes = 256
           self.bn1 = nn.BatchNorm2d(planes)
           self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, dilation=2,
                                  padding=2, bias=False)  # stride = 1, dilation = 2
           self.bn2 = nn.BatchNorm2d(planes)
           self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
           self.bn3 = nn.BatchNorm2d(planes * 4)
           self.relu = nn.ReLU(inplace=True)
           self.downsample = downsample
           self.stride = stride
           self.extra_conv = nn.Sequential(
               nn.Conv2d(inplanes, planes * 4, kernel_size=1, bias=False),
               nn.BatchNorm2d(planes * 4)
           )
   
       def forward(self, x):
           out = self.conv1(x)
           out = self.bn1(out)
           out = self.relu(out)
   
           out = self.conv2(out)
           out = self.bn2(out)
           out = self.relu(out)
   
           out = self.conv3(out)
           out = self.bn3(out)
   
           residual = self.extra_conv(x)
   
           if self.downsample is not None:  # downsample always is None, because stride=1 and inplanes=expansion * planes
               residual = self.downsample(x)
   
           out += residual
           out = self.relu(out)
   
           return out
   
   @property
   def _freeze_stages(self):
       if self.frozen_stages >= 0:
           for m in [self.layer0]:
               m.eval()
               for param in m.parameters():
                   param.requires_grad = False
       for i in range(1, self.frozen_stages + 1):
           m = getattr(self, 'layer{}'.format(i))
           m.eval()
           for param in m.parameters():
               param.requires_grad = False
   
   block_dic = {
       'BasicBlock': BasicBlock,
       'BottleneckA': BottleneckA,
       'BottleneckB': BottleneckB,
       'Bottleneck':Bottleneck
   }
   
   @BACKBONES.register_module
   class DetNet(nn.Module):
   
       def __init__(self, block, layers, num_classes=1000):
           self.inplanes = 64
           block = block_dic[block]
           super(DetNet, self).__init__()
           self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
           self.bn1 = nn.BatchNorm2d(64)
           self.relu = nn.ReLU(inplace=True)
           self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
           self.layer1 = self._make_layer(block, 64, layers[0])
           self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
           self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
           self.layer4 = self._make_new_layer(256, layers[3])
           self.layer5 = self._make_new_layer(256, layers[4])
           # self.avgpool = nn.AdaptiveAvgPool2d(1)
           # self.fc = nn.Linear(1024, num_classes)
   
           for m in self.modules():
               if isinstance(m, nn.Conv2d):
                   n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                   m.weight.data.normal_(0, math.sqrt(2. / n))
               elif isinstance(m, nn.BatchNorm2d):
                   m.weight.data.fill_(1)
                   m.bias.data.zero_()
   
       def init_weights(self, pretrained=None):
           if isinstance(pretrained, str):
               logger = logging.getLogger()
               load_checkpoint(self, pretrained, strict=False, logger=logger)
       def _make_layer(self, block, planes, blocks, stride=1):
           downsample = None
           if stride != 1 or self.inplanes != planes * block.expansion:
               downsample = nn.Sequential(
                   nn.Conv2d(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride, bias=False),
                   nn.BatchNorm2d(planes * block.expansion),
               )
   
           layers = []
           layers.append(block(self.inplanes, planes, stride, downsample))
           self.inplanes = planes * block.expansion
           for i in range(1, blocks):
               layers.append(block(self.inplanes, planes))
   
           return nn.Sequential(*layers)
   
       def _make_new_layer(self, planes, blocks):
           downsample = None
           block_b = BottleneckB
           block_a = BottleneckA
   
           layers = []
           layers.append(block_b(self.inplanes, planes, stride=1, downsample=downsample))
           self.inplanes = planes * block_b.expansion
           for i in range(1, blocks):
               layers.append(block_a(self.inplanes, planes))
   
           return nn.Sequential(*layers)
   
       def forward(self, x):
           x = self.conv1(x)
           x = self.bn1(x)
           x = self.relu(x)
           x = self.maxpool(x)
   
           outputs = []
           x = self.layer1(x)
           outputs.append(x)
           x = self.layer2(x)
           outputs.append(x)
           x = self.layer3(x)
           outputs.append(x)
           x = self.layer4(x)
           outputs.append(x)
           x = self.layer5(x)
           outputs.append(x)
   
           # x = self.avgpool(x)
           # x = x.view(x.size(0), -1)
           # x = self.fc(x)
   
           return outputs
   
   
   def load_pretrained_imagenet_weights(model, state_dict):
       own_state = model.state_dict()
       for name, param in state_dict.items():
           if ('layer4' in name) or ('layer5' in name) or ('fc' in name):
               continue
           if (name in own_state):
               if isinstance(param, nn.Parameter):
                   # backwards compatibility for serialized parameters
                   param = param.data
               try:
                   own_state[name].copy_(param)
               except Exception:
                   raise RuntimeError('While copying the parameter named {}, '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(name, own_state[name].size(), param.size()))
           else:
               raise KeyError('unexpected key "{}" in state_dict'
                              .format(name))
   
   ~~~

   

2. 在`mmdet/models/backbones`中把`__init__.py`改为：

   ~~~python、
   from .hrnet import HRNet
   from .resnet import ResNet, make_res_layer
   from .resnext import ResNeXt
   from .ssd_vgg import SSDVGG
   from .detnet import DetNet
   
   __all__ = ['ResNet', 'make_res_layer', 'ResNeXt', 'SSDVGG', 'HRNet', "DetNet"]
   
   ~~~

   主要是上面的第五行和第七行，加入`DetNet`。

3. 在mmdet/models/necks中修改fpn.py

   ```python
   # 原FPN
   # # build top-down path
   # used_backbone_levels = len(laterals)
   # for i in range(used_backbone_levels - 1, 0, -1):
   #     laterals[i - 1] += F.interpolate(
   #         laterals[i], scale_factor=2, mode='nearest')
   
   
   # 为detnet改造的FPN
   """
   原因：因为之的FPN使用的featus尺寸都是递减的，因此在构建FPN网络时有一个插值操作， 即上面的 F.interpolate(laterals[i], scale_factor=2, mode='nearest')函数。而detnet网络后面的3层featurs尺寸就没再下降，因此不需要这里的插值。
   """
   # build top-down path  开始进行特征的自顶向下融合
   used_backbone_levels = len(laterals) # 值为5
   for i in range(used_backbone_levels - 1, 2, -1):
       laterals[i - 1] += laterals[i]
   
   for i in range(2, 0, -1):
       laterals[i - 1] += F.interpolate(
           laterals[i], scale_factor=2, mode='nearest')
   ```

建议上边的操作不要直接在fpn.py文件中修改，因为修改之后就不能被其他backbone使用了，所以采用和上面类似的做法，新建一个文件detnet_fpn.py， 复制原文件，然后修改，修改之后，同样在`mmdet/models/necks/__init__.py` 中修改为以下内容：

~~~python

from .bfp import BFP
from .fpn import FPN
from .hrfpn import HRFPN
from .detnet_FPN import DetNetFPN

__all__ = ['FPN', 'BFP', 'HRFPN','DetNetFPN']
~~~

4. 修改config文件，以 `configs/mask_rcnn_r50_fpn_1x.py`为例:

   ~~~python
   model = dict(
       type='MaskRCNN',
       pretrained='work_dirs/pre_train/detnet59.pth',
       backbone=dict(
           type='DetNet', 
           block='Bottleneck',
           layers=[3, 4, 6, 3, 3],
           num_classes=2),
       neck=dict(
           type='DetNetFPN',
           in_channels=[256, 512, 1024, 1024, 1024],
           out_channels=256,
           end_level = 5,
           num_outs=5),
   ~~~

   DetNet59的预训练权重下载地址为：

   https://github.com/Morris-Chen007/deep_learning_models

   

   其他的修改就是改成自己的数据路径之类，现在就可以开始训练。

   `./tools/dist_train.sh configs/mask_rcnn_r50_fpn_1x.py 4 --validate`

   