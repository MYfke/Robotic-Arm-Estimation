import os
import logging

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


# noinspection PyTypeChecker
def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    带填充的3x3卷积层

    输入参数：
        in_planes: 输入通道数
        out_planes: 输出通道数
        stride：步长，默认为 1
    返回值：
        返回一个二维卷积层对象
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """resnet由一个个残差块拼接而成，此处定义最基础的残差块"""
    expansion = 1  # 此参数对应残差结构当中主分支所采用的卷积核的个数变化系数

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        输入参数：
            inplanes：输入特征矩阵的深度
            planes：残差模块输出的特征矩阵深度

        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)  # 归一化层，此层不改变通道数，因此只需要传入输入深度
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample  # 下采样
        self.stride = stride

    def forward(self, x):  # 输入特征矩阵x
        residual = x  # 将x赋值给残差分支上特征矩阵

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 是否对残差分支进行下采样
            residual = self.downsample(x)

        out += residual  # 将主分支与残差分支相加
        out = self.relu(out)  # 进行激活

        return out  # 返回一个tensor类型的特征矩阵


# noinspection PyTypeChecker
class Bottleneck(nn.Module):
    """另一种残差结构块，用于更深的resnet"""
    expansion = 4  # 同一个残差块中第三层卷积核个数的扩大倍率

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,  # 此处的步长为传入的步长
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,  # 卷积核个数扩大4倍，使深度扩大4倍
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
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


# noinspection PyTypeChecker
class PoseResNet(nn.Module):
    """定义整个网络的筐框架部分

    在resnet中，多个残差块组成一个大层（第一个卷积层不属于大层），整个网络由多个大层组成

    传入参数：
        block：传入一个将要使用的残差结构块的类型，网络较浅使用BasicBlock，较深使用Bottleneck残差块（50层以上）
        layers：列表类型参数，传入每个大层使用残差块结构的数目
    """

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64  # 输入特征矩阵的深度（第一个大卷积层并池化之后的深度）
        self.deconv_with_bias = cfg.POSE_RESNET.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  # 宽高缩减为一半
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 以下为大层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers 用于反卷积层
        self.deconv_layers = self._make_deconv_layer(
            cfg.POSE_RESNET.NUM_DECONV_LAYERS,
            cfg.POSE_RESNET.NUM_DECONV_FILTERS,
            cfg.POSE_RESNET.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.POSE_RESNET.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.NETWORK.NUM_JOINTS,  #输出维度为关节点的数量
            kernel_size=cfg.POSE_RESNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.POSE_RESNET.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        """构建大层

        使用多个残差结构，构成resnet中的一个大层

        传入参数：
            block:残差块的类型
            planes：残差结构中第一个卷积层所使用的卷积核的个数
            blocks：包含的残差块的个数
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 构建实线部分的残差结构
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 构建虚线部分的残差结构
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)  # 将list列表转化为非关键字参数

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """构建反卷积层"""
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)

            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),  # 选择网络层数的resnet
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, **kwargs):
    """
    得到姿态识别网络

    输出参数：
        cfg: 固定参数，config字典
        is_train: 传入布尔值
    返回值：
        model:返回一个模型
    """
    num_layers = cfg.POSE_RESNET.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if is_train:
        model.init_weights(cfg.NETWORK.PRETRAINED)

    return model
