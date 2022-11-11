"""
此模块含有两大基本内容，一是名为config的字典，其含有基本配置信息
其二为有关跟新config字典的一些函数
"""
import os
import yaml
import numpy as np
from easydict import EasyDict

config = EasyDict()  # 创建一个名为config的字典对象

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'  # 骨干网络
config.MODEL = 'multiview_pose_resnet'  # 视角融合网络
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100  # frequency - 每训练多少的batch进行一次打印

# hrnet definition  hrnet 的定义
config.MODEL_EXTRA = EasyDict()
config.MODEL_EXTRA.PRETRAINED_LAYERS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2',
    'transition2', 'stage3', 'transition3', 'stage4'
]  # 配置预训练层
config.MODEL_EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL_EXTRA.STAGE2 = EasyDict()
config.MODEL_EXTRA.STAGE2.NUM_MODULES = 1
config.MODEL_EXTRA.STAGE2.NUM_BRANCHES = 2
config.MODEL_EXTRA.STAGE2.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
config.MODEL_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
config.MODEL_EXTRA.STAGE2.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE3 = EasyDict()
config.MODEL_EXTRA.STAGE3.NUM_MODULES = 4
config.MODEL_EXTRA.STAGE3.NUM_BRANCHES = 3
config.MODEL_EXTRA.STAGE3.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.MODEL_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.MODEL_EXTRA.STAGE3.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE4 = EasyDict()
config.MODEL_EXTRA.STAGE4.NUM_MODULES = 3
config.MODEL_EXTRA.STAGE4.NUM_BRANCHES = 4
config.MODEL_EXTRA.STAGE4.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.MODEL_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.MODEL_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

# Cudnn related params  Cudnn 相关参数
config.CUDNN = EasyDict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK  NETWORK 的常用参数
config.NETWORK = EasyDict()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.NUM_JOINTS = 6
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.SIGMA = 2
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.AGGRE = True

# pose_resnet related params  pose_resnet 相关参数
config.POSE_RESNET = EasyDict()
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.LOSS = EasyDict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET related params  数据集相关参数
config.DATASET = EasyDict()
config.DATASET.ROOT = '../data/'
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multi_view_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 0
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True

# training data augmentation  训练数据增强
config.DATASET.SCALE_FACTOR = 0
config.DATASET.ROT_FACTOR = 0

# train  训练
config.TRAIN = EasyDict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 10

config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

# testing  测试
config.TEST = EasyDict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = ''
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.HEATMAP_LOCATION_FILE = 'predicted_heatmaps.h5'

# debug  调试
config.DEBUG = EasyDict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True

# pictorial structure 图片结构
config.PICT_STRUCT = EasyDict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.PAIRWISE_FILE = ''
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = 2000
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    """用来更新config模块

    此函数通过加载不同的配置数据文件，用来更新
    config模块，从而实现快速更改底层特征提取网络

    传入参数：
        config_file：将要使用的模型的配置数据文件(.yaml文件)的文件路径及名称

    """
    # exp_config = None
    with open(config_file) as f:
        exp_config = EasyDict(yaml.load(f, Loader=yaml.Loader))
        for k, v in exp_config.items():
            # k即为key，关键字
            # v即为values，关键字所对应的值
            if k in config:
                if isinstance(v, dict):  # 判断v是否为字典类型
                    _update_dict(k, v)  # 如果是则使用配置数据文件(.yaml文件)更新config模块
                else:
                    if k == 'SCALES':  # 判断k是否为SCALES，即比例，如果是则特定要求新添“比例”键值对
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v  # 如果都不是，直接在config文件后新添一个键值对
            else:
                raise ValueError("{} not exist in config.py".format(k))  # 如果k不是字典类型则报错


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, EasyDict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)  # aml.dump 将一个python对象生成为yaml文档


def reset_config(args):
    """
    重置配置文件,将关于命令行参数导入到配置文件中去
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.modelDir:
        config.OUTPUT_DIR = args.modelDir
    if args.logDir:
        config.LOG_DIR = args.logDir
    if args.dataDir:
        config.DATA_DIR = args.dataDir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR, config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    """获取将要使用的模型名称

    在models目录下含有使用不同底层网络构建的模型，
    此函数可以通过返回不同的模块名称，来实现构建不同的多视角信息融合网络

    输入参数：
        cfg：固定参数，即config字典

    返回值：
        name：一个带有模型名称及层数的字符串 (例如 multiview_pose_resnet_50 )
        full name：一个信息更加全的字符串 (例如 320x320_multiview_pose_resnet_50_d256d256d256 )
    """

    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=cfg.POSE_RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.POSE_RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    config_file = "../../experiments-local/RoboArm/resnet50/256_fusion.yaml"
    print(os.getcwd())
    update_config(config_file)
