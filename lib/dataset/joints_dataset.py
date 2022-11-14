import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from lib.utils.transforms import get_affine_transform
from lib.utils.transforms import affine_transform


class JointsDataset(Dataset):
    """关节点数据集类，这是一个父类，将由其他具体数据集对象来继承

    """

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset  # 子集

        self.root = cfg.DATASET.ROOT  # 数据集处理模块的路径
        self.data_format = cfg.DATASET.DATA_FORMAT  # 图片的格式
        self.scale_factor = cfg.DATASET.SCALE_FACTOR  # 数据增强比例系数
        self.rotation_factor = cfg.DATASET.ROT_FACTOR  # 旋转系数

        self.image_size = cfg.NETWORK.IMAGE_SIZE  # 图片尺寸
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE  # 热力图尺寸

        self.sigma = cfg.NETWORK.SIGMA  # 学习率
        self.transform = transform
        self.db = []  # 数据库

        self.num_joints = cfg.NETWORK.NUM_JOINTS  # 关节点数量
        self.union_joints = {  # 联合关节点，包含所有可能出现的关节点，每一个关节点对应一个索引
            0: 'base',       # 基座
            1: 'shoulder',   # 肩部
            2: 'big arm',    # 大臂
            3: 'small arm',  # 小臂
            4: 'wrist',      # 腕部
            5: 'end',        # 末端
        }
        self.actual_joints = {}  # 实际数据集中的键值对
        self.u2a_mapping = {}  # 用户到应用程序的热力图，字典类型，用来接收get_mapping()的返回值

    def get_mapping(self):
        """
        创建并返回一张联合关节点和实际关节点的映射表 (字典类型)

        因为不同的数据集标注的关节点不同，因此返回一张实际数据集的联合关节点的映射表，没有的关节点用 * 替代，
        例如 {0: 0, 1: 1, 2: '*', 3: 2...} 就表示联合关节点0对应实际关节点0，联合关节点2无对应，联合关节点3对应实际关节点2
        """
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        u2a_mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            u2a_mapping[key] = k
        return u2a_mapping  # 这个返回值用self.u2a_mapping接收

    def do_mapping(self):
        # TODO 此函数的作用暂时未知
        mapping = self.u2a_mapping  # 这张mapping就是get_mapping()的返回值

        for item in self.db:  # 循环遍历数据库中的每个项，此时的数据集其实就是注释文件
            joints = item['joints_2d']  # 得到关节点的二维坐标
            joints_vis = item['joints_vis']  # 关节点可见值，1可见，0不可见

            njoints = len(mapping)
            joints_union = np.zeros(shape=(njoints, 2))
            joints_union_vis = np.zeros(shape=(njoints, 3))

            for i in range(njoints):
                if mapping[i] != '*':
                    index = int(mapping[i])
                    joints_union[i] = joints[index]
                    joints_union_vis[i] = joints_vis[index]
            item['joints_2d'] = joints_union  # TODO 扩张到三维的作用未知
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError

    def __len__(self, ):
        # 返回数据集中元素的个数
        return len(self.db)

    def __getitem__(self, idx):
        """
        魔法方法，实现了数据集取出数据的功能
        """
        db_rec = copy.deepcopy(self.db[idx])

        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                              db_rec['imgname'])
        if self.data_format == 'zip':
            from lib.utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(  # 将图片数据读入
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        joints = db_rec['joints_2d'].copy()
        joints_vis = db_rec['joints_vis'].copy()

        center = np.array(db_rec['bbox_center']).copy()
        scale = np.array(db_rec['bbox_size']).copy()
        rotation = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

        trans = get_affine_transform(center, scale, rotation, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                if (np.min(joints[i, :2]) < 0 or
                        joints[i, 0] >= self.image_size[0] or
                        joints[i, 1] >= self.image_size[1]):
                    joints_vis[i, :] = 0

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],
            'joints_2d_transformed': joints,
            'joints_vis': joints_vis,
            'source': db_rec['source']
        }
        return input, target, target_weight, meta

    def generate_target(self, joints_3d, joints_vis):
        target, weight = self.generate_heatmap(joints_3d, joints_vis)
        return target, weight

    def generate_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
