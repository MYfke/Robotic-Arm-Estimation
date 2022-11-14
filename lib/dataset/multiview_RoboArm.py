import os
import json
import numpy as np
import collections

from lib.dataset.joints_dataset import JointsDataset


class MultiviewRoboArmDataset(JointsDataset):
    """
    创建一个用于机械臂数据集的类
    """
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'base',  # 基座
            1: 'shoulder',  # 肩部
            2: 'big arm',  # 大臂
            3: 'small arm',  # 小臂
            4: 'wrist',  # 腕部
            5: 'end',  # 末端
        }
        self.db = self._get_db()  # 得到一个包含图片名与注释的数据库

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        # 一个grouping，包含了多个视角同一时刻的图片
        # 也就是说，这个数据集的每取出一份数据形状应该为 [batch,grouping,channel,width,height]
        self.group_size = len(self.grouping)  # 视角个数

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def _get_db(self):
        """
        返回一个列表类型的数据库，包含所有的标签信息
        """
        # 得到数据库
        file_name = os.path.join(self.root, 'RoboArm', 'annot',  # 注释文件的文件名
                                 self.subset + '.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)  # 加载注释文件

        gt_db = []
        for a in anno:  # a为json文件中的一个元素
            gt_db.append({
                'imgname': a['imgname'],
                'camera_id': a['camera_id'],
                'bbox_center': a['bbox_center'],
                'bbox_size': a['bbox_size'],
                'label': [keypoint['label'] for keypoint in a['keypoints']],
                'joints_2d': [keypoint['coord'] for keypoint in a['keypoints']],
                # 'joints_3d': np.zeros((16, 3)),TODO 目前数据集没有3D信息，后续需要补充
                'joints_vis': [keypoint['visible'] for keypoint in a['keypoints']],
                'source': 'RoboArm'
            })

        return gt_db

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            camera_id = db[i]['camera_id']
            if db[i]['imgname'][2:] not in grouping:
                grouping[db[i]['imgname'][2:]] = [-1, -1, -1, -1]
            grouping[db[i]['imgname'][2:]][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if not self.is_train:
            filtered_grouping = filtered_grouping[::64]

        return filtered_grouping

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()

        headsize = self.image_size[0] / 10.0
        threshold = 0.5

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
        gt = np.array(gt)
        pred = pred[:, su, :2]

        distance = np.sqrt(np.sum((gt - pred) ** 2, axis=2))
        detected = (distance <= headsize * threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        return name_values, np.mean(joint_detection_rate)
