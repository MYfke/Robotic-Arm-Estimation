import os
import json
import numpy as np

import collections

from lib.dataset.joints_dataset import JointsDataset


class MultiViewH36M(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }
        self.union_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'thorax',
            9: 'neck',
            10: 'upper neck',
            11: 'nose',
            12: 'head',
            13: 'head top',
            14: 'lsho',
            15: 'lelb',
            16: 'lwri',
            17: 'rsho',
            18: 'relb',
            19: 'rwri'
        }  # 由于父类是机械臂的关节点，因此在此处重写联合关节点
        self.db = self._get_db()

        self.u2a_mapping = super().get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

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
        返回一个列表类型的数据库
        """
        # 得到数据库
        file_name = os.path.join(self.root, 'h36m', 'annot',  # 注释文件的文件名
                                 self.subset + '.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)  # 加载注释文件

        gt_db = []
        for a in anno:  # a为json文件中的一个元素
            gt_db.append({
                'imgname': a['image'],
                'camera_id': a['camera_id'],
                'image_id':a['image_id'],
                'bbox_center': a['center'],
                'bbox_size': a['scale'],
                # 'label': [keypoint['label'] for keypoint in a['keypoints']],
                'joints_2d': a['joints_2d'],
                'joints_3d': a['joints_3d'],
                'joints_vis': a['joints_vis'],
                'source': 'h36m',
                'camera': a['camera'],
                'subject': a['subject'],
                'action': a['action'],
                'subaction': a['subaction'],

            })

        return gt_db

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if self.is_train:
            filtered_grouping = filtered_grouping[::5]
        else:
            filtered_grouping = filtered_grouping[::64]

        return filtered_grouping

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

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
