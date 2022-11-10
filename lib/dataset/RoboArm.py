import os
import json
import numpy as np

from lib.dataset.joints_dataset import JointsDataset


class RoboArmDataset(JointsDataset):
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

        self.u2a_mapping = super().get_mapping()  # {0: 6, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5, 7: '*', 8: 7, 9: '*', 10: 8, 11: '*', 12: '*', 13: 9, 14: 13, 15: 14, 16: 15, 17: 12, 18: 11, 19: 10}
        super().do_mapping()  # {0: 6, 1: 2, 2: 1, 3: 0, 4: 3, 5: 4, 6: 5, 7: '*', 8: 7, 9: '*', 10: 8, 11: '*', 12: '*', 13: 9, 14: 13, 15: 14, 16: 15, 17: 12, 18: 11, 19: 10}

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)

    def _get_db(self):
        """
        返回一个列表类型的数据库
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
                # 'joints_3d': np.zeros((16, 3)),
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

    def __getitem__(self, idx):
        item = np.random.choice(self.grouping[idx])
        return super().__getitem__(item)

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])