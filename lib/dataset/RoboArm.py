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
        print()

    def _get_db(self):
        """返回一个列表类型的数据库

        列表不断添加一个这个样子的东西
        [{'image': '015601864.jpg',
        'center': array([593.     , 301.31569]),
        'scale': array([3.7763075, 3.7763075]),
        'joints_2d': array([[619.    , 393.    ],[615.    , 268.    ],[572.    , 184.    ],
                            [646.    , 187.    ],[660.    , 220.    ],[655.    , 230.    ],
                            [609.    , 186.    ],[646.    , 175.    ],[636.0201, 188.8183],
                            [694.9799, 107.1817],[605.    , 216.    ],[552.    , 160.    ],
                            [600.    , 166.    ],[691.    , 184.    ],[692.    , 239.    ],
                            [687.    , 312.    ]]),
        'joints_3d': array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],
                            [0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.],
                            [0., 0., 0.],[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]]),
        'joints_vis': array([[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],
                            [1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.],
                            [1., 1., 0.],[1., 1., 0.],[1., 1., 0.],[1., 1., 0.]]),
        'source': 'mpii'}  {……}]
        不断添加格式如上的元素
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
                'bbox_center': a['bbox_center'],
                'bbox_size': a['bbox_size'],
                'label': [keypoint['label'] for keypoint in a['keypoints']],
                'joints_2d': [keypoint['coord'] for keypoint in a['keypoints']],
                # 'joints_3d': np.zeros((16, 3)),
                'joints_vis': [keypoint['visible'] for keypoint in a['keypoints']],
                'source': 'RoboArm'
            })

        return gt_db
