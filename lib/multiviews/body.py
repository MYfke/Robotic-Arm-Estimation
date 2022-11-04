"""
定义了一个类，实现建立人体骨骼框架的功能
"""

import numpy as np


class HumanBody(object):
    """
    建立人体骨骼框架，其中关节节点有父子关系
    """

    def __init__(self):
        self.skeleton = self.get_skeleton()  # 骨骼
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)

    def get_skeleton(self):
        """
        得到骨骼框架，包含每个节点的编号，名称，子节点，节点等级
        """
        joint_names = [
            'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly',
            'neck', 'nose', 'head', 'lsho', 'lelb', 'lwri', 'rsho', 'relb',
            'rwri'
        ]
        # 将节点父子关系进行对应，例如root的子节点为1,4,7(rhip,lhip,belly)
        children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 11, 14],
                    [10], [], [12], [13], [], [15], [16], []]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[0]]   # 队列  这里的骨骼是骨骼点
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]

        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton


if __name__ == '__main__':
    hb = HumanBody()
    print(hb.skeleton)
    print(hb.skeleton_sorted_by_level)
