import torch.nn as nn


class JointsMSELoss(nn.Module):
    """损失函数"""
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(heatmap_pred.mul(target_weight.expand(-1, heatmap_pred.size(1))), # TODO 问题出在Tensor不自动广播了
                                       heatmap_gt.mul(target_weight.expand(-1, heatmap_pred.size(1))))  # 所以将Tensor扩张，但此写法比较丑陋
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss
