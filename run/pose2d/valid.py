import argparse
import os
import pprint

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# noinspection PyUnresolvedReferences
import lib.models as models
# noinspection PyUnresolvedReferences
import lib.dataset as dataset

from lib.core.config import config
from lib.core.config import update_config
from lib.core.config import reset_config
from lib.core.function import validate
from lib.core.loss import JointsMSELoss
from lib.utils.utils import create_logger


def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, _ = parser.parse_known_args()
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--state', help='the state of model which is used to test (best or final)', default='best', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--model-file', help='model state file', type=str)
    parser.add_argument('--flip-test', help='use flip test', action='store_true')
    parser.add_argument('--post-process', help='use post process', action='store_true')
    parser.add_argument('--shift-heatmap', help='shift heatmap', action='store_true')

    # philly
    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--data-format', help='data format', type=str, default='')

    args = parser.parse_args()
    reset_config(args)  # 将命令行参数传入到config模块中去

    return args


def main():
    args = parse_args()
    reset_config(args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting    cudnn 相关设置
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=False)

    model = eval('models.' + config.MODEL + '.get_multiview_pose_net')(
        backbone_model, config)

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_path = 'model_best.pth.tar' if config.TEST.STATE == 'best' else 'final_state.pth.tar'
        model_state_file = os.path.join(final_output_dir, model_path)
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    # 定义损失函数（标准）和优化器
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # evaluate on validation set  评估验证集
    validate(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)


if __name__ == '__main__':
    main()
