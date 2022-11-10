import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import lib.models as models
import lib.dataset as dataset
from lib.core.config import config
from lib.core.config import get_model_name
from lib.core.config import update_config
from lib.core.config import update_dir
from lib.core.function import train
from lib.core.function import validate
from lib.core.loss import JointsMSELoss
from lib.utils.utils import create_logger
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint, load_checkpoint


def parse_args():
    """实现添加命令行参数功能，并可直接执行一部分参数

    此函数实现了当文件在命令行窗口下运行时，可以输入命令行参数的功能，可以更加便捷的更换底层特征提取网络，
    其中，--cfg参数为必需选项，其值应为配置数据文件(.yaml文件)的路径及名称
    此函数直接实现更改模型目录、日志目录、数据目录功能

    返回值：
        返回一个特定容器，其含有使用命令行参数时输入的数据
    """
    parser = argparse.ArgumentParser(description='Train keypoints network')  # 创建一个训练关键点的解析器
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)  # 更新config

    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)  # 日志的频率
    parser.add_argument('--gpus', help='gpus', type=str)  # GPU数量
    parser.add_argument('--workers', help='num of dataloader workers', type=int)  # 线程数
    parser.add_argument('--modelDir', help='model directory', type=str, default='')  # 模型目录
    parser.add_argument('--logDir', help='log directory', type=str, default='')  # 日志目录
    parser.add_argument('--dataDir', help='data directory', type=str, default='')  # 数据目录
    parser.add_argument('--data-format', help='data format', type=str, default='')  # 数据格式
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):  # TODO 此函数应该写到config里
    """重置配置文件
    将关于训练网络的命令行参数导入到配置文件中去
    此函数实现更改GPU、训练用图片格式、数据加载器数量的功能

    输入参数：
        config：固定参数，即传入config字典
        args：固定参数，即接收命令行参数的容器
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()  # 接收命令行参数传入的数据
    reset_config(config, args)  # 将命令行参数传入到config模块中去

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')  # 创建日志文件

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    print(final_output_dir)
    print(tb_log_dir)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=True)  # 设置骨干网络模型，用于特征提取

    model = eval('models.' + config.MODEL + '.get_multiview_pose_net')(
        backbone_model, config)  # 设置完整的模型


    this_dir = os.path.dirname(__file__)  # 得到当前目录
    shutil.copy2(
        os.path.join(this_dir, '../../lib/models', config.MODEL + '.py'),
        final_output_dir)
    shutil.copy2(args.cfg, final_output_dir)
    logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()  # 将模型设置为使用多块GPU计算

    criterion = JointsMSELoss(  # 实例化一个损失函数
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()

    optimizer = get_optimizer(config, model)  # 设置模型的优化器
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer = load_checkpoint(model, optimizer,
                                                        final_output_dir)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)
    # ————————————————————————————————————————————————————————————————————————————
    # Data loading code 数据加载代码
    normalize = transforms.Normalize(  # 使用均值和标准差对张量图像进行归一化，即设置标准化参数
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 建立训练集
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(  # 读取脚本，得到一个dataset对象
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),  # 将图片转为tensor类型
            normalize,  # 对图片归一化处理
        ]))
    # 建立验证集
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    # 建立训练集加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,  # 传递数据集
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),  # 设置每一个batch的大小
        shuffle=config.TRAIN.SHUFFLE,  # 是否打乱
        num_workers=config.WORKERS,  # 使用几个线程进行数据加载
        pin_memory=True)
    # 建立测试集加载器
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    best_model = False  # 布尔值指标，记录当前模型状态是否最好
    best_perf = 0.0  # 用于记录最好的性能指标
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        train(config, train_loader, model, criterion, optimizer, epoch,  # 训练
              final_output_dir, writer_dict)

        perf_indicator = validate(config, valid_loader, valid_dataset, model,  # 证实模型，得到当前的性能指标
                                  criterion, final_output_dir, writer_dict)

        if perf_indicator > best_perf:  # 判断当前指标是否为最好指标
            best_perf = perf_indicator  # 如果是则设置当前指标为最好指标
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))  # 记录当前时期(epoch)下的日志信息
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.module.state_dict(),  # 保存模型的所有状态信息
            'perf': perf_indicator,  # perf即performance，表现
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,  # 最终模型状态文件保存路径
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)  # 保存最终模型的状态信息
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
