import logging
import os
import time

import h5py
import numpy as np
import torch

from lib.core.config import get_model_name
from lib.core.evaluate import accuracy
from lib.core.inference import get_final_preds
from lib.utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def routing(raw_features, aggre_features, is_aggre, meta):
    if not is_aggre:
        return raw_features

    output = []
    for r, a, m in zip(raw_features, aggre_features, meta):
        view = torch.zeros_like(a)  # 生成和括号内变量维度维度一致的全是零的内容
        batch_size = a.size(0)
        for i in range(batch_size):
            s = m['source'][i]
            view[i] = a[i] if s == 'h36m' else r[i]
        output.append(view)
    return output


def train(config, data, model, criterion, optim, epoch, output_dir,
          writer_dict):
    """
    训练函数，每训练完一次即为每个epoch

    传入参数:
        cofig: 固定参数，config字典
        data: 传入一个数据加载器对象
        model: 传入一个完整的模型
        criterion: 传入一个损失函数对象
        optim: 传入一个优化器对象
        epoch: 传入当前时期(epoch)的值,用于后面输出日志文件
        output_dir:训练结果最终输出文件夹
        writer_dict:
    """
    is_aggre = config.NETWORK.AGGRE
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    # switch to train mode    切换到训练模式
    model.train()

    end = time.time()
    for i, (input, target, weight, meta) in enumerate(data):  # 每一个循环传入一个batch进行迭代训练
        data_time.update(time.time() - end)  # 每一个batch训练所用的时间

        # 前向传播
        raw_features, aggre_features = model(input)  # 将输入送入model里，得到模型预测的输出
        # raw_features原始特征，aggre_features聚合特征
        output = routing(raw_features, aggre_features, is_aggre, meta)  # 转化一下输出的格式

        loss = 0
        target_cuda = []
        for t, w, o in zip(target, weight, output):
            t = t.cuda(non_blocking=True)
            w = w.cuda(non_blocking=True)
            target_cuda.append(t)
            loss += criterion(o, t, w)  # 得到损失值
        target = target_cuda

        if is_aggre:
            for t, w, r in zip(target, weight, raw_features):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                loss += criterion(r, t, w)

        # 反向传播
        optim.zero_grad()  # 优化器权重清零
        loss.backward()  # 做反向传播
        optim.step()  # 进行优化更新
        losses.update(loss.item(), len(input) * input[0].size(0))

        nviews = len(output)
        acc = [None] * nviews
        cnt = [None] * nviews
        pre = [None] * nviews
        for j in range(nviews):
            _, acc[j], cnt[j], pre[j] = accuracy(
                output[j].detach().cpu().numpy(),
                target[j].detach().cpu().numpy())
        acc = np.mean(acc)
        cnt = np.mean(cnt)
        avg_acc.update(acc, cnt)

        batch_time.update(time.time() - end)
        end = time.time()

        # 按照特定频率打印训练的成果
        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Memory {memory:.1f}'.format(
                epoch, i, len(data), batch_time=batch_time,
                speed=len(input) * input[0].size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=avg_acc, memory=gpu_memory_usage)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            for k in range(len(input)):
                view_name = 'view_{}'.format(k + 1)
                prefix = '{}_{}_{:08}'.format(
                    os.path.join(output_dir, 'train'), view_name, i)
                save_debug_images(config, input[k], meta[k], target[k],
                                  pre[k] * 4, output[k], prefix)


def validate(config, loader, dataset, model, criterion, output_dir, writer_dict=None):
    """
    证实模型的性能

    输入参数:
        config:配置参数
        loader:数据集加载器
        dataset:数据集
        model:模型
        criterion:损失函数
        output_dir:输出目录

    返回值:
        perf_indicator:返回一个性能指标，反应出模型的能力
    """
    model.eval()  # 将模型设置为验证模式
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()

    nsamples = len(dataset) * 4  # 样本的数量
    is_aggre = config.NETWORK.AGGRE
    njoints = config.NETWORK.NUM_JOINTS  # 关节的数量
    height = int(config.NETWORK.HEATMAP_SIZE[0])  # 热力图的高度
    width = int(config.NETWORK.HEATMAP_SIZE[1])  # 热力图的宽度
    all_preds = np.zeros((nsamples, njoints, 3), dtype=np.float32)
    all_heatmaps = np.zeros(
        (nsamples, njoints, height, width), dtype=np.float32)

    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, weight, meta) in enumerate(loader):
            raw_features, aggre_features = model(input)
            output = routing(raw_features, aggre_features, is_aggre, meta)

            loss = 0
            target_cuda = []
            for t, w, o in zip(target, weight, output):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                target_cuda.append(t)
                loss += criterion(o, t, w)

            if is_aggre:
                for t, w, r in zip(target, weight, raw_features):
                    t = t.cuda(non_blocking=True)
                    w = w.cuda(non_blocking=True)
                    loss += criterion(r, t, w)
            target = target_cuda

            nimgs = len(input) * input[0].size(0)
            losses.update(loss.item(), nimgs)

            nviews = len(output)
            acc = [None] * nviews
            cnt = [None] * nviews
            pre = [None] * nviews
            for j in range(nviews):
                _, acc[j], cnt[j], pre[j] = accuracy(
                    output[j].detach().cpu().numpy(),
                    target[j].detach().cpu().numpy())
            acc = np.mean(acc)
            cnt = np.mean(cnt)
            avg_acc.update(acc, cnt)

            batch_time.update(time.time() - end)
            end = time.time()

            preds = np.zeros((nimgs, njoints, 3), dtype=np.float32)
            heatmaps = np.zeros(
                (nimgs, njoints, height, width), dtype=np.float32)
            for k, o, m in zip(range(nviews), output, meta):
                pred, maxval = get_final_preds(config,
                                               o.clone().cpu().numpy(),
                                               m['center'].numpy(),
                                               m['scale'].numpy())
                pred = pred[:, :, 0:2]
                pred = np.concatenate((pred, maxval), axis=2)
                preds[k::nviews] = pred
                heatmaps[k::nviews] = o.clone().cpu().numpy()

            all_preds[idx:idx + nimgs] = preds
            all_heatmaps[idx:idx + nimgs] = heatmaps
            idx += nimgs

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    loss=losses, acc=avg_acc)
                logger.info(msg)

                for k in range(len(input)):
                    view_name = 'view_{}'.format(k + 1)
                    prefix = '{}_{}_{:08}'.format(
                        os.path.join(output_dir, 'validation'), view_name, i)
                    save_debug_images(config, input[k], meta[k], target[k],
                                      pre[k] * 4, output[k], prefix)

        # save heatmaps and joint locations
        u2a = dataset.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = np.array(list(a2u.values()))

        save_file = config.TEST.HEATMAP_LOCATION_FILE
        file_name = os.path.join(output_dir, save_file)
        file = h5py.File(file_name, 'w')
        file['heatmaps'] = all_heatmaps[:, u, :, :]
        file['locations'] = all_preds[:, u, :]
        file['joint_names_order'] = a
        file.close()

        name_value, perf_indicator = dataset.evaluate(all_preds)
        names = name_value.keys()
        values = name_value.values()
        num_values = len(name_value)
        _, full_arch_name = get_model_name(config)
        logger.info('| Arch ' +
                    ' '.join(['| {}'.format(name) for name in names]) + ' |')
        logger.info('|---' * (num_values + 1) + '|')
        logger.info('| ' + full_arch_name + ' ' +
                    ' '.join(['| {:.3f}'.format(value) for value in values]) +
                    ' |')

    return perf_indicator


class AverageMeter(object):  # 继承 object 类的是新式类，不继承 object 类的是经典类
    """
    一个用于存储多个值的类

    Computes and stores the average and current value
    计算并存储当前值和平均值
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    #  self.reset()

    # def reset(self):
    #    pass

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
