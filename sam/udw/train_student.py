#os 库是Python标准库，包含几百个函数，常用的有路径操作、进程管理、环境参数等。
import os
#高级的 文件、文件夹、压缩包 处理模块
import shutil
#JSON(JavaScript Object Notation, JS 对象简谱) 是一种轻量级的数据交换格式。
import json
import time
#加速
# from apex import amp
from torch.cuda import amp
import tqdm
# import apex
import numpy as np
#分布式通信包
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
#寻找最适合当前配置的高效算法，来达到优化运行效率的问题
import torch.backends.cudnn as cudnn
#调整学习率（learning rate）的方法

from torch.optim.lr_scheduler import LambdaLR, StepLR
#实现自由的数据读取,dataloadateset读取训练集dataset (Dataset): 加载数据的数据集
# * batch_size (int, optional): 每批加载多少个样本
# * shuffle (bool, optional): 设置为“真”时,在每个epoch对数据打乱.（默认：False）
# * sampler (Sampler, optional): 定义从数据集中提取样本的策略,返回一个样本
# * batch_sampler (Sampler, optional): like sampler, but returns a batch of indices at a time 返回一批样本. 与atch_size, shuffle, sampler和 drop_last互斥.
# * num_workers (int, optional): 用于加载数据的子进程数。0表示数据将在主进程中加载​​。（默认：0）
# * collate_fn (callable, optional): 合并样本列表以形成一个 mini-batch.  #　callable可调用对象
# * pin_memory (bool, optional): 如果为 True, 数据加载器会将张量复制到 CUDA 固定内存中,然后再返回它们.
# * drop_last (bool, optional): 设定为 True 如果数据集大小不能被批量大小整除的时候, 将丢掉最后一个不完整的batch,(默认：False).
# * timeout (numeric, optional): 如果为正值，则为从工作人员收集批次的超时值。应始终是非负的。（默认：0）
# * worker_init_fn (callable, optional): If not None, this will be called on each worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding and before data loading. (default: None)．

#from toolbox.datasets.nyuv2 import train_collate_fn
#from lib.data_fetcher import DataPrefetcher
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox.loss import lovaszSoftmax
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
# from toolbox import get_model_t
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt,load_ckpt
# from KD_loss.ContrastiveSeg.lib.loss.loss_contrast import PixelContrastLoss
from toolbox import Ranger
# from toolbox.kdlosses import *
torch.manual_seed(123)
#程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
cudnn.benchmark = True
def map_ignore(labels, ignore_values, index=-1):
    labels = labels.clone()
    for v in ignore_values:
        labels[labels == v] = index
    return labels


def run(args):
#载configs下的配置文件
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)
    #用于保存日志文件或其他的与时间相关的数据
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-SAM_onlyadapter(c)'

    #logdir = 'run/2020-12-23-18-38'
    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    #将源文件路径复制到logdir
    shutil.copy(args.config, logdir)

    #方便调试维护代码
    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    # model.load_pre('/media/wby/shuju/Seg_Water/Udw3/sam/udw/toolbox/Pretrain/mit_b0.pth')
    print('****************student_PTH loading Finish!*************')
    #将get_dataset返回的对象分别传给train、test
    trainset, *testset = get_dataset(cfg)
#torch.device代表将torch.Tensor分配到的设备的对象
    device = torch.device('cuda:0')
    args.distributed = False
#environ是一个字符串所对应环境的映像对象
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        # model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
##################################加载PTH预训练############################################
    # args.model_pretrained = True
    # if args.model_pretrained:
    #         print('load weights from ckpt_pretrained  ...')
    #         save_model = torch.load('/home/yangenquan/aabyby/ckpt_pretrained/model_SegformerB2_best.pth')
    #         model_dict = model.state_dict()
    #         state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys() and (v.shape == model_dict[k].shape)}
    #         model_dict.update(state_dict)
    #         model.load_state_dict(model_dict, strict=False)
    #         print('加载OK!!!')
    # else:
    #     assert 'unsupported model'
###########################################################################################
    model.to(device)
    # teacher.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    #                                             drop_last=True解决照片留单然后导致batch变成1
    val_loader = DataLoader(testset[0], batch_size=1, shuffle=False,num_workers=cfg['num_workers'],pin_memory=True, drop_last=True)
    params_list = model.parameters()
    # wd_params, non_wd_params = model.get_params()
    # params_list = [{'params': wd_params, },
    #                {'params': non_wd_params, 'weight_decay': 0}]
    #weight_decy是放在正则项（regularization）前面的一个系数,调节模型复杂度对损失函数的影响,防止过拟合
    # optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    Scaler = amp.GradScaler()
    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    # if args.distributed:
    #   model = torch.nn.parallel.DistributedDataParallel(model)

    # class weight 计算
    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)
    # print(class_weight)
    # class_weight[cfg['id_unlabel']] = 0

    # 损失函数 & 类别权重平衡 & 训练时ignore unlabel
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    # contrastive = PixelContrastLoss().to(device)
    # criterion = MscCrossEntropyLoss().to(device)
    # criterion = lovaszSoftmax(weight=class_weight).to(device)

    # 指标 包含unlabel
    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    # running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=None)
    # 每个epoch迭代循环

    flag = True #为了先保存一次模型做的判断
    #设置一个初始miou
    miou = 0
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        # training
        model.train()
        train_loss_meter.reset()
        # teacher.eval()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # print("image:", depth.shape)

            # print("label:", label.shape)

            with amp.autocast():

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))

                # predict = model(torch.squeeze(image, dim=1), depth)

                predict = model(image, depth)
                # print(predict.shape)
                # predict = model(image)
                # print('predict', predict.shape)
                # loss = criterion(predict[0], label) + criterion(predict[1], label) + criterion(predict[2], label)
                # loss = criterion(predict, label)
                # label = label.permute(0,3,1,2)
                # print(label[i].shape)
                # print(label[i].permute(2, 0, 1).shape)
                # print(predict[i].shape)
                # b, c, h, w = predict.size()

                # loss = criterion(predict[3], label)
                # loss = criterion(predict[0], label) + criterion(predict[1][0], label)
                # loss = criterion(predict, label.permute(3, 0, 1, 2))
                # + criterion(predict[1][0], label) + criterion(predict[1][1], label) + criterion(predict[1][2], label)
                loss = criterion(predict[0], label)
                # loss = criterion(predict, label)  + contrastive(predict ,label,predict)

            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            # optimizer.step()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        # val
        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()

            val_loss_meter.reset()
            ################### val edit #######################
            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                # predict = model(torch.cat((image.unsqueeze(2), depth.unsqueeze(2)), dim=2))
                # loss = criterion(predict, label)
                predict = model(image, depth)
                # predict = model(image)
                # print("label", label)
                # loss = criterion(predict[3], label)
                loss = criterion(predict[0], label)    #############################2

                val_loss_meter.update(loss.item())    ##########################

                # predict = predict[0].cpu().numpy()  # [1, h, w]     #############################3
                # print('predict',predict.shape)
                # predict = predict[3].max(1)[1].cpu().numpy()  # [1, h, w
                predict = predict[0].max(1)[1].cpu().numpy()
                # predict = predict[0].max(1)[1].cpu().nump         y()# [1, h, w]
                # print('predict',predict.shape)
                label = label.cpu().numpy()

            ###################edit end#########################
                running_metrics_val.update(label, predict)

        if args.local_rank == 0:
            logger.info(
                 f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f} '
                # f', PA={running_metrics_val.get_scores()[0]["pixel_acc:6 " ]:.3f}'
                # f', CA={running_metrics_val.get_scores()[0]["class_acc: " ]:.3f}'
                f', mAcc={running_metrics_val.get_scores()[0]["mAcc: "]:.3f}'
                f', miou={running_metrics_val.get_scores()[0]["mIou: "]:.3f}'
                f', best_miou={miou:.3f}')
            save_ckpt(logdir, model, kind='end')

            # save_ckpt(logdir, model, kind='epoch', cur_epoch=ep)
            newmiou = running_metrics_val.get_scores()[0]["mIou: "]
            if newmiou > 0.740:
                save_ckpt(logdir, model, kind='epoch', cur_epoch=ep)

            if newmiou > miou:
                save_ckpt(logdir, model, kind='best')  #消融可能不一样
                miou = newmiou

    save_ckpt(logdir, model, kind='end')  #保存最后一个模型参数

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/SUIM.json",
        # default="configs/nyuv2.json",
        # default="configs/sunrgbd.json",
        # default="configs/WE3DS.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )

    args = parser.parse_args()

    run(args)
