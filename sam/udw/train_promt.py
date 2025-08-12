import os
import shutil
import json
import time
from torch.cuda import amp
import tqdm
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset, get_logger, get_model
from toolbox import averageMeter, runningScore, ClassWeight, save_ckpt
from scipy.ndimage import label as scipy_label

# 提示生成函数
def generate_topk_prompts(labels, features, num_points_base=3, topk_points=10):
    batch_size, H, W = labels.shape
    _, C, Hf, Wf = features.shape
    points = []
    point_labels = []

    for b in range(batch_size):
        label = labels[b].cpu().numpy()
        feat = features[b].abs().mean(dim=0).detach().cpu().numpy()
        feat = feat / (feat.max() + 1e-6)
        feat_resized = F.interpolate(
            torch.tensor(feat[None, None, :, :], dtype=torch.float32),
            size=(H, W), mode='bilinear', align_corners=True
        )[0, 0].numpy()

        batch_points = []
        batch_labels = []
        for cls in range(1, 8):  # 类别 1-7
            cls_mask = label == cls
            labeled, num_regions = scipy_label(cls_mask)
            num_points = min(max(1, num_regions // 2), num_points_base)
            if cls_mask.sum() > 0:
                cls_scores = feat_resized * cls_mask
                flat_scores = cls_scores.flatten()
                topk_indices = np.argsort(flat_scores)[-topk_points:]
                topk_indices = np.random.choice(topk_indices, min(num_points, len(topk_indices)), replace=False)
                cls_points = np.stack([topk_indices % W, topk_indices // W], axis=1)
                cls_labels = torch.ones(len(topk_indices), dtype=torch.float32)
                if len(topk_indices) < num_points:
                    cls_points = np.pad(cls_points, ((0, num_points - len(topk_indices)), (0, 0)), mode='constant')
                    cls_labels = torch.cat([cls_labels, -torch.ones(num_points - len(topk_indices))])
            else:
                cls_points = np.zeros((num_points, 2), dtype=np.float32)
                cls_labels = -torch.ones(num_points, dtype=torch.float32)
            batch_points.append(cls_points)
            batch_labels.append(cls_labels)

        bg_mask = label == 0
        num_points = num_points_base
        if bg_mask.sum() > 0:
            bg_scores = feat_resized * bg_mask
            flat_scores = bg_scores.flatten()
            topk_indices = np.argsort(flat_scores)[:topk_points]
            topk_indices = np.random.choice(topk_indices, min(num_points, len(topk_indices)), replace=False)
            neg_points = np.stack([topk_indices % W, topk_indices // W], axis=1)
            neg_labels = torch.zeros(len(topk_indices), dtype=torch.float32)  # 负点设为 0
            if len(topk_indices) < num_points:
                neg_points = np.pad(neg_points, ((0, num_points - len(topk_indices)), (0, 0)), mode='constant')
                neg_labels = torch.cat([neg_labels, -torch.ones(num_points - len(topk_indices))])
        else:
            neg_points = np.zeros((num_points, 2), dtype=np.float32)
            neg_labels = -torch.ones(num_points, dtype=torch.float32)

        batch_points = np.concatenate(batch_points + [neg_points], axis=0)
        batch_labels = torch.cat(batch_labels + [neg_labels], dim=0)
        points.append(batch_points)
        point_labels.append(batch_labels)

    points = torch.tensor(np.stack(points), dtype=torch.float32)
    point_labels = torch.stack(point_labels)
    return points, point_labels

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}-SAM_NEW_py+promt'
    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    print('****************student_PTH loading Finish!*************')

    trainset, *testset = get_dataset(cfg)
    device = torch.device('cuda:0')
    args.distributed = False
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
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(testset[0], batch_size=1, shuffle=False, num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    
    params_list = model.parameters()
    optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    Scaler = amp.GradScaler()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])

    class_weight = torch.from_numpy(class_weight).float().to(device)
    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=None)

    flag = True
    miou = 0
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)

        model.train()
        train_loss_meter.reset()

        for i, sample in enumerate(tqdm.tqdm(train_loader)):
            optimizer.zero_grad()

            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            with amp.autocast():
                # 无提示运行一次，获取 fuse4 特征
                predict, features = model(image, depth)
                fuse1, fuse2, fuse3, fuse4 = features
                
                # 生成 top-k 提示
                points4, point_labels4 = generate_topk_prompts(label, fuse4)
                points4, point_labels4 = points4.to(device), point_labels4.to(device)
                points3, point_labels3 = generate_topk_prompts(label, fuse3)
                points3, point_labels3 = points3.to(device), point_labels3.to(device)
                points2, point_labels2 = generate_topk_prompts(label, fuse2)
                points2, point_labels2 = points2.to(device), point_labels2.to(device)
                points1, point_labels1 = generate_topk_prompts(label, fuse1)
                points1, point_labels1 = points1.to(device), point_labels1.to(device)

                points = [points1, points2, points3, points4]
                point_labels = [point_labels1, point_labels2, point_labels3, point_labels4]
                
                # 使用提示运行模型
                predict, _ = model(image, depth, points=points, point_labels=point_labels)
                loss = criterion(predict, label)

            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()
            val_loss_meter.reset()

            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                predict, features = model(image, depth)  # 验证阶段无提示
                loss = criterion(predict, label)
                val_loss_meter.update(loss.item())

                predict = predict.max(1)[1].cpu().numpy()
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

        if args.local_rank == 0:
            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f} '
                f', mAcc={running_metrics_val.get_scores()[0]["mAcc: "]:.3f}'
                f', mIoU={running_metrics_val.get_scores()[0]["mIou: "]:.3f}'
                f', best_mIoU={miou:.3f}')
            save_ckpt(logdir, model, kind='end')
            newmiou = running_metrics_val.get_scores()[0]["mIou: "]
            if newmiou > 0.755:
                save_ckpt(logdir, model, kind='epoch', cur_epoch=ep)
            if newmiou > miou:
                save_ckpt(logdir, model, kind='best')
                miou = newmiou

    save_ckpt(logdir, model, kind='end')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/SUIM.json")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--opt_level", type=str, default='O1')
    args = parser.parse_args()
    run(args)