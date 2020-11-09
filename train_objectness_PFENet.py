"""Training Script"""
import os
import shutil

import random
import argparse

from config import cfg
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from distutils.version import LooseVersion
import time
import pdb

from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS
from utils_seg import AverageMeter, parse_devices, setup_logger
from models import ModelBuilder

from util_PFENet import dataset_obj as dataset
from util_PFENet import transform

import sys
sys.path.append('/home/yz9244/PANet/cocoapi/PythonAPI/')

def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def checkpoint(nets, history, cfg, iter_idx):
    print('Saving checkpoints...')
    (net_objectness, net_decoder, crit) = nets

    dict_objectness = net_objectness.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_objectness,
        '{}/objectness_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_decoder,
        '{}/objectness_decoder_iter_{}.pth'.format(cfg.DIR, iter_idx))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, cfg):
    (net_objectness, net_decoder, crit) = nets
    if cfg.TRAIN.fix_encoder:
        optimizer_objectness = None
    else:
        optimizer_objectness = torch.optim.SGD(
            group_weight(net_objectness),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    
    return (optimizer_objectness, optimizer_decoder)

def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_objectness, optimizer_decoder) = optimizers
    if not cfg.TRAIN.fix_encoder:
        for param_group in optimizer_objectness.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    torch.cuda.set_device(gpus[0])
    print('###### Create model ######')
    net_objectness = ModelBuilder.build_objectness(
        arch=cfg.MODEL.arch_objectness,
        weights=cfg.MODEL.weights_enc_query,
        fix_encoder=cfg.TRAIN.fix_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        input_dim=cfg.MODEL.decoder_dim,
        fc_dim=cfg.MODEL.fc_dim,
        ppm_dim=cfg.MODEL.ppm_dim,
        num_class=2,
        weights=cfg.MODEL.weights_decoder,
        dropout_rate=cfg.MODEL.dropout_rate,
        use_dropout=cfg.MODEL.use_dropout)

    crit = nn.NLLLoss(ignore_index=255)

    print('###### Load data ######')
    data_name = cfg.DATASET.name
    if data_name == 'VOC':
        max_label = 20
    elif data_name == 'COCO':
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    labels_val = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    exclude_labels = labels_val

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = [
        transform.RandScale([0.9, 1.1]),
        transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([cfg.DATASET.input_size[0], cfg.DATASET.input_size[1]], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.SemData(split=cfg.TASK.fold_idx, shot=cfg.TASK.n_shots, data_root=cfg.DATASET.data_dir,
                                data_list=cfg.DATASET.train_list, transform=train_transform, mode='train', \
                                use_coco=False, use_split_coco=False)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.TRAIN.n_batch, shuffle=(train_sampler is None), num_workers=cfg.TRAIN.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_transform = transform.Compose([
        transform.Resize(size=cfg.DATASET.input_size[0]),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])    
     
    val_data = dataset.SemData(split=cfg.TASK.fold_idx, shot=cfg.TASK.n_shots, data_root=cfg.DATASET.data_dir, data_list=cfg.DATASET.val_list, transform=val_transform, mode='val', use_coco=False, use_split_coco=False)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.VAL.n_batch, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)


    #segmentation_module = nn.DataParallel(segmentation_module, device_ids=gpus)
    net_objectness.cuda()
    net_decoder.cuda()

    # Set up optimizers
    nets = (net_objectness, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    history = {'train': {'iter': [], 'loss': [], 'acc': []}}

    net_objectness.train(not cfg.TRAIN.fix_bn)
    net_decoder.train(not cfg.TRAIN.fix_bn)

    best_iou = 0
    # main loop
    tic = time.time()
    i_iter = 0
    print('###### Training ######')    
    for epoch in range(0, 200):
        for _, (input, target) in enumerate(train_loader):
            # Prepare input
            i_iter += 1
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            data_time.update(time.time() - tic)
            net_objectness.zero_grad()
            net_decoder.zero_grad()

            # adjust learning rate
            adjust_learning_rate(optimizers, i_iter, cfg)

            # forward pass
            feat = net_objectness(input, return_feature_maps=True)
            pred = net_decoder(feat, segSize=cfg.DATASET.input_size)
            loss = crit(pred, target)
            acc = pixel_acc(pred, target)
            loss = loss.mean()
            acc = acc.mean()

            # Backward
            loss.backward()
            for optimizer in optimizers:
                if optimizer:
                    optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - tic)
            tic = time.time()

            # update average loss and acc
            ave_total_loss.update(loss.data.item())
            ave_acc.update(acc.data.item()*100)

            # calculate accuracy, and display
            if i_iter % cfg.TRAIN.disp_iter == 0:
                print('Iter: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                      'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                      'Ave_Accuracy: {:4.2f}, Accuracy:{:4.2f}, Ave_Loss: {:.6f}, Loss: {:.6f}'
                      .format(i_iter, i_iter, cfg.TRAIN.n_iters,
                              batch_time.average(), data_time.average(),
                              cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                              ave_acc.average(), acc.data.item()*100, ave_total_loss.average(), loss.data.item()))

                history['train']['iter'].append(i_iter)
                history['train']['loss'].append(loss.data.item())
                history['train']['acc'].append(acc.data.item())

            if (i_iter+1) % cfg.TRAIN.save_freq == 0:
                checkpoint(nets, history, cfg, i_iter+1)

            if (i_iter+1) % cfg.TRAIN.eval_freq == 0:
                metric = Metric(max_label=max_label, n_runs=cfg.VAL.n_runs)
                with torch.no_grad():
                    print ('----Evaluation----')
                    net_objectness.eval()
                    net_decoder.eval()
                    net_decoder.use_softmax = True
                    #for run in range(cfg.VAL.n_runs):
                    for run in range(3):
                        print(f'### Run {run + 1} ###')
                        set_seed(cfg.VAL.seed + run)

                        print(f'### Load validation data ###')

                        #for sample_batched in tqdm.tqdm(testloader):
                        for (input, target, _) in val_loader:
                            feat = net_objectness(input, return_feature_maps=True)
                            query_pred = net_decoder(feat, segSize=cfg.DATASET.input_size)
                            metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                                          np.array(target[0].cpu()),
                                          labels=None, n_run=run)


                classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

                print('----- Evaluation Result -----')
                print(f'best meanIoU_binary: {best_iou}')
                print(f'meanIoU_binary mean: {meanIoU_binary}')
                print(f'meanIoU_binary std: {meanIoU_std_binary}')

                if meanIoU_binary > best_iou:
                    best_iou = meanIoU_binary
                    checkpoint(nets, history, cfg, 'best')
                net_objectness.train(not cfg.TRAIN.fix_bn)
                net_decoder.train(not cfg.TRAIN.fix_bn)
                net_decoder.use_softmax = False


    print('Training Done!')

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--memory_enc_pretrained",
        action='store_true',
        help="use a pretrained memory encoder",
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.memory_enc_pretrained = args.memory_enc_pretrained
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_from:
        cfg.MODEL.weights_enc_query = os.path.join(
            cfg.DIR, 'objectness_iter_{}.pth'.format(cfg.TRAIN.start_from))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'objectness_decoder_iter_{}.pth'.format(cfg.TRAIN.start_from))
        assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)

    cfg.TRAIN.max_iters = cfg.TRAIN.n_iters
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
