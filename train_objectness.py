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

#from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS
from utils_seg import AverageMeter, parse_devices, setup_logger
from models import ModelBuilder

import sys
sys.path.append('/home/yz9244/PANet/cocoapi/PythonAPI/')

def pixel_acc(pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

def data_preprocess(sample_batched, cfg, is_val=False):
    feed_dict = {}
    feed_dict['img_data'] = sample_batched['query_images'][0].cuda()
    if is_val:
        feed_dict['seg_label'] = sample_batched['query_labels'][0].cuda()
    else:
        tmp = sample_batched['query_labels'][0]
        tmp = torch.unsqueeze(tmp, 1).float()
        tmp = nn.functional.interpolate(tmp, size=(cfg.DATASET.input_size[0]//cfg.DATASET.segm_downsampling_rate,
            cfg.DATASET.input_size[1]//cfg.DATASET.segm_downsampling_rate), mode='nearest')
        feed_dict['seg_label'] = tmp[:,0,:,:].long().cuda() 

    n_ways = cfg.TASK.n_ways
    n_shots = cfg.TASK.n_shots
    n_batch = sample_batched['support_images'][0][0].shape[0]
    n_channel = sample_batched['support_images'][0][0].shape[1]
    height = sample_batched['support_images'][0][0].shape[2]
    width = sample_batched['support_images'][0][0].shape[3]
    feed_dict['img_refs_rgb'] = torch.zeros(n_batch, n_channel, n_ways*n_shots, height, width, dtype=sample_batched['support_images'][0][0].dtype)
    for i in range(n_ways):
        for j in range(n_shots):
            feed_dict['img_refs_rgb'][:,:,i*n_ways+j,:,:] = sample_batched['support_images'][i][j]
    feed_dict['img_refs_rgb'] = feed_dict['img_refs_rgb'].cuda()

    n_channel_mask = sample_batched['support_labels'][0][0].shape[1]
    feed_dict['img_refs_mask'] = torch.zeros(n_batch, n_channel_mask, n_ways*n_shots, height, width, dtype=sample_batched['support_labels'][0][0].dtype)
    for i in range(n_ways):
        for j in range(n_shots):
            feed_dict['img_refs_mask'][:,:,i*n_ways+j,:,:] = sample_batched['support_labels'][i][j]
    feed_dict['img_refs_mask'] = feed_dict['img_refs_mask'].cuda()

    return feed_dict

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
        from dataloaders.customized_objectness import voc_fewshot
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        from dataloaders.customized_objectness import coco_fewshot
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    labels_val = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    exclude_labels = labels_val
    transforms = Compose([Resize(size=cfg.DATASET.input_size),
                          RandomMirror()])
    dataset = make_data(
        base_dir=cfg.DATASET.data_dir,
        split=cfg.DATASET.data_split,
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=cfg.TRAIN.n_iters * cfg.TRAIN.n_batch,
        n_ways=cfg.TASK.n_ways,
        n_shots=cfg.TASK.n_shots,
        n_queries=cfg.TASK.n_queries,
        permute=cfg.TRAIN.permute_labels,
        exclude_labels=exclude_labels
    )
    trainloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.n_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

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

    print('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        feed_dict = data_preprocess(sample_batched, cfg)

        data_time.update(time.time() - tic)
        net_objectness.zero_grad()
        net_decoder.zero_grad()

        # adjust learning rate
        adjust_learning_rate(optimizers, i_iter, cfg)

        # forward pass
        feat = net_objectness(feed_dict['img_data'], return_feature_maps=True)
        pred = net_decoder(feat)
        loss = crit(pred, feed_dict['seg_label'])
        acc = pixel_acc(pred, feed_dict['seg_label'])
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
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(i_iter, i_iter, cfg.TRAIN.n_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

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
                for run in range(cfg.VAL.n_runs):
                    print(f'### Run {run + 1} ###')
                    set_seed(cfg.VAL.seed + run)

                    print(f'### Load validation data ###')
                    dataset_val = make_data(
                        base_dir=cfg.DATASET.data_dir,
                        split=cfg.DATASET.data_split,
                        transforms=transforms,
                        to_tensor=ToTensorNormalize(),
                        labels=labels_val,
                        max_iters=cfg.VAL.n_iters * cfg.VAL.n_batch,
                        n_ways=cfg.TASK.n_ways,
                        n_shots=cfg.TASK.n_shots,
                        n_queries=cfg.TASK.n_queries,
                        permute=cfg.VAL.permute_labels,
                        exclude_labels=[]
                    )
                    if data_name == 'COCO':
                        coco_cls_ids = dataset_val.datasets[0].dataset.coco.getCatIds()
                    testloader = DataLoader(dataset_val, batch_size=cfg.VAL.n_batch, shuffle=False,
                                            num_workers=1, pin_memory=True, drop_last=False)
                    print(f"Total # of validation Data: {len(dataset)}")

                    #for sample_batched in tqdm.tqdm(testloader):
                    for sample_batched in testloader:
                        feed_dict = data_preprocess(sample_batched, cfg, is_val=True)
                        if data_name == 'COCO':
                            label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                        else:
                            label_ids = list(sample_batched['class_ids'])

                        feat = net_objectness(feed_dict['img_data'], return_feature_maps=True)
                        query_pred = net_decoder(feat, segSize=cfg.DATASET.input_size)
                        metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                                      np.array(feed_dict['seg_label'][0].cpu()),
                                      labels=label_ids, n_run=run)

                    classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels_val), n_run=run)
                    classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels_val))
            classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

            print('----- Evaluation Result -----')
            print(f'best meanIoU_binary: {best_iou}')
            print(f'meanIoU mean: {meanIoU}')
            print(f'meanIoU std: {meanIoU_std}')
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
