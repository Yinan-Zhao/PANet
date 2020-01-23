"""Training Script"""
import os
import shutil

import random
import argparse

from config import cfg

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
from util.utils import set_seed, CLASS_LABELS
from utils_seg import AverageMeter, parse_devices, setup_logger
from models import ModelBuilder, SegmentationAttentionSeparateModule
from lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
import pdb


def data_preprocess(sample_batched, cfg):
    feed_dict = {}
    feed_dict['img_data'] = sample_batched['query_images'][0].cuda()
    tmp = sample_batched['query_labels'][0]
    tmp = torch.unsqueeze(tmp, 1)
    tmp = nn.functional.interpolate(tmp, size=(cfg.DATASET.input_size[0]//cfg.DATASET.segm_downsampling_rate,
        cfg.DATASET.input_size[1]//cfg.DATASET.segm_downsampling_rate), mode='nearest')
    feed_dict['seg_label'] = tmp[:,0,:,:].cuda() 

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
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets

    dict_enc_query = net_enc_query.state_dict()
    dict_enc_memory = net_enc_memory.state_dict()
    dict_att_query = net_att_query.state_dict()
    dict_att_memory = net_att_memory.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_enc_query,
        '{}/enc_query_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_enc_memory,
        '{}/enc_memory_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_att_query,
        '{}/att_query_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_att_memory,
        '{}/att_memory_iter_{}.pth'.format(cfg.DIR, iter_idx))
    torch.save(
        dict_decoder,
        '{}/decoder_iter_{}.pth'.format(cfg.DIR, iter_idx))

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
    (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit) = nets
    optimizer_enc_query = torch.optim.SGD(
        group_weight(net_enc_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_enc_memory = torch.optim.SGD(
        group_weight(net_enc_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_query = torch.optim.SGD(
        group_weight(net_att_query),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_att_memory = torch.optim.SGD(
        group_weight(net_att_memory),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder)

def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_enc_query, optimizer_enc_memory, optimizer_att_query, optimizer_att_memory, optimizer_decoder) = optimizers
    for param_group in optimizer_enc_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_enc_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_query.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_att_memory.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    print('###### Create model ######')
    net_enc_query = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_query)
    if cfg.MODEL.memory_encoder_arch:
        net_enc_memory = ModelBuilder.build_encoder_memory_separate(
            arch=cfg.MODEL.memory_encoder_arch.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            weights=cfg.MODEL.weights_enc_memory,
            num_class=cfg.TASK.n_ways+1,
            RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
            segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
    else:
        if cfg.MODEL.memory_encoder_noBN:
            net_enc_memory = ModelBuilder.build_encoder_memory_separate(
                arch=cfg.MODEL.arch_encoder.lower()+'_nobn',
                fc_dim=cfg.MODEL.fc_dim,
                weights=cfg.MODEL.weights_enc_memory,
                num_class=cfg.TASK.n_ways+1,
                RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
                segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
        else:
            net_enc_memory = ModelBuilder.build_encoder_memory_separate(
                arch=cfg.MODEL.arch_encoder.lower(),
                fc_dim=cfg.MODEL.fc_dim,
                weights=cfg.MODEL.weights_enc_memory,
                num_class=cfg.TASK.n_ways+1,
                RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
                segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate,
                pretrained=cfg.memory_enc_pretrained)
    net_att_query = ModelBuilder.build_att_query(
        arch=cfg.MODEL.arch_attention,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_att_query)
    net_att_memory = ModelBuilder.build_att_memory(
        arch=cfg.MODEL.arch_attention,
        fc_dim=cfg.MODEL.fc_dim,
        att_fc_dim=cfg.MODEL.att_fc_dim,
        weights=cfg.MODEL.weights_att_memory)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.TASK.n_ways+1,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=255)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationAttentionSeparateModule(
            net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, cfg.TRAIN.deep_sup_scale, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate)
    else:
        segmentation_module = SegmentationAttentionSeparateModule(
            net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, zero_memory=cfg.MODEL.zero_memory, random_memory_bias=cfg.MODEL.random_memory_bias, random_memory_nobias=cfg.MODEL.random_memory_nobias, random_scale=cfg.MODEL.random_scale, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate)


    print('###### Load data ######')
    data_name = cfg.DATASET.name
    if data_name == 'VOC':
        from dataloaders.customized import voc_fewshot
        make_data = voc_fewshot
    elif data_name == 'COCO':
        from dataloaders.customized import coco_fewshot
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][cfg.TASK.fold_idx]
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
        n_queries=cfg.TASK.n_queries
    )
    trainloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.n_batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    #segmentation_module = UserScatteredDataParallel(
        #segmentation_module,
        #device_ids=gpus)
    # For sync bn
    #patch_replication_callback(segmentation_module)
    segmentation_module = nn.DataParallel(segmentation_module, device_ids=gpus)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    history = {'train': {'iter': [], 'loss': [], 'acc': []}}

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()

    print('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        feed_dict = data_preprocess(sample_batched, cfg)

        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        adjust_learning_rate(optimizers, i_iter, cfg)

        # forward pass
        #print(batch_data)
        loss, acc = segmentation_module(feed_dict)
        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
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
    if cfg.TRAIN.start_iter > 0:
        cfg.MODEL.weights_enc_query = os.path.join(
            cfg.DIR, 'enc_query_iter_{}.pth'.format(cfg.TRAIN.start_iter))
        cfg.MODEL.weights_enc_memory = os.path.join(
            cfg.DIR, 'enc_memory_iter_{}.pth'.format(cfg.TRAIN.start_iter))
        cfg.MODEL.weights_att_query = os.path.join(
            cfg.DIR, 'att_query_iter_{}.pth'.format(cfg.TRAIN.start_iter))
        cfg.MODEL.weights_att_memory = os.path.join(
            cfg.DIR, 'att_memory_iter_{}.pth'.format(cfg.TRAIN.start_iter))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_iter_{}.pth'.format(cfg.TRAIN.start_iter))
        assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_enc_memory) and \
            os.path.exists(cfg.MODEL.weights_att_query) and os.path.exists(cfg.MODEL.weights_att_memory) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    #print('gpus')
    #print(gpus)
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.n_iters
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
