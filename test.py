"""Evaluation Script"""
import os
import argparse
import shutil
from distutils.version import LooseVersion

import tqdm
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from config import cfg
from models import ModelBuilder, SegmentationAttentionSeparateModule
from utils_seg import AverageMeter, colorEncode, accuracy, intersectionAndUnion, parse_devices, setup_logger

#from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox

def data_preprocess(sample_batched, cfg):
    feed_dict = {}
    feed_dict['img_data'] = sample_batched['query_images'][0].cuda()
    feed_dict['seg_label'] = sample_batched['query_labels'][0].cuda()

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


def main(cfg, gpus):
    torch.cuda.set_device(gpus[0])

    # Network Builders
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
                segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)
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
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=255)

    segmentation_module = SegmentationAttentionSeparateModule(net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, crit, zero_memory=cfg.MODEL.zero_memory, zero_qval=cfg.MODEL.zero_qval, qval_qread_BN=cfg.MODEL.qval_qread_BN, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, debug=False, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate)

    segmentation_module = nn.DataParallel(segmentation_module, device_ids=gpus)
    segmentation_module.cuda()
    segmentation_module.eval()


    print('###### Prepare data ######')
    data_name = cfg.DATASET.name
    if data_name == 'VOC':
        from dataloaders.customized import voc_fewshot
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        from dataloaders.customized import coco_fewshot
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    transforms = [Resize(size=cfg.DATASET.input_size)]
    transforms = Compose(transforms)


    print('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=cfg.VAL.n_runs)
    with torch.no_grad():
        for run in range(cfg.VAL.n_runs):
            print(f'### Run {run + 1} ###')
            set_seed(cfg.VAL.seed + run)

            print(f'### Load data ###')
            dataset = make_data(
                base_dir=cfg.DATASET.data_dir,
                split=cfg.DATASET.data_split,
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=cfg.VAL.n_iters * cfg.VAL.n_batch,
                n_ways=cfg.TASK.n_ways,
                n_shots=cfg.TASK.n_shots,
                n_queries=cfg.TASK.n_queries,
                permute=cfg.VAL.permute_labels
            )
            if data_name == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=cfg.VAL.n_batch, shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            print(f"Total # of Data: {len(dataset)}")


            for sample_batched in tqdm.tqdm(testloader):
                feed_dict = data_preprocess(sample_batched, cfg)
                if data_name == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                
                query_pred = segmentation_module(feed_dict, segSize=cfg.DATASET.input_size)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(feed_dict['seg_label'].cpu()),
                              labels=label_ids, n_run=run)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            '''_run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')'''

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    print('----- Final Result -----')
    print('final_classIoU', classIoU.tolist())
    print('final_classIoU_std', classIoU_std.tolist())
    print('final_meanIoU', meanIoU.tolist())
    print('final_meanIoU_std', meanIoU_std.tolist())
    print('final_classIoU_binary', classIoU_binary.tolist())
    print('final_classIoU_std_binary', classIoU_std_binary.tolist())
    print('final_meanIoU_binary', meanIoU_binary.tolist())
    print('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    print(f'classIoU mean: {classIoU}')
    print(f'classIoU std: {classIoU_std}')
    print(f'meanIoU mean: {meanIoU}')
    print(f'meanIoU std: {meanIoU_std}')
    print(f'classIoU_binary mean: {classIoU_binary}')
    print(f'classIoU_binary std: {classIoU_std_binary}')
    print(f'meanIoU_binary mean: {meanIoU_binary}')
    print(f'meanIoU_binary std: {meanIoU_std_binary}')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
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
        "--debug_with_gt",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_double_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_double_complete_random",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_translated_gt",
        action='store_true',
        help="put gt in the memory",
    )
    parser.add_argument(
        "--debug_with_randomSegNoise",
        action='store_true',
        help="put gt in the memory",
    )


    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.DATASET.debug_with_gt = args.debug_with_gt
    cfg.DATASET.debug_with_random = args.debug_with_random
    cfg.DATASET.debug_with_translated_gt = args.debug_with_translated_gt
    cfg.DATASET.debug_with_double_random = args.debug_with_double_random
    cfg.DATASET.debug_with_double_complete_random = args.debug_with_double_complete_random
    cfg.DATASET.debug_with_randomSegNoise = args.debug_with_randomSegNoise
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_enc_query = os.path.join(
        cfg.DIR, 'enc_query_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_enc_memory = os.path.join(
        cfg.DIR, 'enc_memory_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_att_query = os.path.join(
        cfg.DIR, 'att_query_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_att_memory = os.path.join(
        cfg.DIR, 'att_memory_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_enc_memory) and \
            os.path.exists(cfg.MODEL.weights_att_query) and os.path.exists(cfg.MODEL.weights_att_memory) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
