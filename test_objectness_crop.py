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
from PIL import Image
from scipy.misc import imread, imresize

from config import cfg
from models import ModelBuilder
from utils_seg import AverageMeter, colorEncodeGray, accuracy, intersectionAndUnion, parse_devices, setup_logger

#from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders import transforms
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS, get_bbox
from lib.utils import as_numpy

import sys
sys.path.append('/home/yz9244/PANet/cocoapi/PythonAPI/')
from pycocotools.coco import COCO


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncodeGray(seg)

    # prediction
    pred_color = colorEncodeGray(pred)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name+'.png'))

def data_preprocess(sample_batched, cfg):
    feed_dict = {}
    feed_dict['img_data'] = sample_batched['query_images'][0].float().cuda()
    feed_dict['img_data_noresize'] = sample_batched['query_images_noresize'][0].cuda()
    feed_dict['seg_label'] = sample_batched['query_labels'][0].long().cuda()
    feed_dict['seg_label_noresize'] = sample_batched['query_labels_noresize'][0].cuda()

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
        use_dropout=cfg.MODEL.use_dropout,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=255)

    net_objectness.cuda()
    net_objectness.eval()

    net_decoder.cuda()
    net_decoder.eval()

    print('###### Prepare data ######')
    data_name = cfg.DATASET.name
    if data_name == 'VOC':
        if cfg.VAL.test_with_classes:
            from dataloaders.customized import voc_fewshot
        else:
            from dataloaders.customized_objectness import voc_fewshot
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        if cfg.VAL.test_with_classes:
            from dataloaders.customized import coco_fewshot
        else:
            from dataloaders.customized_objectness import coco_fewshot
        make_data = coco_fewshot
        max_label = 80
        split = cfg.DATASET.data_split + '2014'
        annFile = f'{cfg.DATASET.data_dir}/annotations/instances_{split}.json'
        cocoapi = COCO(annFile)
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    #labels = CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    #transforms = [Resize_test(size=cfg.DATASET.input_size)]
    val_transforms = [transforms.ToNumpy(),
        transforms.Resize_pad(size=cfg.DATASET.input_size[0])]
    val_transforms = Compose(val_transforms)


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
                transforms=val_transforms,
                to_tensor=transforms.ToTensorNormalize_noresize(),
                labels=labels,
                max_iters=cfg.VAL.n_iters * cfg.VAL.n_batch,
                n_ways=cfg.TASK.n_ways,
                n_shots=cfg.TASK.n_shots,
                n_queries=cfg.TASK.n_queries,
                permute=cfg.VAL.permute_labels,
            )
            if data_name == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=cfg.VAL.n_batch, shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            print(f"Total # of Data: {len(dataset)}")

            count = 0

            for sample_batched in tqdm.tqdm(testloader):
                feed_dict = data_preprocess(sample_batched, cfg)
                if data_name == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])

                feat = net_objectness(feed_dict['img_data'], return_feature_maps=True)
                query_pred = net_decoder(feat, segSize=(473,473))

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(feed_dict['seg_label'][0].cpu()),
                              labels=label_ids, n_run=run)

                if cfg.VAL.visualize:
                    #print(as_numpy(feed_dict['seg_label'][0].cpu()).shape)
                    #print(as_numpy(np.array(query_pred.argmax(dim=1)[0].cpu())).shape)
                    #print(feed_dict['img_data'].cpu().shape)
                    query_name = sample_batched['query_ids'][0][0]
                    support_name = sample_batched['support_ids'][0][0][0]
                    if data_name == 'VOC':
                        img = imread(os.path.join(cfg.DATASET.data_dir, 'JPEGImages', query_name+'.jpg'))
                    else:
                        query_name = int(query_name)
                        img_meta = cocoapi.loadImgs(query_name)[0]
                        img = imread(os.path.join(cfg.DATASET.data_dir, split, img_meta['file_name']))
                    #img = imresize(img, cfg.DATASET.input_size)
                    visualize_result(
                        (img, as_numpy(feed_dict['seg_label_noresize'][0].cpu()), '%05d'%(count)),
                        as_numpy(np.array(query_pred.argmax(dim=1)[0].cpu())),
                        os.path.join(cfg.DIR, 'result')
                    )
                count += 1

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
    parser.add_argument(
        "--eval_att_voting",
        action='store_true',
        help="evaluate with attention-based voting",
    )
    parser.add_argument(
        "--is_debug",
        action='store_true',
        help="store intermediate results, such as probability",
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="visualize results",
    )
    parser.add_argument(
        "--eval_from_scratch",
        action='store_true',
        help="evaluate from scratch",
    )
    parser.add_argument(
        "--n_runs",
        help="number of runs in evaluation",
        default=3,
        type=int
    )
    parser.add_argument(
        "--fold_idx",
        help="fold index",
        default=-1,
        type=int
    )
    parser.add_argument(
        "--checkpoint",
        default='iter_10000.pth',
        help="which checkpoint to evaluate",
        type=str,
    )
    parser.add_argument(
        "--data_split",
        default='trainaug',
        help="data split",
        type=str,
    )
    parser.add_argument(
        "--test_with_classes",
        action='store_true',
        help="evaluate from scratch",
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
    cfg.is_debug = args.is_debug
    cfg.eval_att_voting = args.eval_att_voting
    cfg.DATASET.data_split = args.data_split
    cfg.VAL.visualize = args.visualize
    cfg.VAL.n_runs = args.n_runs
    cfg.VAL.checkpoint = args.checkpoint
    cfg.VAL.test_with_classes = args.test_with_classes
    if args.fold_idx >= 0:
        cfg.TASK.fold_idx = args.fold_idx
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    if not args.eval_from_scratch:
        cfg.MODEL.weights_enc_query = os.path.join(
            cfg.DIR, 'objectness_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'objectness_decoder_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_enc_query) and os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
