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
from models import ModelBuilder, SegmentationAttentionSeparateModule
from utils_seg import AverageMeter, colorEncodeGray, accuracy, intersectionAndUnion, parse_devices, setup_logger

#from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, Resize_test
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
    feed_dict['img_data'] = sample_batched['query_images'][0].cuda()
    feed_dict['img_data_noresize'] = sample_batched['query_images_noresize'][0].cuda()
    feed_dict['seg_label'] = sample_batched['query_labels'][0].cuda()
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
    net_enc_query = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_query,
        fix_encoder=cfg.TRAIN.fix_encoder)
    net_enc_memory = ModelBuilder.build_encoder_memory_separate(
        arch=cfg.MODEL.arch_memory_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_enc_memory,
        num_class=cfg.TASK.n_ways+1,
        RGB_mask_combine_val=cfg.DATASET.RGB_mask_combine_val,
        segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate)    
    net_att_query = ModelBuilder.build_attention(
        arch=cfg.MODEL.arch_attention,
        input_dim=cfg.MODEL.encoder_dim,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_att_query)
    net_att_memory = ModelBuilder.build_attention(
        arch=cfg.MODEL.arch_attention,
        input_dim=cfg.MODEL.fc_dim,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_att_memory)
    net_projection = ModelBuilder.build_projection(
        arch=cfg.MODEL.arch_projection,
        input_dim=cfg.MODEL.encoder_dim,
        fc_dim=cfg.MODEL.projection_dim,
        weights=cfg.MODEL.weights_projection)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        input_dim=cfg.MODEL.decoder_dim,
        fc_dim=cfg.MODEL.decoder_fc_dim,
        ppm_dim=cfg.MODEL.ppm_dim,
        num_class=cfg.TASK.n_ways+1,
        weights=cfg.MODEL.weights_decoder,
        dropout_rate=cfg.MODEL.dropout_rate,
        use_dropout=cfg.MODEL.use_dropout,
        use_softmax=True)
    if cfg.MODEL.weights_objectness and cfg.MODEL.weights_objectness_decoder:
        '''net_objectness = ModelBuilder.build_objectness(
            arch='resnet50_deeplab',
            weights=cfg.MODEL.weights_objectness,
            fix_encoder=True)
        net_objectness_decoder = ModelBuilder.build_decoder(
            arch='aspp_few_shot',
            input_dim=2048,
            fc_dim=256,
            ppm_dim=256,
            num_class=2,
            weights=cfg.MODEL.weights_objectness_decoder,
            dropout_rate=0.5,
            use_dropout=True)'''
        net_objectness = ModelBuilder.build_objectness(
            arch='hrnetv2',
            weights=cfg.MODEL.weights_objectness,
            fix_encoder=True)
        net_objectness_decoder = ModelBuilder.build_decoder(
            arch='c1_nodropout',
            input_dim=720,
            fc_dim=720,
            ppm_dim=256,
            num_class=2,
            weights=cfg.MODEL.weights_objectness_decoder,
            use_dropout=False)
        for param in net_objectness.parameters():
            param.requires_grad = False
        for param in net_objectness_decoder.parameters():
            param.requires_grad = False
    else:
        net_objectness = None
        net_objectness_decoder = None

    crit = nn.NLLLoss(ignore_index=255)

    segmentation_module = SegmentationAttentionSeparateModule(net_enc_query, net_enc_memory, net_att_query, net_att_memory, net_decoder, net_projection, net_objectness, net_objectness_decoder, crit, zero_memory=cfg.MODEL.zero_memory, zero_qval=cfg.MODEL.zero_qval, normalize_key=cfg.MODEL.normalize_key, p_scalar=cfg.MODEL.p_scalar, memory_feature_aggregation=cfg.MODEL.memory_feature_aggregation, memory_noLabel=cfg.MODEL.memory_noLabel, debug=cfg.is_debug or cfg.eval_att_voting, mask_feat_downsample_rate=cfg.MODEL.mask_feat_downsample_rate, att_mat_downsample_rate=cfg.MODEL.att_mat_downsample_rate, objectness_feat_downsample_rate=cfg.MODEL.objectness_feat_downsample_rate, segm_downsampling_rate=cfg.DATASET.segm_downsampling_rate, mask_foreground=cfg.MODEL.mask_foreground, global_pool_read=cfg.MODEL.global_pool_read, average_memory_voting=cfg.MODEL.average_memory_voting, average_memory_voting_nonorm=cfg.MODEL.average_memory_voting_nonorm, mask_memory_RGB=cfg.MODEL.mask_memory_RGB, linear_classifier_support=cfg.MODEL.linear_classifier_support, decay_lamb=cfg.MODEL.decay_lamb, linear_classifier_support_only=cfg.MODEL.linear_classifier_support_only, qread_only=cfg.MODEL.qread_only, feature_as_key=cfg.MODEL.feature_as_key, objectness_multiply=cfg.MODEL.objectness_multiply)

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
        split = cfg.DATASET.data_split + '2014'
        annFile = f'{cfg.DATASET.data_dir}/annotations/instances_{split}.json'
        cocoapi = COCO(annFile)
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][cfg.TASK.fold_idx]
    transforms = [Resize_test(size=cfg.DATASET.input_size)]
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
                permute=cfg.VAL.permute_labels,
                exclude_labels=[]
            )
            if data_name == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()
            testloader = DataLoader(dataset, batch_size=cfg.VAL.n_batch, shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            print(f"Total # of Data: {len(dataset)}")

            count = 0

            if cfg.multi_scale_test:
                scales = [224, 328, 424]
            else:
                scales = [328]

            for sample_batched in tqdm.tqdm(testloader):
                feed_dict = data_preprocess(sample_batched, cfg)
                if data_name == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])

                for q, scale in enumerate(scales):
                    if len(scales) > 1:
                        feed_dict['img_data'] = nn.functional.interpolate(feed_dict['img_data'].cuda(), size=(scale, scale), mode='bilinear')
                    if cfg.eval_att_voting or cfg.is_debug:
                        query_pred, qread, qval, qk_b, mk_b, mv_b, p, feature_enc, feature_memory = segmentation_module(feed_dict, segSize=cfg.DATASET.input_size)
                        if cfg.eval_att_voting:
                            height, width = qread.shape[-2], qread.shape[-1]
                            assert p.shape[0] == height*width
                            img_refs_mask_resize = nn.functional.interpolate(feed_dict['img_refs_mask'][0].cuda(), size=(height, width), mode='nearest')
                            img_refs_mask_resize_flat = img_refs_mask_resize[:,0,:,:].view(img_refs_mask_resize.shape[0], -1)
                            mask_voting_flat = torch.mm(img_refs_mask_resize_flat, p)
                            mask_voting = mask_voting_flat.view(mask_voting_flat.shape[0], height, width)
                            mask_voting = torch.unsqueeze(mask_voting, 0)
                            query_pred = nn.functional.interpolate(mask_voting[:,0:-1], size=cfg.DATASET.input_size, mode='bilinear', align_corners=False)
                            if cfg.is_debug:
                                np.save('debug/img_refs_mask-%04d-%s-%s.npy'%(count, sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), img_refs_mask_resize.detach().cpu().float().numpy())
                                np.save('debug/query_pred-%04d-%s-%s.npy'%(count, sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), query_pred.detach().cpu().float().numpy())
                        if cfg.is_debug:
                            np.save('debug/qread-%04d-%s-%s.npy'%(count, sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), qread.detach().cpu().float().numpy())
                            np.save('debug/qval-%04d-%s-%s.npy'%(count, sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), qval.detach().cpu().float().numpy())
                            #np.save('debug/qk_b-%s-%s.npy'%(sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), qk_b.detach().cpu().float().numpy())
                            #np.save('debug/mk_b-%s-%s.npy'%(sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), mk_b.detach().cpu().float().numpy())
                            #np.save('debug/mv_b-%s-%s.npy'%(sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), mv_b.detach().cpu().float().numpy())
                            np.save('debug/p-%04d-%s-%s.npy'%(count, sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), p.detach().cpu().float().numpy())
                            #np.save('debug/feature_enc-%s-%s.npy'%(sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), feature_enc[-1].detach().cpu().float().numpy())
                            #np.save('debug/feature_memory-%s-%s.npy'%(sample_batched['query_ids'][0][0], sample_batched['support_ids'][0][0][0]), feature_memory[-1].detach().cpu().float().numpy())
                    else:
                        #query_pred = segmentation_module(feed_dict, segSize=cfg.DATASET.input_size)
                        query_pred = segmentation_module(feed_dict, segSize=(feed_dict['seg_label_noresize'].shape[1], feed_dict['seg_label_noresize'].shape[2]))
                        if q == 0:
                            query_pred_final = query_pred/len(scales)
                        else:
                            query_pred_final += query_pred/len(scales)
                query_pred = query_pred_final
                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(feed_dict['seg_label_noresize'][0].cpu()),
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
        "--multi_scale_test",
        action='store_true',
        help="use multi-scale testing",
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
    cfg.multi_scale_test = args.multi_scale_test
    cfg.DATASET.data_split = args.data_split
    cfg.VAL.visualize = args.visualize
    cfg.VAL.n_runs = args.n_runs
    cfg.VAL.checkpoint = args.checkpoint
    if args.fold_idx >= 0:
        cfg.TASK.fold_idx = args.fold_idx
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    if not args.eval_from_scratch:
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
        if cfg.MODEL.arch_projection:
            cfg.MODEL.weights_projection = os.path.join(
                cfg.DIR, 'projection_' + cfg.VAL.checkpoint)
            assert os.path.exists(cfg.MODEL.weights_projection)

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)
