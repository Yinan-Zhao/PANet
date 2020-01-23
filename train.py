"""Training Script"""
import os
import shutil

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

#from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex
import pdb

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
    for i in n_ways:
        for j in n_shots:
            feed_dict['img_refs_rgb'][:,:,i*n_ways+j,:,:] = sample_batched['support_images'][i][j]
    feed_dict['img_refs_rgb'] = feed_dict['img_refs_rgb'].cuda()

    n_channel_mask = sample_batched['support_labels'][0][0].shape[1]
    feed_dict['img_refs_mask'] = torch.zeros(n_batch, n_channel_mask, n_ways*n_shots, height, width, dtype=sample_batched['support_labels'][0][0].dtype)
    for i in n_ways:
        for j in n_shots:
            feed_dict['img_refs_mask'][:,:,i*n_ways+j,:,:] = sample_batched['support_labels'][i][j]
    feed_dict['img_refs_mask'] = feed_dict['img_refs_mask'].cuda()

    return feed_dict


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        from dataloaders.customized import voc_fewshot
        make_data = voc_fewshot
    elif data_name == 'COCO':
        from dataloaders.customized import coco_fewshot
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        #n_ways=_config['task']['n_ways'],
        n_ways=2,
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    trainloader = DataLoader(
        dataset,
        #batch_size=_config['batch_size'],
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')

    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    for i_iter, sample_batched in enumerate(trainloader):
        # Prepare input
        feed_dict = data_preprocess(sample_batched)

        # Forward and Backward
        pdb.set_trace()
