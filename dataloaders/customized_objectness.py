"""
Customized dataset
"""

import os
import random

import torch
import numpy as np

from .pascal import VOC
from .coco import COCOSeg
from .common import PairedDataset


def segm_one_hot(segm, n_ways):
    size = segm.size()
    assert len(size) == 2
    segm = segm.unsqueeze(0)
    oneHot_size = (n_ways+2, size[0], size[1])
    segm_oneHot = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    segm_oneHot = segm_oneHot.scatter_(0, segm, 1.0)
    return segm_oneHot


def attrib_basic(_sample, class_id):
    """
    Add basic attribute

    Args:
        _sample: data sample
        class_id: class label asscociated with the data
            (sometimes indicting from which subset the data are drawn)
    """
    return {'class_id': class_id}


def getMask(label, class_id, class_ids):
    """
    Generate FG/BG mask from the segmentation mask

    Args:
        label:
            semantic mask
        scribble:
            scribble mask
        class_id:
            semantic class of interest
        class_ids:
            all class id in this episode
    """
    # Dense Mask
    fg_mask = torch.where(label == class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    bg_mask = torch.where(label != class_id,
                          torch.ones_like(label), torch.zeros_like(label))
    for class_id in class_ids:
        bg_mask[label == class_id] = 0

    return {'fg_mask': fg_mask,
            'bg_mask': bg_mask}


def fewShot(paired_sample, n_ways, n_shots, cnt_query, coco=False, permute=False):
    """
    Postprocess paired sample for fewshot settings

    Args:
        paired_sample:
            data sample from a PairedDataset
        n_ways:
            n-way few-shot learning
        n_shots:
            n-shot few-shot learning
        cnt_query:
            number of query images for each class in the support set
        coco:
            MS COCO dataset
    """
    if permute:
        perm_mapping = np.random.permutation(n_ways+1)
    else:
        perm_mapping = np.array(range(n_ways+1))
    ###### Compose the support and query image list ######
    cumsum_idx = np.cumsum([0,] + [n_shots + x for x in cnt_query])

    # support class ids
    class_ids = [paired_sample[cumsum_idx[i]]['basic_class_id'] for i in range(n_ways)]

    # support images
    support_ids = [[paired_sample[cumsum_idx[i] + j]['id'] for j in range(n_shots)]
                      for i in range(n_ways)]
    support_images = [[paired_sample[cumsum_idx[i] + j]['image'] for j in range(n_shots)]
                      for i in range(n_ways)] # [way][shot]
    support_images_t = [[paired_sample[cumsum_idx[i] + j]['image_t'] for j in range(n_shots)]
                        for i in range(n_ways)]

    # support image labels
    if coco:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'][class_ids[i]]
                           for j in range(n_shots)] for i in range(n_ways)]
    else:
        support_labels = [[paired_sample[cumsum_idx[i] + j]['label'] for j in range(n_shots)]
                          for i in range(n_ways)]


    # query images, masks and class indices
    query_ids = [paired_sample[cumsum_idx[i+1] - j - 1]['id'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images = [paired_sample[cumsum_idx[i+1] - j - 1]['image'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    query_images_t = [paired_sample[cumsum_idx[i+1] - j - 1]['image_t'] for i in range(n_ways)
                      for j in range(cnt_query[i])]
    query_images_noresize = [paired_sample[cumsum_idx[i+1] - j - 1]['image_noresize'] for i in range(n_ways)
                    for j in range(cnt_query[i])]
    if coco:
        #query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'][class_ids[i]]
                        #for i in range(n_ways) for j in range(cnt_query[i])]
        query_labels = []
        for i in range(n_ways):
            for j in range(cnt_query[i]):
            tmp = 0
            for k in paired_sample[cumsum_idx[i+1] - j - 1]['label'].keys():
                tmp += paired_sample[cumsum_idx[i+1] - j - 1]['label'][k]
            query_labels.append(tmp)
    else:
        query_labels = [paired_sample[cumsum_idx[i+1] - j - 1]['label'] for i in range(n_ways)
                        for j in range(cnt_query[i])]

    if coco:
        #query_labels_noresize = [paired_sample[cumsum_idx[i+1] - j - 1]['label_noresize'][class_ids[i]]
                        #for i in range(n_ways) for j in range(cnt_query[i])]
        query_labels_noresize = []
        for i in range(n_ways):
            for j in range(cnt_query[i]):
            tmp = 0
            for k in paired_sample[cumsum_idx[i+1] - j - 1]['label_noresize'].keys():
                tmp += paired_sample[cumsum_idx[i+1] - j - 1]['label_noresize'][k]
            query_labels_noresize.append(tmp)
    else:
        query_labels_noresize = [paired_sample[cumsum_idx[i+1] - j - 1]['label_noresize'] for i in range(n_ways)
                        for j in range(cnt_query[i])]

    query_cls_idx = [sorted([0,] + [class_ids.index(x) + 1
                                    for x in set(np.unique(query_label)) & set(class_ids)])
                     for query_label in query_labels]


    ###### Generate support image masks ######
    support_mask = [[getMask(support_labels[way][shot],
                             class_ids[way], class_ids)
                     for shot in range(n_shots)] for way in range(n_ways)]


    ###### Generate query label (class indices in one episode, i.e. the ground truth)######
    query_labels_tmp = [torch.zeros_like(x) for x in query_labels]
    for i, query_label_tmp in enumerate(query_labels_tmp):
        query_label_tmp[query_labels[i] != 0] = 1

    query_labels_noresize_tmp = [torch.zeros_like(x) for x in query_labels_noresize]
    for i, query_label_noresize_tmp in enumerate(query_labels_noresize_tmp):
        query_label_noresize_tmp[query_labels_noresize[i] != 0] = 1
        
    support_labels_tmp = [[torch.zeros_like(support_labels[way][shot]) 
                        for shot in range(n_shots)] for way in range(n_ways)]
    support_labels_return = []
    for way in range(n_ways):
        tmp = []
        for shot in range(n_shots):
            support_labels_tmp[way][shot][support_labels[way][shot] != 0] = 1
            tmp.append(segm_one_hot(support_labels_tmp[way][shot], n_ways))
        support_labels_return.append(tmp)

    ###### Generate query mask for each semantic class (including BG) ######
    # BG class
    query_masks = [[torch.where(query_label == 0,
                                torch.ones_like(query_label),
                                torch.zeros_like(query_label))[None, ...],]
                   for query_label in query_labels]
    # Other classes in query image
    for i, query_label in enumerate(query_labels):
        for idx in query_cls_idx[i][1:]:
            mask = torch.where(query_label == class_ids[idx - 1],
                               torch.ones_like(query_label),
                               torch.zeros_like(query_label))[None, ...]
            query_masks[i].append(mask)


    return {'class_ids': class_ids,

            'support_ids': support_ids,
            'support_images_t': support_images_t,
            'support_images': support_images,
            'support_labels': support_labels_return,
            'support_mask': support_mask,

            'query_ids': query_ids,
            'query_images_t': query_images_t,
            'query_images': query_images,
            'query_images_noresize': query_images_noresize,
            'query_labels': query_labels_tmp,
            'query_labels_noresize': query_labels_noresize_tmp,
            'query_masks': query_masks,
            'query_cls_idx': query_cls_idx,
           }


def voc_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                n_queries=1, permute=False, exclude_labels=[]):
    """
    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            object class labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of object class labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    voc = VOC(base_dir=base_dir, split=split, transforms=transforms, to_tensor=to_tensor)
    voc.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    sub_ids = []
    for label in labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            sub_ids.append(f.read().splitlines())

    total_count = 0
    for sub_list in sub_ids:
        total_count += len(sub_list)
    print('the number of training images before excluding: %d' % (total_count))

    exclude_sub_ids = []
    for label in exclude_labels:
        with open(os.path.join(voc._id_dir, voc.split,
                               'class{}.txt'.format(label)), 'r') as f:
            exclude_sub_ids += f.read().splitlines()

    for sub_ids_item in sub_ids:
        for id_item in sub_ids_item:
            if id_item in exclude_sub_ids:
                sub_ids_item.remove(id_item)

    after_count = 0
    for sub_list in sub_ids:
        after_count += len(sub_list)
    print('the number of training images after excluding: %d' % (after_count))
    
    # Create sub-datasets and add class_id attribute
    subsets = voc.subsets(sub_ids, [{'basic': {'class_id': cls_id}} for cls_id in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries), minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'permute': permute})])
    return paired_data


def coco_fewshot(base_dir, split, transforms, to_tensor, labels, n_ways, n_shots, max_iters,
                 n_queries=1, permute=False, exclude_labels=[]):
    """
    Args:
        base_dir:
            COCO dataset directory
        split:
            which split to use
            choose from ('train', 'val')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
        labels:
            labels of the data
        n_ways:
            n-way few-shot learning, should be no more than # of labels
        n_shots:
            n-shot few-shot learning
        max_iters:
            number of pairs
        n_queries:
            number of query images
    """
    cocoseg = COCOSeg(base_dir, split, transforms, to_tensor)
    cocoseg.add_attrib('basic', attrib_basic, {})

    # Load image ids for each class
    cat_ids = cocoseg.coco.getCatIds()
    sub_ids = [cocoseg.coco.getImgIds(catIds=cat_ids[i - 1]) for i in labels]

    total_count = 0
    for sub_list in sub_ids:
        total_count += len(sub_list)
    print('the number of training images before excluding: %d' % (total_count))

    exclude_sub_ids = []
    for label in exclude_labels:
        exclude_sub_ids += cocoseg.coco.getImgIds(catIds=cat_ids[label - 1])

    print('length of list exclude_sub_ids %d' % (len(exclude_sub_ids)))
    exclude_sub_ids = set(exclude_sub_ids)
    print('length of set exclude_sub_ids %d' % (len(exclude_sub_ids)))
    #print(exclude_sub_ids)

    for sub_ids_item in sub_ids:
        for id_item in sub_ids_item:
            if id_item in exclude_sub_ids:
                sub_ids_item.remove(id_item)

    after_count = 0
    for sub_list in sub_ids:
        after_count += len(sub_list)
    print('the number of training images after excluding: %d' % (after_count))

    # Create sub-datasets and add class_id attribute
    subsets = cocoseg.subsets(sub_ids, [{'basic': {'class_id': cat_ids[i - 1]}} for i in labels])

    # Choose the classes of queries
    cnt_query = np.bincount(random.choices(population=range(n_ways), k=n_queries),
                            minlength=n_ways)
    # Set the number of images for each class
    n_elements = [n_shots + x for x in cnt_query]
    # Create paired dataset
    paired_data = PairedDataset(subsets, n_elements=n_elements, max_iters=max_iters, same=False,
                                pair_based_transforms=[
                                    (fewShot, {'n_ways': n_ways, 'n_shots': n_shots,
                                               'cnt_query': cnt_query, 'coco': True, 'permute': permute})])
    return paired_data
    
