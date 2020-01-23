#!/bin/bash
python -u train.py --gpus 0, --cfg config/resnet50dilated-resnet18dilated-noBN-c1-attDouble-normKey-voc-fold0-ways1-shots1.yaml \
2>&1 | tee ./output/resnet50dilated-resnet18dilated-noBN-c1-attDouble-normKey-voc-fold0-ways1-shots1.log

