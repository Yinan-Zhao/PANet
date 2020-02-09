#!/bin/bash
python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-trainaug-testfold1.log

python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --eval_att_voting --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-trainaug-testfold1-evalAttVoting.log

python test.py --gpu 0 --data_split trainaug --fold_idx 0 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-trainaug-testfold0.log

python test.py --gpu 0 --data_split trainaug --fold_idx 2 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-trainaug-testfold2.log


python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-trainaug-testfold1.log

python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --eval_att_voting --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-trainaug-testfold1-evalAttVoting.log

python test.py --gpu 0 --data_split trainaug --fold_idx 0 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-trainaug-testfold0.log

python test.py --gpu 0 --data_split trainaug --fold_idx 2 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-trainaug-testfold2.log

