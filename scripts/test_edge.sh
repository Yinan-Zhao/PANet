#!/bin/bash
python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8-trainaug-testfold1.log

python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --eval_att_voting --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8-trainaug-testfold1-evalAttVoting.log

python test.py --gpu 0 --data_split trainaug --fold_idx 0 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8-trainaug-testfold0.log

python test.py --gpu 0 --data_split trainaug --fold_idx 2 --n_runs 3 --cfg config/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-globalPool-attDouble-c1-voc-fold1-ways1-shots1-lr2e3-b8-trainaug-testfold2.log


python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --cfg config/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0-trainaug-testfold1.log

python test.py --gpu 0 --data_split trainaug --fold_idx 1 --n_runs 3 --eval_att_voting --cfg config/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0-trainaug-testfold1-evalAttVoting.log

python test.py --gpu 0 --data_split trainaug --fold_idx 0 --n_runs 3 --cfg config/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0-trainaug-testfold0.log

python test.py --gpu 0 --data_split trainaug --fold_idx 2 --n_runs 3 --cfg config/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0.yaml --checkpoint iter_best.pth 2>&1 | tee output_test/resnet50deeplab-attDouble-c1-featAggre-noLabel-voc-fold1-ways1-shots1-pScalar1e0-trainaug-testfold2.log

