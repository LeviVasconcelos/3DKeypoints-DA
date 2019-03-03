#!/bin/bash

#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/epochs=60_loss=FROB__normalizer=WEIGHTED-SQR-ROOT__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='frobenius' -weightedNorm

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/criterion=PROPS__=LOG-LIKELIHOOD__epochs=60_bn=TARGET__normalizer=NONE__eps=$2__mask=DIAG*DIAGT__dataset=$1_bs=64_lr=0.0001 -nViews=1 -eps=$2 -lossNorm='likelihood'

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.001 -logDir ../logs/likelihood/criterion=PROPS__=LOG-LIKELIHOOD__epochs=60_bn=TARGET__normalizer=NONE__eps=$2__mask=DIAG*DIAGT__dataset=$1_bs=64_lr=0.001 -nViews=1 -eps=$2 -lossNorm='likelihood'

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.01 -logDir ../logs/likelihood/criterion=PROPS__=LOG-LIKELIHOOD__epochs=60_bn=TARGET__normalizer=NONE__eps=$2__mask=DIAG*DIAGT__dataset=$1_bs=64_lr=0.01 -nViews=1 -eps=$2 -lossNorm='likelihood'

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.00001 -logDir ../logs/likelihood/criterion=PROPS__=LOG-LIKELIHOOD__epochs=60_bn=TARGET__normalizer=NONE__eps=$2__mask=DIAG*DIAGT__dataset=$1_bs=64_lr=0.00001 -nViews=1 -eps=$2 -lossNorm='likelihood'

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.000001 -logDir ../logs/likelihood/criterion=PROPS__=LOG-LIKELIHOOD__epochs=60_bn=TARGET__normalizer=NONE__eps=$2__mask=DIAG*DIAGT__dataset=$1_bs=64_lr=0.000001 -nViews=1 -eps=$2 -lossNorm='likelihood'



#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/criterion=likelihood__epochs=60_bn=TARGET_loss=L1__normalizer=WEIGHTED__eps=$2__mask=NONE__dataset=$1 -nViews=1 -eps=$2 -lossNorm='l1' -weightedNorm

#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/criterion=likelihood__epochs=60_bn=TARGET_loss=FROB__normalizer=WEIGHTED__eps=$2__mask=NONE__dataset=$1 -nViews=1 -eps=$2 -lossNorm='frobenius' -weightedNorm



#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/epochs=60_loss=L1__normalizer=WEIGHTED__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='l1' -weightedNorm

#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/epochs=60_loss=L1__normalizer=AVG__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='l1'


#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/epochs=60_loss=L2__normalizer=WEIGHTED__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='l2' -weightedNorm

#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/likelihood/epochs=60_loss=L2__normalizer=AVG__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='l2'

