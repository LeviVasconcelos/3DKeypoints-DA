#!/bin/bash

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=FROB__normalizer=WEIGHTED__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='frobenius' -weightedNorm

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=FROB__normalizer=AVG__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='frobenius'


python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=L1__normalizer=WEIGHTED__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='l1' -weightedNorm

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=L1__normalizer=AVG__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='l1'


python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=L2__normalizer=WEIGHTED__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='l2' -weightedNorm

python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset Redwood -LR=0.0001 -logDir ../logs/loss=L2__normalizer=AVG__mask=DIAG*DIAGT__dataset=REDWOOD -nViews=1 -lossNorm='l2'

