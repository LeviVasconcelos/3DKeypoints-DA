#!/bin/bash

#python main_vis.py -expID exp_with_priors -loadModel ../models/ModelNet120.tar -propsFile ../priors/RND_distances.npy -targetDataset $1 -LR=0.0001 -logDir ../logs/upper-bound/epochs=60_loss=FROB__normalizer=WEIGHTED-SQR-ROOT__mask=DIAG*DIAGT__dataset=$1 -nViews=1 -lossNorm='frobenius' -weightedNorm

python main_vis.py -expID gaussian_weightd_masked_120_e20_$1-$5 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=$5 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e20_$1-$5 -nViews=1 -eps=$3 -lossNorm='l2' -runs=10 -epochs=20 -dropLR=10

python main_vis.py -expID gaussian_weightd_masked_120_e20_$2-$5 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $2 -LR=$5 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e20_$2-$5 -nViews=1 -eps=$3 -lossNorm='l2' -runs=10 -epochs=20 -dropLR=10




python main_vis.py -expID gaussian_weightd_masked_120_e30_$1-$5 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=0.00005 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e30_$1-00005 -nViews=1 -eps=$3-$5 -lossNorm='l2' -runs=10 -epochs=30 -dropLR=20

python main_vis.py -expID gaussian_weightd_masked_120_e30_$2-$5 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $2 -LR=0.00005 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e30_$2-00005 -nViews=1 -eps=$3-$5 -lossNorm='l2' -runs=10 -epochs=30 -dropLR=20


python main_vis.py -expID gaussian_weightd_masked_120_e30_$1-00005 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=$5 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e30_$1-$5 -nViews=1 -eps=$3-00005 -lossNorm='l2' -runs=10 -epochs=30 -dropLR=20

python main_vis.py -expID gaussian_weightd_masked_120_e30_00005 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $2 -LR=$5 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e30_$2-$5 -nViews=1 -eps=$3 -lossNorm='l2' -runs=10 -epochs=30 -dropLR=20

python main_vis.py -expID gaussian_weightd_masked_120_e30_00005 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $2 -LR=$5 -logDir ../logs/exp_gaussian/gaussian_priortodist_l2_masked_e30_$2-$5 -nViews=1 -eps=$3 -lossNorm='l2' -runs=10 -epochs=30 -dropLR=20


