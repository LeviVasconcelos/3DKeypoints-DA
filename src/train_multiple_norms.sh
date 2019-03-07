#!/bin/bash
python main_vis.py -expID gaussian_mds_weighted_masked_dataset=$1_epochs=$2_step=$3_lr=$4_views=$5 -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $1 -LR=$4 -logDir ../logs/mds_base/gaussian_mds_weighted_masked_dataset=$1_epochs=$2_step=$3_lr=$4_views=$5 -eps=0.000001 -lossNorm='l2' -runs=$6 -epochs=$2 -dropLR=$3 -nViews=$5


