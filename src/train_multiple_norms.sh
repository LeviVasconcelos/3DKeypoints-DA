#!/bin/bash
DATASET=$1
EPOCHS=$2
STEP=$3
LR=$4
VIEWS=$5
RUNS=$6
BN=$7
NAME=$8
FOLDER=$9

echo $NAME
echo $BN
python main_vis.py -expID "$NAME"_dataset="$DATASET"_epochs="$EPOCHS"_step="$STEP"_lr="$LR"_views="$VIEWS"_bn="$BN" -loadModel ../models/ModelNet120.tar -propsFile ../priors/GT-source-props.npy -targetDataset $DATASET -LR=$LR -logDir ../logs/$FOLDER/"$NAME"_dataset="$DATASET"_epochs="$EPOCHS"_step="$STEP"_lr="$LR"_views="$VIEWS"_bn="$BN" -eps=0.000001 -lossNorm='l2' -runs=$RUNS -epochs=$EPOCHS -dropLR=$STEP -nViews=$VIEWS -batch_norm=$BN


