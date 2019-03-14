#!/bin/bash

EPOCHS=30
STEP=20
LR=$5
VIEWS=8
RUNS=$3
BN=$1
NAME=$2
FOLDER=$4
NORM=$6

mkdir $FOLDER
./train_multiple_norms.sh RedwoodRGB $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $NORM
./train_multiple_norms.sh Redwood $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $NORM
./train_multiple_norms.sh ShapeNet $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $NORM



