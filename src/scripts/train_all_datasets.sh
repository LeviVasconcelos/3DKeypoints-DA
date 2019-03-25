#!/bin/bash

EPOCHS=30
STEP=20
LR=$5
VIEWS=$6
RUNS=$3
BN=$1
NAME=$2
FOLDER=$4
TH=$7

mkdir $FOLDER
./train_multiple_norms.sh ShapeNet $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $TH
./train_multiple_norms.sh Redwood $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $TH
./train_multiple_norms.sh RedwoodRGB $EPOCHS $STEP $LR $VIEWS $RUNS $BN $NAME $FOLDER $TH


