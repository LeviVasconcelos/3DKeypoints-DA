#!/bin/bash

NAME=$1

mkdir $FOLDER
./train_all_datasets.sh 1 $NAME 5 ablation
./train_all_datasets.sh 0 $NAME 5 ablation



