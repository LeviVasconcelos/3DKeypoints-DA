#!/bin/bash
./train_all_datasets.sh 1 dists_l2_r5 5 checks 0.0001 l2
./train_all_datasets.sh 1 dists_l1 5 checks 0.0001 l1
./train_all_datasets.sh 1 dists_frob 5 checks 0.0001 frobenius

