#!/bin/bash
./train_all_datasets.sh 2 annealed_full_grad 1 test_with_annealing 0.0002 1 0.8
./train_all_datasets.sh 2 annealed_full_grad 1 test_with_annealing 0.0002 1 0.85
./train_all_datasets.sh 2 annealed_full_grad 1 test_with_annealing 0.0002 1 0.9
./train_all_datasets.sh 2 annealed_full_grad 1 test_with_annealing 0.0002 1 0.95

