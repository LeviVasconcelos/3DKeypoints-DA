#!/bin/bash

./train_multiple_norms.sh Redwood 20 10 0.000100 1 5
./train_multiple_norms.sh RedwoodRGB 20 10 0.0001 1 5

./train_multiple_norms.sh Redwood 20 10 0.0001 8 5
./train_multiple_norms.sh RedwoodRGB 20 10 0.0001 8 5

./train_multiple_norms.sh Redwood 30 20 0.000005 1 5
./train_multiple_norms.sh RedwoodRGB 30 20 0.000005 1 5

./train_multiple_norms.sh Redwood 30 20 0.000005 8 5
./train_multiple_norms.sh RedwoodRGB 30 20 0.000005 8 5




