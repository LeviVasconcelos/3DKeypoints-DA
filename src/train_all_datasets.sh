#!/bin/bash

./train_multiple_norms.sh Redwood 0.001
./train_multiple_norms.sh RedwoodRGB 0.001

./train_multiple_norms.sh Redwood 0.0001
./train_multiple_norms.sh RedwoodRGB 0.0001

./train_multiple_norms.sh Redwood 0.00001
./train_multiple_norms.sh RedwoodRGB 0.00001

./train_multiple_norms.sh Redwood 0.0000001
./train_multiple_norms.sh RedwoodRGB 0.0000001

./train_multiple_norms.sh Redwood 0.00000001
./train_multiple_norms.sh RedwoodRGB 0.00000001

./train_multiple_norms.sh Redwood 0.000000001
./train_multiple_norms.sh RedwoodRGB 0.000000001
#./train_multiple_norms.sh ShapeNet

