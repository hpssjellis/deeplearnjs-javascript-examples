#!/bin/bash  
source ~/virtual-tf/bin/activate


echo "running python softmax.py --train simdata/linear_data_train.csv --test simdata/linear_data_eval.csv --num_epochs 2"
echo "from the bcomposes-examples folder"
echo ""

python softmax.py --train simdata/linear_data_train.csv --test simdata/linear_data_eval.csv --num_epochs 2



echo ""
echo "--------------------Done--------------------------"
