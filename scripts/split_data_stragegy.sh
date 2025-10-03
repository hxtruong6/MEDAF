#!/bin/bash

# echo "Creating strategy 1 splits"

# python utils/create_chestxray_strategy1_splits.py \
#     --source datasets/data/NIH/Data_Entry_2017.csv \
#     --test-list datasets/data/NIH/test_list.txt \
#     --output-dir datasets/data/NIH \
#     --val-ratio 0.1 \
#     --seed 42

echo "Creating strategy 2 splits"

python utils/create_chestxray_strategy2_splits.py \
    --source datasets/data/NIH/Data_Entry_2017.csv \
    --test-list datasets/data/NIH/test_list.txt \
    --output-dir datasets/data/NIH