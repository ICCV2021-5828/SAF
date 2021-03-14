#!/bin/bash

python main.py --network SAF --total_iters 100000 --naive_entropy --MIXUP_high 0.1 --MIXUP_NAIVE_low 0.1 --MIXUP_NAIVE_high 0.5 --MIXUP_NAIVE_max 50000 \
	--dataset visda-2017 --batch_size 48 --eval_batch_size 100 --num_workers 8 --iter_per_epoch 500 --MDD_mask_all
