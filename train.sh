# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

NUM_NODE=$1; shift
NUM_PROC=$1; shift
NODE_RANK=$1; shift

python3 pretrain.py --num-machines=$NUM_NODE --ngpus-per-node=$NUM_PROC --machine-rank=$NODE_RANK "$@"

