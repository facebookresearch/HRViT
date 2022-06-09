# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

MMSEG_PATH=$1
cp -r configs/hrvit $MMSEG_PATH/configs/;
cp configs/_base_/datasets/cityscapes_1024x1024_repeat.py $MMSEG_PATH/configs/_base_/datasets;
cp configs/_base_/models/segformer_hrvit.py $MMSEG_PATH/configs/_base_/models;
cp configs/_base_/models/upernet_hrvit.py $MMSEG_PATH/configs/_base_/models;
cp mmseg/models/backbones/hrvit.py $MMSEG_PATH/mmseg/models/backbones/;
cp mmseg/models/backbones/__init__.py $MMSEG_PATH/mmseg/models/backbones/;
cp mmseg/models/decode_heads/segformer_head.py $MMSEG_PATH/mmseg/models/decode_heads/;
cp mmseg/models/decode_heads/__init__.py $MMSEG_PATH/mmseg/models/decode_heads/;
cp mmcv_custom/checkpoint.py $MMSEG_PATH/mmcv_custom/;
