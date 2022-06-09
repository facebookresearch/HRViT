# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='hrvit_b1',
        drop_path_rate=0.1,
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type="SegformerHead",
        in_channels=[32, 64, 128, 256],
        in_index=[0, 1, 2, 3],
        mlp_norm=False,
        mlp_act=False,
        channels=256,
        dropout_ratio=0.1,
        num_classes=21,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
