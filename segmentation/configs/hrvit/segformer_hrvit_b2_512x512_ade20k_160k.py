# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

_base_ = [
    '../_base_/models/segformer_hrvit.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='hrvit_b2',
        drop_path_rate=0.2,
        with_cp=False,
    ),
    decode_head=dict(
        in_channels=[48, 96, 192, 384],
        num_classes=150,
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)
evaluation = dict(interval=16000, metric='mIoU')

