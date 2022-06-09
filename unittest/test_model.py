# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import models

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.models import create_model
from timm.utils import *

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')

import warnings
warnings.filterwarnings('ignore')


def test_model():
    model = create_model(
        "HRViT_b3_224",
        pretrained=False,
        in_channels=3,
        num_classes=1000,
        with_cp=False,
    )
    print(model)
    # with open("./model_def.log", "w") as f:
    #     f.write(str(model))
    x = torch.randn(1, 3, 224, 224)
    y = model.forward_features(x)
    print(f"{[i.size() for i in y]}")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
    print(sum(p.numel() for p in list(model.stem.parameters())+list(model.features.parameters()) if p.requires_grad)/1e6, "M")
    print(sum(p.numel() for p in model.head.parameters() if p.requires_grad)/1e6, "M")

if __name__ == '__main__':
    test_model()
