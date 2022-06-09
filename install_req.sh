#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

pip install --user bcolz mxnet tensorboardX matplotlib easydict opencv-python einops --no-cache-dir -U | cat
pip install --user scikit-image imgaug PyTurboJPEG --no-cache-dir -U | cat
pip install --user scikit-learn --no-cache-dir -U | cat
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir -U | cat
pip install --user  termcolor imgaug prettytable --no-cache-dir -U | cat
pip install --user timm==0.4.12 --no-cache-dir -U | cat

