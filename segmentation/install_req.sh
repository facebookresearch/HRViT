#!/bin/bash

# Copyright (c) Facebook, Inc. All Rights Reserved

pip install --user bcolz mxnet tensorboardX matplotlib easydict opencv-python einops --no-cache-dir -U | cat
pip install --user scikit-image imgaug PyTurboJPEG --no-cache-dir -U | cat
pip install --user scikit-learn --no-cache-dir -U | cat
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir -U | cat
pip install --user  termcolor imgaug prettytable --no-cache-dir -U | cat
pip install --user timm==0.4.12 --no-cache-dir -U | cat
pip install mmcv-full==1.3.0 --user  --no-cache-dir -U | cat
pip install terminaltables --user --no-cache-dir - U | cat
