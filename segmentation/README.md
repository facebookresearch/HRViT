# ADE20k/Cityscapes Semantic Segmentation with HRViT


## Getting started

1. Install the [Swin_Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) repository and some required packages.

```bash
git clone https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
bash install_req.sh
```

2. Move the HRViT configs and backbone/decode_head file to the corresponding folder.

```bash
cp -r configs/hrvit <MMSEG_PATH>/configs/
cp configs/_base_/datasets/cityscapes_1024x1024_repeat.py <MMSEG_PATH>/configs/_base_/datasets
cp configs/_base_/models/segformer_hrvit.py <MMSEG_PATH>/configs/_base_/models
cp configs/_base_/models/upernet_hrvit.py <MMSEG_PATH>/configs/_base_/models
cp mmseg/models/backbones/hrvit.py <MMSEG_PATH>/mmseg/models/backbones/
cp mmseg/models/backbones/__init__.py <MMSEG_PATH>/mmseg/models/backbones/
cp mmseg/models/decode_heads/segformer_head.py <MMSEG_PATH>/mmseg/models/decode_heads/
cp mmseg/models/decode_heads/__init__.py <MMSEG_PATH>/mmseg/models/decode_heads/
cp mmcv_custom/checkpoint.py <MMSEG_PATH>/mmcv_custom/
```
or
run the following script
```bash
bash install_mmseg_hrvit.sh <path-to-Swin-Transformer-Semantic-Segmentation>
```

3. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

4. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k and Cityscapes dataset.

## Training on ADE20K or Cityscapes

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS> --options model.pretrained=<PRETRAIN_MODEL_PATH>
```

For example, using an HRViT-b1 backbone with Segformer head:
```bash
bash tools/dist_train.sh \
    configs/hrvit/segformer_hrvit_b1_512x512_ade20k_160k.py 8 \
    --options model.pretrained=<PRETRAIN_MODEL_PATH>
```

More config files can be found at [`configs/hrvit`](configs/hrvit).


## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```

For example, evaluate an HRViT_b1 backbone with Segformer head:
```bash
bash tools/dist_test.sh configs/hrvit/segformer_hrvit_b1_512x512_ade20k_160k.py \
    <CHECKPOINT_PATH> 8 --eval mIoU
```


---

## Acknowledgment

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library, the [Swin](https://github.com/microsoft/Swin-Transformer) repository, the [CSWin](https://github.com/microsoft/CSwin-Transformer) repository.
