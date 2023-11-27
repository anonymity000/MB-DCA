# MB-CrosA
This is an implementation of CrosA on the Meta_baseline (baseline network)

# Data
1. Create a directory: ${\color{red}materials}$
2. Mini-ImageNet: extract dataset (mini-imagenet.tar.gz) in  ${\color{red}./materials/mini-imagenet}$  [Link](https://drive.google.com/file/d/1uvE6rG_QM_tIUViEqN08filSkyYHsfpU/view)
3. Tiered Imagenet: extract dataset in  ${\color{red}./materials/tiered-imagenet}$  [Link](https://drive.google.com/file/d/1Y54Nwimfilhf245BaTnyZ7x16hnNc0B5/view)
4. Cifar： extract dataset in  ${\color{red}./materials/cifar}$ [Link](https://drive.google.com/file/d/1JfnX_8MIHHOdmiOTX96B8IGSgR8d6hZL/view)
## For cross-domain testing, only the test data is provided:
5. CUB200-2011: extract dataset in  ${\color{red}./materials/cub}$  [Link](https://drive.google.com/file/d/17P0W-pTWPZUvN5Ul8MYxxzduXAz-LpDM/view)
6. VGG-flower: extract dataset in  ${\color{red}./materials/vggflower}$  [Link](https://drive.google.com/file/d/1czK3osLvtyfa6YHQciPadC6QZllvbPL7/view)
7. FGVC-Aircraft: extract dataset in  ${\color{red}./materials/aircraft}$  [Link](https://drive.google.com/file/d/1sb-xvQC2b1xXkecEWc2BX5JK2bIoHd-W/view)
8. FGVC-Fungi: extract dataset in  ${\color{red}./materials/fungi}$  [Link](https://drive.google.com/file/d/1y9jl3xHKj3_9tNfuvpsGj196rBgCErZV/view)

# Usage
1. Create a directory: ./save, the training/testing results are saved in this directory.
2. Create a directory: ./test, copy the trained model into this directory for testing.
3. We have offerred the trained models. [Link](https://drive.google.com/drive/folders/1PTcUwVxuBRHVWkI2dTo_Ls00lZsrn1Zr)
## Train
```
python train_meta.py --config CONFIG_PATH.yam
```
example:
```
python train_meta.py --config configs/train_meta_mini.yaml
```
When using the different backbones on MB-CrosA, set the encoder: resnet4/convnet4, and the channels of ./models/convnet4.py or ./models/resnet4. The default channel is 64.
If you change the channel, also need to modify the value of hdim in meta_baseline.py.  32:800, 64:1600, 128:3200.
## Test
To test the performance, modify configs/test_few_shot.yaml by setting "load" to the saving file of the trained model. e.g. load: ./test/max-va-mini-resn4-64-1shot.pth
<br>Or copy the trained model to ./test directory.
<br> Edit the dataset of the test_few_shot.yaml for testing different datasets (including cross-domain testing).
```
python test_few_shot.py
```
