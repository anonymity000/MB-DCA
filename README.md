# MB-CrosA
This is an implementation of CrosA on the Meta_baseline (baseline network)
<tspan fill="red">Hello</tspan>
# Data
1. Create a directory: ${\color{red}materials\_qwe}$  
2. Mini-ImageNet: extract dataset (mini-imagenet.tar.gz) in  ${\color{red}./materials/mini-imagenet}$  [Link](https://drive.google.com/file/d/1uvE6rG_QM_tIUViEqN08filSkyYHsfpU/view)
3. Tiered Imagenet: extract dataset in  ${\color{red}./materials/tiered-imagenet}$  [Link](https://drive.google.com/file/d/1Y54Nwimfilhf245BaTnyZ7x16hnNc0B5/view)
4. Cifar： extract dataset in  ${\color{red}./materials/cifar}$ [Link](https://drive.google.com/file/d/1JfnX_8MIHHOdmiOTX96B8IGSgR8d6hZL/view)
## For cross-domain testing, only the test data is provided:
5. CUB200-2011: extract dataset in  ${\color{red}./materials/cub}$  [Link](https://drive.google.com/file/d/17P0W-pTWPZUvN5Ul8MYxxzduXAz-LpDM/view)
6. VGG-flower: extract dataset in  ${\color{red}./materials/vggflower}$  [Link](https://drive.google.com/file/d/1czK3osLvtyfa6YHQciPadC6QZllvbPL7/view)
7. FGVC-Aircraft: extract dataset in  ${\color{red}./materials/aircraft}$  [Link](https://drive.google.com/file/d/1sb-xvQC2b1xXkecEWc2BX5JK2bIoHd-W/view)
8. FGVC-Fungi: extract dataset in  ${\color{red}./materials/fungi}$  [Link](https://drive.google.com/file/d/1y9jl3xHKj3_9tNfuvpsGj196rBgCErZV/view)

# Usage
1. Create a directory:  ${\color{red}./save}$, the training/testing results are saved in this directory.
2. Create a directory:  ${\color{red}./test}$, copy the trained model into this directory for testing.
3. We have offerred the trained models. [Link](https://drive.google.com/drive/folders/1PTcUwVxuBRHVWkI2dTo_Ls00lZsrn1Zr)
   <br>You can download the models in ${\color{red}./test}$ for testing directly.
## Train
```
python train_meta.py --config CONFIG_PATH.yam
```
example:
```
python train_meta.py --config configs/train_meta_mini.yaml
```
1. Set the backbone in train_meta_mini.yaml e.g. encoder: resnet12 or encoder: convnet4
2. Set the channels of  ${\color{red}./models/convnet4.py}$ or  ${\color{red}./models/resnet12.py}$. The default channel is 64.
3. If you change the channel, also need to modify the value of hdim in ${\color{red}./models/meta_baseline.py}$.  
<br>if channel=32, hdim=800
<br>if channel=64, hdim=1600
<br>if channel=128, hdim=3200
## Test
1. Edit the dataset of the test_few_shot.yaml for testing different datasets (including cross-domain testing).
2. Modify test_few_shot.yaml by setting "load" to the saving file of the trained model. e.g. load: ./test/max-va-mini-resn4-64-1shot.pth
<br>Or copy the trained model to  ${\color{red}./test}$ directory.
```
python test_few_shot.py
```
