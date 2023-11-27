# MB-CrosA
This is an implementation of CrosA on the Meta_baseline (baseline network)

# Data
1. Create a directory: **./materials** 
2. Mini-ImageNet: extract dataset (mini-imagenet.tar.gz) in  **./materials/mini-imagenet**  [Link](https://drive.google.com/file/d/1uvE6rG_QM_tIUViEqN08filSkyYHsfpU/view)
3. Tiered Imagenet: extract dataset in  **./materials/tiered-imagenet**  [Link](https://drive.google.com/file/d/1Y54Nwimfilhf245BaTnyZ7x16hnNc0B5/view)
4. Cifar： extract dataset in  **./materials/cifar** [Link](https://drive.google.com/file/d/1JfnX_8MIHHOdmiOTX96B8IGSgR8d6hZL/view)
## For cross-domain testing, only the test data is provided:
5. CUB200-2011: extract dataset in  **./materials/cub**  [Link](https://drive.google.com/file/d/17P0W-pTWPZUvN5Ul8MYxxzduXAz-LpDM/view)
6. VGG-flower: extract dataset in  **./materials/vggflower**  [Link](https://drive.google.com/file/d/1czK3osLvtyfa6YHQciPadC6QZllvbPL7/view)
7. FGVC-Aircraft: extract dataset in  **./materials/aircraft**  [Link](https://drive.google.com/file/d/1sb-xvQC2b1xXkecEWc2BX5JK2bIoHd-W/view)
8. FGVC-Fungi: extract dataset in  **./materials/fungi**  [Link](https://drive.google.com/file/d/1y9jl3xHKj3_9tNfuvpsGj196rBgCErZV/view)

# Usage
1. Create a directory:  **./save**, the training/testing results are saved in this directory.
2. Create a directory:  **./test**, copy the trained model into this directory for testing.
3. We have offerred the trained models. [Link](https://drive.google.com/drive/folders/1PTcUwVxuBRHVWkI2dTo_Ls00lZsrn1Zr)
   <br>You can download the models in **./test** for testing directly.
## Train
```
python train_meta.py --config CONFIG_PATH.yam
```
example:
```
python train_meta.py --config configs/train_meta_mini.yaml
```
1. Set n-way,k-shot in **./configs/train_meta_mini.yaml**, the default setting is 5-way,1-shot
1. Set the backbone in **./configs/train_meta_mini.yaml** e.g. encoder: resnet12 or encoder: convnet4
2. Set the channels of  **./models/convnet4.py** or  **./models/resnet12.py**. The default channel is 64.
3. If you change the channel, also need to modify the value of hdim in **./models/meta_baseline.py**.  
<br>if channel=32, hdim=800
<br>if channel=64, hdim=1600
<br>if channel=128, hdim=3200
## Test
1. Edit the **dataset** of **./configs/test_few_shot.yaml** for testing different datasets (including cross-domain testing).
2. Copy the trained model to  **./test** directory.
3. Edit the **load** of **./configs/test_few_shot.yaml** to load the trained model. e.g. load: **./test/max-va-mini-resn4-64-1shot.pth**.

Test in 5-way,1-shot setting:
```
python test_few_shot.py --shot 1
```
<br>Test in 5-way,5-shot setting:
```
python test_few_shot.py --shot 5
```
