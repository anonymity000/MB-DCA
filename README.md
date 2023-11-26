# MB-CrosA
This is an implementation of CrosA on the Meta_baseline (baseline network)

# Data
1. Create a directory: materials. 
2. Mini-ImageNet: extract dataset (mini-imagenet.tar.gz) in ./materials/mini-imagenet  [Link](https://drive.google.com/file/d/1uvE6rG_QM_tIUViEqN08filSkyYHsfpU/view)
3. Tiered Imagenet: extract dataset in ./materials/tiered-imagenet  [Link] (https://drive.google.com/file/d/1_4FsUC4ofwRiwTOhKh8j_hvhdcYucIDE/view)
4. Cifar： extract dataset in ./materials/cifar [Link](https://drive.google.com/file/d/1JfnX_8MIHHOdmiOTX96B8IGSgR8d6hZL/view)
For cross-domain testing, only the test data is provided:
5. CUB200-2011: extract dataset in ./dataset/cub  [Link]()
6. VGG-flower: extract dataset in ./dataset/vggflower  [Link]()
7. FGVC-Aircraft: extract dataset in ./dataset/  [Link]()
8. FGVC-Fungi: extract dataset in ./dataset/  [Link]()

# Usage
1. Create a directory: ./save The training/testing results are saved in this directory.
2. Create a directory: ./materials/mini-imagenet
3. ???The trained models are copied in this directory. We have offerred the trained models.
## Train
```
python train_meta.py --config CONFIG_PATH.yam
```
example:
```
python train_meta.py --config configs/train_meta_mini.yaml
```

## Test
To test the performance, modify configs/test_few_shot.yaml by setting "load" to the saving file of the trained model. Or create a new directory "test" and move the "max-va.pth" to this directory.
```
python test_few_shot.py
```
