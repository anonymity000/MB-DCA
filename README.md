# MB-CrosA
This is an implementation of CrosA on the Meta_baseline (baseline network)

# Data
1. Create a directory: materials. 
2. miniImagenet: extract dataset (mini-imagenet.tar.gz) in ./materials/mini-imagenet  [Link](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view?pli=1)
         Rename the three files as:
           miniImageNet_category_split_train_phase_train.pickle
           miniImageNet_category_split_val.pickle
           miniImageNet_category_split_test.pickle
4. tieredImagenet: extract dataset in ./materials/mini-imagenet  [Link] ()
5. CUB200-2011: extract dataset in ./dataset/cubirds2  [Link](https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view)???

# Usage
1. Create a directory: ./save The training/testing results are saved in this directory.
2. Create a directory: ./materials/mini-imagenet
3. ???The trained models are copied in this directory. We have offerred the trained models.
## Train
```
python train_meta.py --config CONFIG_PATH.yaml
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
