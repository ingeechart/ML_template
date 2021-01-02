# Pytorch Code Template
this is base code template for semantic segmentation experiments

# Prepare Dataset
```
ln -s /dataDisk/Datasets/[option]/ .
```
option: `cityscapes`, `coco`, `ImageNet` 

# Training.
for Single-Node multi-process distributed training, use
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_TRAINING_SCRIPT.py --arg1 --arg2

python -m torch.distributed.launch --nproc_per_node=4 train.py --
```
# Test.
```

```

# Visualize 
```
```

# Structure
```
base
├── README.md
├── train.py
│   ├── main
│   ├── train
│   ├── validation
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── buildDataloader.py
│   │   ├── build_train_dataloader
│   │   ├── build_val_dataloader
│   │
│   ├── transforms.py
│
├── model/ (will be modified)
│   ├── __init__.py
│   ├── backbone 
│   ├── modules
│   ├── model
│   ├── loss_function
│
├── utils/ (will be modified)
│   ├── utils.py
│       ├── get_logger
│       ├── update/merge config
│       ├── AverageMeter
│       ├── adjust_learning_rate
│       ├── intersectionAndUnion
│       ├── intersectionAndUnionGPU
│   
├── configs/
│   ├── base.yaml
│
├── config.py
│
├── result/
│   ├── tb_data
│   │   ├── train
│   │   ├── val
│   │
│   ├── logs
│
├── cityscapes/
```

# Todo
0. evaluation metric
1. config file
2. define logger str format
3. test code
4. visualize code for predicted images
5. Get FLOPS and params


## reference
https://github.com/pytorch/examples/blob/master/imagenet/main.py#L266
https://github.com/pytorch/examples/tree/master/distributed/ddp
https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

