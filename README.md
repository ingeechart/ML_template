# base code template
this is base code template for semantic segmentation experiments

# Prepare Dataset
```
ln -s /dataDisk/Datasets/cityscapes/
```

# Training.
for Single-Node multi-process distributed training, use
```
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE YOUR_TRAINING_SCRIPT.py --arg1 --arg2
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
├── data.py
│   ├── cityscapes_Dataset
│   ├── build_train_dataloader
│   ├── build_val_dataloader
│
├── model.py
│   ├── build_model
│   ├── backbone 
│   ├── modules
│   ├── model
│   ├── loss_function
│
├── utils
│   ├── utils.py
│   │   ├── get_logger
│   │   ├── AverageMeter
│   │   ├── adjust_learning_rate
│   │   ├── intersectionAndUnion
│   │   ├── intersectionAndUnionGPU
│   │   
│   ├── transform.py
│   │   ├── usilts for transform data
│
├── config.py(Todo)
│
├── result
│   ├── tb_data
│   │   ├── train
│   │   ├── val
│   │
│   ├── exp_name_logfile.log
│
├── cityscapes (ln -s /dataDisk/Datasets/cityscapes/)
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

