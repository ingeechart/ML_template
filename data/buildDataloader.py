import os
import glob
import numpy as np

import torch
from torch.utils import data
import torch.distributed as dist

from data import transform
from PIL import Image



def build_train_loader(cfg):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # make transform
    train_transforms = transform.Compose([
        transform.RandScale(cfg.DATA.AUG.R_SCALE),
        transform.RandomHorizontalFlip(),
        transform.Crop(cfg.DATA.AUG.CROP_SIZE, crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])

    train_data = Cityscapes(data_root='cityscapes/', split='train', transforms=train_transforms)

    # TODO
    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(train_data)
    else:
        sampler =  None

    # TODO check shuffle
    data_loader = torch.utils.data.DataLoader(train_data,
                            num_workers= cfg.SYS.WORKERS,
                            batch_size=cfg.DATASET.TRAIN_BATCH_SIZE//len(cfg.SYS.GPUS),
                            shuffle=cfg.DATA.SHUFFLE,
                            pin_memory=cfg.SYS.PIN_MEMORY,
                            drop_last=cfg.DATA.DROP_LAST,
                            sampler=sampler)

    return data_loader

def build_val_loader(cfg):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    # make transform
    val_transforms = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])
    val_data = Cityscapes(data_root='cityscapes/', split='val', transforms=val_transforms)

    if dist.is_initialized():
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(val_data)
    else:
        sampler =  None

    data_loader = torch.utils.data.DataLoader(val_data,
                                            num_workers=cfg.SYS.WORKERS//2,
                                            batch_size=cfg.DATASET.BATCH_SIAE_VAL//len(cfg.SYS.GPUS),
                                            shuffle=cfg.DATA.SHUFFLE,
                                            pin_memory=cfg.SYS.PIN_MEMORY,
                                            sampler=sampler)

    return data_loader

def build_test_loader(H,W):

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    val_transforms = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)
            ])
    val_data = Cityscapes(data_root='cityscapes/', split='val', transforms=val_transforms)
      
    img_list = val_data.get_file_path()
    
    data_loader = torch.utils.data.DataLoader(val_data,
                                            num_workers=4,
                                            batch_size=1,
                                            shuffle=False,
                                            pin_memory=True,
                                            sampler=None)

    return data_loader, img_list


if __name__=='__main__':
    import torchvision
    import matplotlib.pyplot as plt
    import matplotlib.colors as clr
    from data import transform

    root='data/cityscapes/'
    # print(root)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomHorizontalFlip(),
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])

    dataset = Cityscapes(root,transforms=train_transform)

    img, label =dataset.__getitem__(0)
    print(img.dtype)
    print(label.shape)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig_in.add_subplot(1,2,2)
    ax.imshow(label)
    plt.show()
    # img_path = 'dataset/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000001_044227_leftImg8bit.png'
    # gt_path = img_path.replace('leftImg8bit','gtFine_trainvaltest/gtFine',1)
    # inst_path = gt_path.replace('leftImg8bit','gtFine_instanceIds')
    # instance = Image.open(inst_path)

    # fig_in = plt.figure()
    # ax = fig_in.add_subplot(1,2,1)
    # ax.imshow(img)
    # ax = fig_in.add_subplot(1,2,2)
    # ax.imshow(instance)


    # fig = plt.figure()
    # rows=3
    # cols=3
    # for i, mask in enumerate(cmap):
    #     ax = fig.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # # plt.show()

    # fig2 = plt.figure()
    # for i, mask in enumerate(ox):
    #     ax = fig2.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # # plt.show()

    # fig3 = plt.figure()
    # for i, mask in enumerate(oy):
    #     ax = fig3.add_subplot(rows,cols, i+1)
    #     ax.imshow(mask)
    # plt.show()