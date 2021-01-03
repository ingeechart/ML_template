import os
import glob
import numpy as np

import torch
from torch.utils import data
import torch.distributed as dist

from data import transform
from PIL import Image

class Dataset(data.Dataset):
    def __init__(self, cfg, data_root, split='train',transforms=False, ignore_label=255):
        '''
        path:
            root:
            Image: 
            ground truth:
        '''

        self.root = data_root
        self.split = split

        self.image_base = os.path.join(self.root,self.split)
        self.files = glob.glob(self.image_base+'/*/*.png')
        assert len(self.files) is not 0 , ' cannot find data!'
        
        self.transforms = transforms

        self.nClasses = cfg.DATA.NUM_CLASSES # 19

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_path = self.files[index]
        img = Image.open(img_path)
        img = np.asarray(img)

        gt_path = img_path.replace('gt_path',1)

        label = Image.open(gt_path)
        label = np.asarray(label)


        if self.transforms:
            img, label= self.transforms(img,label)
        label = self.convert_label(label)

        return img, label

     
    def get_file_path(self):
        return self.files


if __name__=='__main__':
