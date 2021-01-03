import os
import yacs
from yacs.config import CfgNode as CN

_C = CN()

''' Define system environment for Training '''
_C.SYS = CN(new_allowed=True)
_C.SYS.EXP_NAME = 'testing'
_C.SYS.OUTPUT_DIR = 'results/' + _C.SYS.EXP_NAME + '/'

_C.SYS.GPUS = (0,1,2,4)
_C.SYS.WORKERS = 8
_C.SYS.PIN_MEMORY = True
_C.SYS.LOCAL_RANK = ''

# Cudnn related params
_C.CUDNN = CN(new_allowed=True)
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


_C.TRAIN = CN(new_allowed=True)
_C.TRAIN.RESUME = ''
_C.TRAIN.START_EPOCH = 1
_C.TRAIN.END_EPOCH = 400
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.BATCH_SIZE_VAL = 4
_C.TRAIN.SEED = 1007

_C.TRAIN.OPT = CN(new_allowed=True)
_C.TRAIN.OPT.NAME = 'SGD'
_C.TRAIN.OPT.LR = 0.02
# _C.TRAIN.OPT.NBB_LR = NONE
_C.TRAIN.OPT.WD = 0.0001
_C.TRAIN.OPT.MOMENTUM = 0.9

''' common params for NETWORK '''
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'model'
_C.MODEL.BB_PRETRAINED = ''
_C.MODEL.LOSS = ''
_C.MODEL.FREEZE_BN = ''
_C.MODEL.NBB_KEYWORDS = []

 

# DATASET related params
_C.DATA = CN(new_allowed=True)
_C.DATA.ROOT = 'cityscapes/'
_C.DATA.SET = 'cityscapes'
_C.DATA.NUM_CLASSES = 19
_C.DATA.TRAIN_SPLIT = 'train'
_C.DATA.VAL_SPLIT = 'val'
_C.DATA.TEST_SPLIT = 'test'
_C.DATA.SHUFFLE = False
_C.DATA.DROP_LAST = True

_C.DATA.AUG = CN(new_allowed=True)
_C.DATA.AUG.R_SCALE = [0.5,2]
_C.DATA.AUG.CROP_SIZE = [768, 1536] # [1024,2048]


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


if __name__ == '__main__':
    # import sys
    # with open(sys.argv[1], 'w') as f:
    #   print(sys.argv)
    #   print(_C, file=f)
    cfg = get_cfg_defaults()
    lists = ['TRAIN.START_EPOCH', 100, 'TRAIN.BEST_ACC', 100]
    print(lists)
    cfg.merge_from_list(lists)
    # cfg.TRAIN.BEST_MIOU = 100
    print(cfg)