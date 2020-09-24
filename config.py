import os
from yacs.config import CfgNode as CN

_C = CN()

_C.EXP_NAME = 'testing'
_C.OUTPUT_DIR = 'result/'
_C.LOG_DIR = 'result/looog.log'
_C.GPUS = (0,1,2,4)
_C.WORKERS = 8
_C.PRINT_FREQ = 20
_C.PIN_MEMORY = True
_C.LOCAL_RANK = ''
# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True


_C.TRAIN = CN()
_C.TRAIN.RESUME = ''
_C.TRAIN.INIT_LR = 0.1
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 5e-4
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.END_EPOCH = 400


''' common params for NETWORK '''
_C.MODEL = CN()
_C.MODEL.NAME = 'model'
_C.MODEL.BB_PRETRAINED = ''
_C.MODEL.LOSS = ''
_C.MODEL.FREEZE_BN = ''
_C.MODEL.NONBACKBONE_KEYWORDS = []
_C.MODEL.NONBACKBONE_MULT = 10
 


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'cityscapes'
_C.DATASET.NUM_CLASSES = 19
_C.DATASET.TRAIN_SET = ''
_C.DATASET.TRAIN_BATCH_SIZE = 16
_C.DATASET.VAL_BATCH_SIZE = 4
_C.DATASET.TEST_SET = ''

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
      print(sys.argv)
      print(_C, file=f)