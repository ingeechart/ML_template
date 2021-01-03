import os
import yacs
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

''' Define system environment for Training '''
_C.SYS = CN(new_allowed=True)
_C.SYS.EXP_NAME = 'testing'
_C.SYS.OUTPUT_DIR = 'results/' + _C.SYS.EXP_NAME + '/'
# _C.SYS.GPUS = (0,1,2,3)
# _C.SYS.WORKERS = 8
# _C.SYS.PIN_MEMORY = True
# _C.SYS.LOCAL_RANK = ''

# Cudnn related params



_C.TRAIN = CN(new_allowed=True)
# _C.TRAIN.RESUME = 
# _C.TRAIN.START_EPOCH =
# _C.TRAIN.END_EPOCH = 
# _C.TRAIN.PRINT_FREQ = 
# _C.TRAIN.BATCH_SIZE = 
# _C.TRAIN.BATCH_SIZE_VAL = 
# _C.TRAIN.SEED = 

# _C.TRAIN.CUDNN = CN(new_allowed=True)
# _C.TRAIN.CUDNN.BENCHMARK = 
# _C.TRAIN.CUDNN.DETERMINISTIC = 
# _C.TRAIN.CUDNN.ENABLED = 

# _C.TRAIN.OPT = CN(new_allowed=True)
# _C.TRAIN.OPT.NAME = 
# _C.TRAIN.OPT.LR = 
# _C.TRAIN.OPT.NBB_LR = 
# _C.TRAIN.OPT.WD = 
# _C.TRAIN.OPT.MOMENTUM =

''' common params for Neural Network '''
_C.MODEL = CN(new_allowed=True)
# _C.MODEL.NAME = 
# _C.MODEL.BB_PRETRAINED = 
# _C.MODEL.LOSS = 
# _C.MODEL.FREEZE_BN = 
# _C.MODEL.NBB_KEYWORDS = 

 
# DATASET related params
_C.DATA = CN(new_allowed=True)
# _C.DATA.ROOT =
# _C.DATA.SET =
# _C.DATA.NUM_CLASSES =
# _C.DATA.TRAIN_SPLIT =
# _C.DATA.VAL_SPLIT =
# _C.DATA.TEST_SPLIT = 
# _C.DATA.SHUFFLE = 
# _C.DATA.DROP_LAST = 

# _C.DATA.AUG = CN(new_allowed=True)
# _C.DATA.AUG.R_SCALE = 
# _C.DATA.AUG.CROP_SIZE = 


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


def summary(cfg):
  def _getStringPerLine(attrs, numSpaces):
    s=''
    for k, v in attrs.items():
      if isinstance(v,CN):
        attrs_str = '{}{}:\n'.format(' '*numSpaces, str(k)) + _getStringPerLine(v, numSpaces+2)
      else:
        attrs_str =  '{}{}: {}\n'.format(' '*numSpaces,str(k),str(v))

      s+= attrs_str
    return s

  r = ''
  attrs = {k: v for k, v in cfg.items()}
  if attrs:
    r += _getStringPerLine(attrs,0)
  return r
      

if __name__ == '__main__':
    # import sys
    # with open(sys.argv[1], 'w') as f:
    #   print(sys.argv)
    #   print(_C, file=f)

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/base.yaml')
    print(summary(cfg))
    print(cfg)

  