# from models import ctx48
import importlib

def build_model(cfg):
    
    importlib.import_module('models.'+cfg.MODEL.NAME)
    model = eval(cfg.MODEL.NAME+'.Ctxnet()')

    return model