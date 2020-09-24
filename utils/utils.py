import logging
import torch
import torch.nn as nn
import torch.nn.init as initer
import torch.distributed as torch_dist


def update_config(cfg, args):
    cfg.defrost()
    
    if args.config is not None:
        cfg.merge_from_file(args.config)
    if args.opts is []:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg

def get_logger(path):
    
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
    file_handler = logging.FileHandler(path)
    logger.addHandler(file_handler)
    
    return logger



class AverageMeter(object):
    """
        code is from pytorch imagenet examples
        Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print(val, n)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9, nbb_mult=10):
    """
        code is from pytorch imagenet examples
        Sets the learning rate to the initial LR decayed with poly learning rate with power 0.9
        
    """

    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr

    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult

    return lr



def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    '''
        'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.

        params:
            output: BxHxW size tensor filled with predicted class labels.
            target: BxHxW size tensor filled with ground truth class labels.
            K:      number of classes

        returns:
            area of intersection
            area of union
            area of target
    '''
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape

    output = output.view(2,-1)
    target = target.view(2,-1)

    output[target == ignore_index] = ignore_index

    # mask of intersection where predict==target
    intersection = output[output == target] 
    
    # compute histogram of tensor. shape: [19]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1) 
    area_output = torch.histc(output, bins=K, min=0, max=K-1) 
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target




def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette

    color = Image.fromarray(gray).convert('P')
    color.putpalette(palette)
    return color