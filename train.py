'''
KAU-RML ingee hong
'''

import argparse
import numpy as np
import os
import time 
import random
import config

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F


from model import build_model
from data.buildDataloader import build_train_loader, build_val_loader
from utils.utils import *

if torch.__version__ <= '1.1.0':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

    

def parse_args():
    ''' This is needed for torch.distributed.launch '''
    parser = argparse.ArgumentParser(description='Train instance classifier')
    # This is passed via launch.py
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--config', default=None, type=str, help='config file')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = config.get_cfg_defaults()
    cfg = merge_config(cfg, args)

    return args, cfg

def main():

    args, cfg = parse_args()

    # *  define paths ( output, logger) * #
    if not os.path.exists(cfg.SYS.OUTPUT_DIR):
        os.makedirs(cfg.SYS.OUTPUT_DIR)
        
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger_path = cfg.SYS.OUTPUT_DIR+'log/'+timestamp +'.log'

    global logger, tWriter, vWriter
    logger = get_logger(logger_path)
    tWriter = SummaryWriter(cfg.OUTPUT_DIR+'tb_data/train')
    vWriter = SummaryWriter(cfg.OUTPUT_DIR+'tb_data/val')
    
    # * controll random seed * #
    torch.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed(cfg.TRAIN.SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.SEED) # if use multi-GPU
    np.random.seed(cfg.TRAIN.SEED)
    random.seed(cfg.TRAIN.SEED)


    # TODO: findout what cudnn options are
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED


    msg = '[{time}]' 'starts experiments setting '\
            '{exp_name}'.format(time = time.ctime(), exp_name = cfg.EXP_NAME)
    logger.info(msg)
    

    # * GPU env setup. * #
    distributed = args.local_rank >= 0
    if distributed:
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )


    # * define MODEL * #
    if dist.get_rank()==0:
        logger.info("=> creating model ...")

    model = build_model(cfg)

    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # convert BN to syncBN
        model = model.to(device)
        model = DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=cfg.gpus).cuda()

    if dist.get_rank()==0:
        logger.info(model)


    # * define OPTIMIZER * #
    params_dict = dict(model.named_parameters())
    if cfg.cfg.MODEL.NBB_KEYWORDS:
        bb_lr = []
        nbb_lr = []
        nbb_keys = set()
        for k, param in params_dict.items():
            if any(part in k for part in cfg.MODEL.NBB_KEYWORDS):
                nbb_lr.append(param)
                nbb_keys.add(k)
            else:
                bb_lr.append(param)
        print(nbb_keys)
        params = [{'params': bb_lr, 'lr': cfg.TRAIN.OPT.LR}, 
                {'params': nbb_lr, 'lr': cfg.TRAIN.OPT.NBB_LR}]
    else:
        params = [{'params': list(params_dict.values()), 'lr': cfg.TRAIN.OPT.LR}]

    optimizer = torch.optim.SGD(params, lr=cfg.TRAIN.OPT.LR, 
                                        momentum=cfg.TRAIN.OPT.MOMENTUM, 
                                        weight_decay=cfg.TRAIN.OPT.WD)

    # * build DATALODER * #
    train_loader = build_train_loader(cfg)
    val_loader = build_val_loader(cfg)

    # * Training setup * #
    best_mIoU = 0
    best_epoch = 0

    # * RESUME * #
    if cfg.TRAIN.RESUME:
        if os.path.isfile(cfg.TRAIN.RESUME):
            if dist.get_rank()==0:
                logger.info("=> loading checkpoint '{}'".format(cfg.TRAIN.RESUME))

            checkpoint = torch.load(cfg.TRAIN.RESUME, map_location=lambda storage, loc: storage.cuda())
            
            new_epoch = ['TRAIN.START_EPOCH' ,checkpoint['epoch']]
            cfg = update_config(cfg, new_epoch)

            best_mIoU = checkpoint['best_mIoU']
            best_epoch = checkpoint['best_epoch']

            # state_dict control
            checkpoint_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            checkpoint_dict = {'module.'+k: v for k, v in checkpoint_dict.items()
                            if 'module.'+k in model_dict.keys()}

            for k, _ in checkpoint_dict.items():
                logger.info('=> loading {} pretrained model {}'.format(k, cfg.TRAIN.RESUME))

            logger.info(set(model_dict)==set(checkpoint_dict))
            assert set(model_dict)==set(checkpoint_dict)

            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict, strict=True)

            # optimizer control
            optimizer.load_state_dict(checkpoint['optimizer'])

            if dist.get_rank()==0:
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN.RESUME, checkpoint['epoch']))
        else:
            if dist.get_rank()==0:
                logger.info("=> no checkpoint found at '{}'".format(cfg.TRAIN.RESUME))


    logger.info('starts training')
    for epoch in range(start_epoch, cfg.TRAIN.END_EPOCH):

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(model, train_loader, optimizer, epoch, cfg)
        loss_val, mIoU_val, mAcc_val, allAcc_val = validation(model, val_loader, cfg)

        if args.local_rank <= 0:
            tWriter.add_scalar('loss', loss_train, epoch)
            tWriter.add_scalar('mIoU', mIoU_train, epoch)
            tWriter.add_scalar('mAcc', mAcc_train, epoch)
            tWriter.add_scalar('allAcc', allAcc_train, epoch)

            vWriter.add_scalar('loss', loss_val, epoch)
            vWriter.add_scalar('mIoU', mIoU_val, epoch)
            vWriter.add_scalar('mAcc', mAcc_val, epoch)
            vWriter.add_scalar('allAcc', allAcc_val, epoch)
        
            if best_mIoU < mIoU_val:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(cfg.OUTPUT_DIR,'best.pth.tar'))
                best_mIoU = mIoU_val
                best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_mIoU': best_mIoU,
                'bset_epoch': best_epoch,
            }, os.path.join(cfg.OUTPUT_DIR,'checkpoint.pth.tar'))

        if args.local_rank <= 0:
            msg = 'Loss_train: {:.10f}  Loss_val: {:.10f}'.format(loss_train, loss_val)
            logger.info(msg)
            msg = 'Best mIoU: {}  Best Epoch: {}'.format(best_mIoU, best_epoch)
            logger.info(msg)



    if args.local_rank <= 0:
        torch.save(model.module.state_dict(),
            os.path.join(cfg.OUTPUT_DIR, 'final_state.pth'))


def train(model, train_loader, optimizer, epoch, cfg):
    
    batch_time = AverageMeter('Batch_Time', ':6.3f')
    data_time = AverageMeter('Data_Time', ':6.3f')
    loss_meter = AverageMeter('Loss', ':.4e')

    intersection_meter = AverageMeter('IoU')
    union_meter = AverageMeter('Union')
    target_meter = AverageMeter('Target')

    model.train()
    end = time.time()
    max_iter = cfg.TRAIN.END_EPOCH * len(train_loader)

    for i_iter, (imgs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
 
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        loss, predict = model(imgs, target) # returns loss and predictions at each GPU

        # model.zero_grad()
        optimizer.zero_grad()
        loss.backward() # distributed.datapaprallel automatically gather and syncronize losses.
        optimizer.step()


        # TODO. this tis for recordings. need to refine this.
        n = imgs.size(0) # n = batch size of each GPU
        if dist.is_initialized():
            loss = loss * n
            dist.all_reduce(loss)
            n = n*dist.get_world_size()
            loss = loss / n
        loss_meter.update(loss.item(), n)


        # TODO 코드 확인. (evaluation metric), need to be changed with eval code of cityscape dataset
        intersection, union, target = intersectionAndUnionGPU(predict, target, 19, 255)
        if dist.is_initialized():
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)


        batch_time.update(time.time() - end)
        end = time.time()


        current_iter = epoch * len(train_loader) + i_iter + 1

        lr = adjust_learning_rate(optimizer,
                                  cfg.TRAIN.OPT.LR,
                                  cfg.TRAIN.OPT.NBB_LR
                                  max_iter,
                                  current_iter)

        # * compute remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        if (i_iter+1) % cfg.PRINT_FREQ == 0 and dist.get_rank() == 0:
            msg ='Epoch: [{}/{}]({:.2f}%) [{}/{}] '\
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '\
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '\
                    'Remain {remain_time} '\
                    'Loss {loss_meter.val:.4f} '\
                    'Accuracy {accuracy:.4f} '\
                    'lr {lr}.'.format(epoch+1, cfg.TRAIN.END_EPOCH, 
                                    ((epoch+1)/cfg.TRAIN.END_EPOCH)*100, 
                                    i_iter + 1, len(train_loader),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    remain_time=remain_time,
                                    loss_meter=loss_meter,
                                    accuracy=accuracy,
                                    lr = [x['lr'] for x in optimizer.param_groups])
            logger.info(msg)

        if dist.get_rank() == 0:
            tWriter.add_scalar('loss_batch', loss_meter.val, current_iter)
            tWriter.add_scalar('mIoU_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            tWriter.add_scalar('mAcc_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            tWriter.add_scalar('allAcc_batch', accuracy, current_iter)

        end = time.time()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if dist.get_rank() == 0:
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
                            epoch+1, cfg.TRAIN.END_EPOCH, mIoU, mAcc, allAcc))

    return loss_meter.avg, mIoU, mAcc, allAcc


def validation(model, val_loader, cfg):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i_iter, (imgs, target) in enumerate(val_loader):
            data_time.update(time.time() - end)
            imgs = imgs.cuda()
            target = target.cuda()

            loss, prediction = model(imgs, target)
    
            n = imgs.size(0)        
            if dist.is_initialized():
                loss = loss * n
                dist.all_reduce(loss)
                n = n*dist.get_world_size()
                loss = loss / n
            else:
                loss = torch.mean(loss)

            intersection, union, target = intersectionAndUnionGPU(prediction, target, 19, 255)
            if dist.is_initialized():
                torch.distributed.all_reduce(intersection), torch.distributed.all_reduce(union), torch.distributed.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            loss_meter.update(loss.item(), n)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i_iter+1) % cfg.PRINT_FREQ) == 0 and dist.get_rank() == 0:
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(i_iter + 1, len(val_loader),
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if dist.get_rank()==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(19):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__=='__main__':
    main()
