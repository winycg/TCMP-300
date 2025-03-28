import torch
import os
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from bisect import bisect_right
import logging
from PIL import Image

def cal_param_size(model):
    return sum([i.numel() for i in model.parameters()])


count_ops = 0
def measure_layer(layer, x, multi_add=1):
    delta_ops = 0
    type_name = str(layer)[:str(layer).find('(')].strip()

    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) //
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) //
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w // layer.groups * multi_add

    ### ops_linear
    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = 0
        delta_ops = weight_ops + bias_ops

    global count_ops
    count_ops += delta_ops
    return


def is_leaf(module):
    return sum(1 for x in module.children()) == 0


def should_measure(module):
    if str(module).startswith('Sequential'):
        return False
    if is_leaf(module):
        return True
    return False


def cal_multi_adds(model, shape=(2,3,32,32)):
    global count_ops
    count_ops = 0
    data = torch.zeros(shape)

    def new_forward(m):
        def lambda_forward(x):
            measure_layer(m, x)
            return m.old_forward(x)
        return lambda_forward

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data)
    restore_forward(model)

    return count_ops




def set_logger(filename, name='experiments'):
    logger = logging.getLogger(name) 
    logger.setLevel(level = logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')


    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter) 

    console = logging.StreamHandler() 
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    
    return logger



class AverageMeter(object):
    """Computes and stores the average and current value"""
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
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res



def adjust_lr(optimizer, epoch, args):
    cur_lr = 0.
    if args.lr_type == 'multistep':
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)
    elif args.lr_type == 'cosine':
        cur_lr = args.init_lr * 0.5 * (1. + math.cos(np.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

        return cur_lr
    
    
def save_img(img_tensor, path):
    
    img = img_tensor.detach().cpu().numpy()[0,:,:,:]
    img = np.transpose(img, (1,2,0))
    img = (img- img.min()) / (img.max()-img.min())
    img = img * 255.
    image_pil= Image.fromarray(np.uint8(img))
    image_pil.save(path)