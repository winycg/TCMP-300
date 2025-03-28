import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import os
import shutil
import argparse
import numpy as np
import re


import models
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
from utils import cal_param_size, cal_multi_adds, AverageMeter, adjust_lr, correct_num, set_logger


import time
import math


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--traindir', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--valdir', default='./data/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet50', type=str, help='network architecture')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight deacy')
parser.add_argument('--lr-type', default='cosine', type=str, help='learning rate strategy')
parser.add_argument('--milestones', default=[150, 225], type=int, nargs='+', help='milestones for lr-multistep')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

parser.add_argument('--imagenet-pretrained', default='', type=str, help='imagenet pretrained')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--checkpoint-dir', default='./checkpoint_auto_mixup', type=str, help='checkpoint directory')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)

testset = torchvision.datasets.ImageFolder(
    args.valdir, 
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
]))

NUM_CLASSES = len(set(testset.classes))

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

from torch.utils.data import default_collate

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=12,
                                          pin_memory=True)

# model = getattr(models, args.arch)
if args.arch == 'convnext_tiny':
    from torchvision.models import convnext_tiny
    model = convnext_tiny(num_classes=NUM_CLASSES)
elif args.arch == 'resnet50':
    from torchvision.models import resnet50
    model = resnet50(num_classes=NUM_CLASSES)
elif args.arch == 'densenet161':
    from torchvision.models import densenet161
    model = densenet161(num_classes=NUM_CLASSES)
elif args.arch == 'efficientnet_b0':
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(num_classes=NUM_CLASSES)
elif args.arch == 'regnet_x_3_2gf':
    from torchvision.models import regnet_x_3_2gf
    model = regnet_x_3_2gf(num_classes=NUM_CLASSES)
elif args.arch == 'vit_b_16':
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(num_classes=NUM_CLASSES)
    '''weights = ViT_B_16_Weights.verify(ViT_B_16_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['heads.head.weight']
    del imagenet_model_dict['heads.head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)'''
elif args.arch == 'mobilenet_v3_large':
    from torchvision.models import mobilenet_v3_large
    model = mobilenet_v3_large(num_classes=NUM_CLASSES)
elif args.arch == 'shufflenet_v2_x1_5':
    from torchvision.models import shufflenet_v2_x1_5
    model = shufflenet_v2_x1_5(num_classes=NUM_CLASSES)
elif args.arch == 'swin_b':
    from torchvision.models import swin_b, Swin_B_Weights
    model = swin_b(num_classes=NUM_CLASSES)
    '''weights = Swin_B_Weights.verify(Swin_B_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)'''
elif args.arch == 'swin_s':
    from torchvision.models import swin_s, Swin_S_Weights
    model = swin_s(num_classes=NUM_CLASSES)
    '''weights = Swin_S_Weights.verify(Swin_S_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)'''
elif args.arch == 'swin_t':
    from torchvision.models import swin_t, Swin_T_Weights
    model = swin_t(num_classes=NUM_CLASSES)
    '''weights = Swin_T_Weights.verify(Swin_T_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)'''
else:
    raise NotImplementedError
# net = model(num_classes=NUM_CLASSES)
net = model
net.eval()
resolution = (1, 3, 224, 224)

del(net)

net = model.cuda()

initalized_model_dict = net.state_dict()
imagenet_model_dict = torch.load(args.imagenet_pretrained, map_location=torch.device('cpu'))
net.load_state_dict(imagenet_model_dict['net'])
print('Load imagenet pretrained weights!')
cudnn.benchmark = True



def test():
    net.eval()
    
    # classes = np.random.choice(np.arange(NUM_CLASSES), 10, replace=False)
    # classes = [72, 101, 131, 149, 46, 237, 68, 259, 73, 20]
    classes = [64, 17, 25, 9, 75, 267, 68, 11, 27, 7]
    classes = np.array(classes)
    embeddings_list = []
    target_list = []
    
    last_conv_layer = net._modules.get('avgpool')

    # 定义一个钩子函数
    def hook(module, input, output):
        # 将输出保存在全局变量中
        global features
        features = output

    # 注册钩子
    last_conv_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        for batch_idx, (inputs, target) in enumerate(testloader):
            inputs, target = inputs.cuda(), target.cuda()
            if np.sum(classes == target.item()) == 1:
                net(inputs)
                embeddings_list.append(features.squeeze(0).squeeze(1).squeeze(1).cpu().numpy())
                target_list.append(target.item())

    embeddings_list = np.asarray(embeddings_list)
    target_list = np.asarray(target_list)
    print(embeddings_list.shape)
    print(target_list.shape)
    np.save(f'npys/top10_{args.arch}_embeddings.npy', embeddings_list)
    np.save(f'npys/top10_{args.arch}_target.npy', target_list)

def test_code():
    last_conv_layer = net._modules.get('layer4')

    # 定义一个钩子函数
    def hook(module, input, output):
        # 将输出保存在全局变量中
        global features
        features = output

    # 注册钩子
    last_conv_layer.register_forward_hook(hook)

    # 创建一个示例输入
    example_input = torch.rand(1, 3, 224, 224).cuda()

    # 使用模型进行前向传播
    net(example_input)

    # 输出捕获的特征
    print(features.shape)

    return


if __name__ == '__main__':
    
    test()
