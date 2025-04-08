import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import shutil
import argparse
import numpy as np
import re


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
parser.add_argument('--hybridmix', action='store_true', help='using data augmentation hybridmix')

parser.add_argument('--imagenet-pretrained', default='', type=str, help='imagenet pretrained')
parser.add_argument('--resume-checkpoint', default='./checkpoint/resnet32.pth.tar', type=str, help='resume checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--evaluate-checkpoint', default='./checkpoint/resnet32_best.pth.tar', type=str, help='evaluate checkpoint')
parser.add_argument('--checkpoint-dir', default='./checkpoint_baseline_new_336', type=str, help='checkpoint directory')

'''class CustomImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        try:
            # 尝试打开图片文件
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        except IOError as e:
            # 如果文件损坏或不完整，打印错误并跳过该文件
            print(f"无法打开图片 {path}: {e}")
            return None, target

        return image, target'''
    

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
          'arch' + '_' +  args.arch + '_'+\
          'seed'+ str(args.manual_seed)
args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
if not os.path.isdir(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

log_txt =  os.path.join(args.checkpoint_dir, args.log_dir +'.txt')

logger = set_logger(log_txt)
logger.info("==========\nArgs:{}\n==========".format(args))


np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed_all(args.manual_seed)


trainset = torchvision.datasets.ImageFolder(
    args.traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]))

testset = torchvision.datasets.ImageFolder(
    args.valdir, 
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
]))

NUM_CLASSES = len(set(trainset.classes))

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

from torch.utils.data import default_collate

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


if args.hybridmix:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.num_workers,
                                            collate_fn=collate_fn)
else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=args.num_workers)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers,
                                          pin_memory=True)

logger.info("Number of train dataset: {}".format(len(trainloader.dataset)))
logger.info("Number of validation dataset: {}".format(len(testloader.dataset)))
logger.info("Number of classes: {}".format(len(set(trainloader.dataset.classes))))
# num_classes = len(set(trainloader.dataset.classes))
logger.info('==> Building model..')
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
    weights = ViT_B_16_Weights.verify(ViT_B_16_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    del imagenet_model_dict['heads.head.weight']
    del imagenet_model_dict['heads.head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'mobilenet_v3_large':
    from torchvision.models import mobilenet_v3_large
    model = mobilenet_v3_large(num_classes=NUM_CLASSES)
elif args.arch == 'shufflenet_v2_x1_5':
    from torchvision.models import shufflenet_v2_x1_5
    model = shufflenet_v2_x1_5(num_classes=NUM_CLASSES)
elif args.arch == 'swin_b':
    from torchvision.models import swin_b, Swin_B_Weights
    model = swin_b(num_classes=NUM_CLASSES)
    weights = Swin_B_Weights.verify(Swin_B_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'swin_s':
    from torchvision.models import swin_s, Swin_S_Weights
    model = swin_s(num_classes=NUM_CLASSES)
    weights = Swin_S_Weights.verify(Swin_S_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
elif args.arch == 'swin_t':
    from torchvision.models import swin_t, Swin_T_Weights
    model = swin_t(num_classes=NUM_CLASSES)
    weights = Swin_T_Weights.verify(Swin_T_Weights.IMAGENET1K_V1)
    imagenet_model_dict = weights.get_state_dict(check_hash=True)
    # print(imagenet_model_dict.keys())
    del imagenet_model_dict['head.weight']
    del imagenet_model_dict['head.bias']
    model.load_state_dict(imagenet_model_dict, strict=False)
else:
    raise NotImplementedError
# net = model(num_classes=NUM_CLASSES)
net = model
net.eval()
resolution = (1, 3, 224, 224)
logger.info('Arch: %s, Params: %.2fM'
        % (args.arch, cal_param_size(net)/1e6))
del(net)

net = model.cuda()
#net = torch.nn.DataParallel(net)
if len(args.imagenet_pretrained) !=0 and not 'vit' in args.arch and not 'swin' in args.arch:
    initalized_model_dict = net.state_dict()
    imagenet_model_dict = torch.load(args.imagenet_pretrained, map_location=torch.device('cpu'))
    if 'densenet' in args.arch:
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        state_dict = imagenet_model_dict
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        imagenet_model_dict = state_dict
    for key in initalized_model_dict.keys():
        if 'fc' in key or 'classifier' in key or 'num_batches_tracked' in key:
            continue
        initalized_model_dict[key] = imagenet_model_dict[key]
    net.load_state_dict(initalized_model_dict)
    logger.info("Load imagenet pretrained weights!")
cudnn.benchmark = True


def train(epoch, criterion_list, optimizer, print_freq=20):
    train_loss = AverageMeter('train_loss', ':.4e')
    train_loss_cls = AverageMeter('train_loss_cls', ':.4e')

    top1_num = 0
    total = 0

    lr = adjust_lr(optimizer, epoch, args)

    start_time = time.time()
    criterion_ce = criterion_list[0]

    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        batch_start_time = time.time()
        inputs = inputs.float().cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        
        logits = net(inputs)

        loss_cls = criterion_ce(logits, targets)

        loss = loss_cls
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        train_loss_cls.update(loss_cls.item(), inputs.size(0))

        '''
        top1 = correct_num(logits, targets, topk=(1,))[0]
        top1_num += top1
        total += targets.size(0)
        '''

        print('Epoch:{}, batch_idx:{}/{}, lr:{:.5f}, Duration:{:.2f}'.format(
            epoch, batch_idx, len(trainloader), lr, time.time()-batch_start_time))


    acc1 = round((top1_num/total).item(), 4)
    logger.info('Epoch:{}\t lr:{:.4f}\t Duration:{:.3f}'
                '\t Train_loss:{:.5f}'
                '\t Train_loss_cls:{:.5f}'
                .format(epoch, lr, time.time() - start_time,
                        train_loss.avg,
                        train_loss_cls.avg))


def test(epoch, criterion_ce, print_freq=10):
    net.eval()
    global best_acc
    test_loss_cls = AverageMeter('test_loss_cls', ':.4e')

    top1_num = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            batch_start_time = time.time()
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = net(inputs)

            loss_cls = criterion_ce(logits, targets)

            test_loss_cls.update(loss_cls, inputs.size(0))

            top1  = correct_num(logits, targets, topk=(1,))[0]
            top1_num += top1
            total += targets.size(0)

            if batch_idx % print_freq == 0:
                print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}'.format(epoch, batch_idx, len(testloader), time.time()-batch_start_time))

        acc1 = round((top1_num/total).item(), 4)

        logger.info('Test epoch:{}\t Test_loss_cls:{:.5f}\t Test top-1 accuracy:{}'
                    .format(epoch, test_loss_cls.avg, str(acc1)))

    return acc1


if __name__ == '__main__':
    
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion_ce = nn.CrossEntropyLoss()

    if args.evaluate:      
        logger.info('Evaluate pre-trained weights from: {}'.format(args.evaluate_checkpoint))
        checkpoint = torch.load(args.evaluate_checkpoint, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch'] + 1
        test(start_epoch, criterion_ce)
    else:
        trainable_list = nn.ModuleList([])
        trainable_list.append(net)

        data = torch.randn(1, 3, 224, 224).cuda()
        net.eval()
        logits = net(data)


        optimizer = optim.SGD(trainable_list.parameters(),
                              lr=0.1, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_ce)
        criterion_list.cuda()

        if args.resume:
            logger.info('Resume pre-trained weights from: {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch'] + 1

        for epoch in range(start_epoch, args.epochs):
            train(epoch, criterion_list, optimizer)
            acc = test(epoch, criterion_ce)

            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'))

            is_best = False
            if best_acc < acc:
                best_acc = acc
                is_best = True

            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_dir, args.arch + '.pth.tar'),
                                os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'))

        logger.info('Evaluate the best model:')
        logger.info('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.arch + '_best.pth.tar'),
                                map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_ce)

        logger.info('Test top-1 best_accuracy: {}'.format(top1_acc))
