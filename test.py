import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from utils import accuracy, ProgressMeter, AverageMeter, get_default_ImageNet_val_loader
from convnet_utils import switch_deploy_flag, switch_conv_bn_impl, build_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--data', metavar='DIR', type=str, help='path to dataset')
parser.add_argument('mode', metavar='MODE', default='deploy', choices=['train', 'deploy'], help='train or deploy')
parser.add_argument('weights', metavar='WEIGHTS', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ResNet-18')
parser.add_argument('-t', '--blocktype', metavar='BLK', default='OREPA')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--val-batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')

def test():
    args = parser.parse_args()
    args.data = '/disk1/humu.hm/ImageNet/ILSVRC2015/Data/CLS-LOC/' 

    switch_deploy_flag(args.mode == 'deploy')
    switch_conv_bn_impl(args.blocktype)
    model = build_model(args.arch)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if 'hdf5' in args.weights:
        from utils import model_load_hdf5
        model_load_hdf5(model, args.weights)
    elif os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = torch.load(args.weights)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
        model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))


    cudnn.benchmark = True

    # Data loading code
    val_loader = get_default_ImageNet_val_loader(args)
    validate(val_loader, model, criterion, use_gpu)


def validate(val_loader, model, criterion, use_gpu):
    batch_time = AverageMeter('Time', ':6.3f', warm=True)
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        #torch.cuda.synchronize()
        #end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            torch.cuda.synchronize()
            end = time.time()

            output = model(images)

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg




if __name__ == '__main__':
    test()