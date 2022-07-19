import os, sys, pdb
import argparse
from models import get_model
from data import make_data_loader
import warnings
from trainer import Trainer
# from trainer_optimizer import Trainer
import torch
import torch.backends.cudnn as cudnn
import random
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./dataset/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', help='number of epochs to change learning rate')  
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--num_classes', default=20, type=int, help='the number of the classses')
parser.add_argument('-o', '--optimizer', default='SGD', type=str, help="The optimizer can be only chosen from {\'SGD\', \'Adam\', \'AdamW\'} for now. More may be implemented later")
parser.add_argument('-backbone','--backbone', default='ResNet101', type=str, help='ResNet101, resnet101, ResNeXt50-swsl, ResNeXt50_32x4d (default: ResNet101)')
# parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--warmup_epoch',  default=2, type=int, help='WarmUp epoch')
parser.add_argument('-up','--warmup_scheduler', action='store_true', default=False, help='star WarmUp')
parser.add_argument('--word_feature_path', default='./wordfeature/', type=str, help='word feature path')


''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014, VOC2007, VOC2012, VG_100K, CoCoDataset, nuswide, mirflickr25k')
parser.add_argument('--model_name', type=str, default='ADD_GCN')
parser.add_argument('--save_dir', default='./checkpoint/COCO2014/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

''' display '''
parser.add_argument('-d', '--display', dest='display', action='store_true', help='display mAP')
parser.add_argument('-s','--summary_writer', action='store_true',  default=False, help="start tensorboard")


def main(args):
    # if args.seed is not None and args.resume is None:
    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    is_train = True if not args.evaluate else False
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=is_train)
    if is_train == True:
        args.iter_per_epoch = len(train_loader)
    else:
        args.iter_per_epoch = 1000 # randm

    args.num_classes = num_classes

    model = get_model(num_classes, args)

    criterion = torch.nn.MultiLabelSoftMarginLoss()

    trainer = Trainer(model, criterion, train_loader, val_loader, args)

    if is_train:
        trainer.train()
    else:
        trainer.validate()

if __name__ == "__main__":
    args = parser.parse_args()
    args.data_root_dir='/media/mlldiskSSD/MLICdataset'
    model_name = {1:'MLGCN'}
    dataset_name = {1:'COCO2014', 2:'VOC2007', 3:'VOC2012'}
    backbone = {1:'ResNet101'}
    args.model_name = model_name[1]  # model name
    args.data = dataset_name[2]
    args.backbone = backbone[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.seed = 1 # seed
    args.epochs = 50 #
    args.optimizer = {1:'SGD', 2: 'Adam', 3:'AdamW'}[1]
    args.display_interval = 1000
    args.batch_size = gpu_num * 16
    args.warmup_scheduler = {1: False, 2: True}[2]
    args.warmup_epoch = 0 if args.warmup_scheduler == False else args.warmup_epoch
    args.word_feature_path = os.path.join(os.getcwd(), 'wordfeature')

    if args.optimizer == 'SGD':
        args.lr = 0.01 # voc is 0.01 and coco is 0.05
        args.lrp = 0.1
        args.epoch_step = [25, 35] # cutout

    elif args.optimizer == 'Adam':
        args.lr = 5 * 1e-5
        args.lrp = 0.1
        args.weight_decay = .0

    elif args.optimizer == 'AdamW':
        args.lr = 5 * 1e-5
        args.lrp = 0.01
        args.weight_decay = 1e-4
        args.epoch_step = [10, 20]

    work = 'SGD_COCO_lr_001_lrp_01_bs16_5'
    args.save_dir = './checkpoint/' + args.data + '/' +args.model_name+'/' + work

    args.evaluate = {1: False, 2: True}[1]
    if args.evaluate == True:
        args.image_size = 576
        args.resume = './checkpoint/COCO2014/checkpoint_best.pth'
    else:
        args.image_size = 448
        args.resume=''

    args.batch_size = 16


    main(args)


