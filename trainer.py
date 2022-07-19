import os, sys, pdb
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from util import AverageMeter, AveragePrecisionMeter, WarmUpLR
from datetime import datetime
from tqdm import tqdm
import logging

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    def __init__(self, model, criterion, train_loader, val_loader, args):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

    def initialize_optimizer_and_scheduler(self):
        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.get_config_optim(self.args.lr, self.args.lrp),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.get_config_optim(self.args.lr, self.args.lrp),
                                        lr=self.args.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.get_config_optim(self.args.lr, self.args.lrp),
                                        lr=self.args.lr,
                                        betas=(0.9, 0.999),
                                        weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError('Only SGD or Adam can be chosen!')
        if self.args.warmup_scheduler == True:
            self.logger.info("======== Start WarmUp ========")
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.epoch_step, gamma=0.1, last_epoch=-1)
            self.warmup_scheduler = WarmUpLR(self.optimizer, self.args.iter_per_epoch * self.args.warmup_epoch)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.args.epoch_step, gamma=0.1, last_epoch=-1)


    def initialize_meters(self):
        self.meters = {}
        
        # meters
        self.meters['loss'] = AverageMeter('loss')
        self.meters['ap_meter'] = AveragePrecisionMeter()
        
        # time measure
        self.meters['batch_time'] = AverageMeter('batch_time')
        self.meters['data_time'] = AverageMeter('data_time')

    def initialization(self, is_train=False):
        """ initialize self.model and self.criterion here """

        # Bulid Logger
        self.build_logger()

        # Bulid Summary
        if self.args.summary_writer:
            self.summary_writer = SummaryWriter(log_dir=self.args.save_dir)
        
        # show config
        self.show_args(self.args, self.logger)

        if is_train:
            self.start_epoch = 1
            self.epoch = 1
            self.end_epoch = self.args.epochs
            self.best_score = 0.
            self.lr_now = self.args.lr

            # initialize some settings
            self.initialize_optimizer_and_scheduler()

        self.initialize_meters()

        # load checkpoint if args.resume is a valid checkpint file.
        if os.path.isfile(self.args.resume) and self.args.resume.endswith('pth'):
            self.load_checkpoint()
        
        if torch.cuda.is_available():
            cudnn.benchmark = True
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.criterion = self.criterion.cuda()
            self.model.eval()
            torch.cuda.empty_cache()
            self.val_loader.pin_memory = True
            if not self.args.evaluate:
                self.train_loader.pin_memory = True
                

    def build_logger(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        # args.save_dir
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        file_path = os.path.join(self.args.save_dir, '{}_{}.log.txt'.format(self.args.data, self.args.model_name))
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    # show config
    def show_args(self, args, logger):
        logger.info("==========================================")
        logger.info("==========       CONFIG      =============")
        logger.info("==========================================")

        for arg, content in args.__dict__.items():
            logger.info("{}: {}".format(arg, content))

        logger.info("==========================================")
        logger.info("===========        END        ============")
        logger.info("==========================================")

        logger.info("\n")

    def reset_meters(self):
        for k, v in self.meters.items():
            self.meters[k].reset()

    def on_start_epoch(self):
        self.reset_meters()

    def on_end_epoch(self, is_train=False):

        if is_train:
            # maybe you can do something like 'print the training results' here.
            return 
        else:
            # map = self.meters['ap_meter'].value().mean()
            ap =  self.meters['ap_meter'].value()
            map = ap.mean()
            loss = self.meters['loss'].average()
            data_time = self.meters['data_time'].average()
            batch_time = self.meters['batch_time'].average()

            OP, OR, OF1, CP, CR, CF1 = self.meters['ap_meter'].overall()
            OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.meters['ap_meter'].overall_topk(3)

            self.logger.info('* Test')
            if self.args.display is True:
                self.logger.info(ap) 
            self.logger.info('* Loss: {loss:.4f}\t mAP: {map:.4f}\t' 
                    'Data_time: {data_time:.4f}\t Batch_time: {batch_time:.4f}'.format(
                    loss=loss, map=map, data_time=data_time, batch_time=batch_time))
            self.logger.info('OP: {OP:.3f}\t OR: {OR:.3f}\t OF1: {OF1:.3f}\t'
                    'CP: {CP:.3f}\t CR: {CR:.3f}\t CF1: {CF1:.3f}'.format(
                    OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            self.logger.info('OP_3: {OP:.3f}\t OR_3: {OR:.3f}\t OF1_3: {OF1:.3f}\t'
                    'CP_3: {CP:.3f}\t CR_3: {CR:.3f}\t CF1_3: {CF1:.3f}'.format(
                    OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))
            
            # tensorboard logger
            if self.args.summary_writer:
                self.summary_writer.add_scalar('avg_val_loss', loss, self.epoch)
                self.summary_writer.add_scalar('val_mAP', map, self.epoch)            

            return map

    def on_forward(self, inputs, targets, is_train):
        inputs = Variable(inputs).float()
        targets = Variable(targets).float()

        if not is_train:
            with torch.no_grad():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)

        loss = self.criterion(outputs, targets)
        self.meters['loss'].update(loss.item(), inputs.size(0))

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_clip_grad_norm)
            self.optimizer.step()
            if self.args.warmup_scheduler == True and self.epoch <= self.args.warmup_epoch:
                self.warmup_scheduler.step()

        return outputs

    def train(self):
        self.initialization(is_train=True)
        torch.cuda.empty_cache()
        for epoch in range(self.start_epoch, self.end_epoch):

            self.epoch = epoch

            # train for one epoch
            torch.cuda.empty_cache()
            self.run_iteration(self.train_loader, is_train=True)

            # evaluate on validation set
            torch.cuda.empty_cache()
            score = self.run_iteration(self.val_loader, is_train=False)

            # adjust learning rate
            self.lr_scheduler.step()
            self.logger.info('Now lrp:{:.8f}, lr:{:.8f}'.format(self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"]))

            # record best score, save checkpoint and result
            is_best = score > self.best_score
            self.best_score = max(score, self.best_score)
            checkpoint = {
                'epoch': epoch + 1,
                'model_name': self.args.model_name,
                'state_dict': self.model.module.state_dict() if torch.cuda.is_available() else self.model.state_dict(),
                'best_score': self.best_score
                }
            model_dir = self.args.save_dir
            # assert os.path.exists(model_dir) == True
            self.save_checkpoint(checkpoint, model_dir, is_best)
            self.save_result(model_dir, is_best)

            self.logger.info(' * best mAP={best:.4f}'.format(best=self.best_score))
        
        if self.args.summary_writer:
            self.summary_writer.close()

        return self.best_score

    def run_iteration(self, data_loader, is_train=True):
        self.on_start_epoch()
        torch.cuda.empty_cache()
        if not is_train:
            self.logger.info('====== star eval ======')
            # data_loader = tqdm(data_loader, desc='Validate')
            self.model.eval()
        else:
            self.model.train()

        st_time = time.time()
        torch.cuda.empty_cache()
        for i, data in enumerate(data_loader):

            # measure data loading time
            data_time = time.time() - st_time
            self.meters['data_time'].update(data_time)

            # inputs, targets, targets_gt, filenames = self.on_start_batch(data)
            inputs = data['image']
            targets = data['target']

            # import ipdb; ipdb.set_trace()
            # for voc
            labels = targets.clone()

            targets[targets==0] = 1
            targets[targets==-1] = 0

            # import ipdb; ipdb.set_trace()

            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                # inp=inp.cuda()

            outputs = self.on_forward(inputs, targets, is_train=is_train)

            # measure elapsed time
            batch_time = time.time() - st_time
            self.meters['batch_time'].update(batch_time)

            self.meters['ap_meter'].add(outputs.data, labels.data, data['name'])
            st_time = time.time()

            if is_train and i % self.args.display_interval == 0:
                self.logger.info('{}, {} Epoch, lrp:{:.8f}, lr:{:.8f}, {} Iter, Loss: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  self.epoch,
                        self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"], 
                        i, 
                        self.meters['loss'].value(), self.meters['data_time'].value(), 
                        self.meters['batch_time'].value()))
                if self.args.summary_writer:
                    self.summary_writer.add_scalar('train_loss', self.meters['loss'].value(), self.epoch)
                    self.summary_writer.add_scalar('learning_rate', self.optimizer.param_groups[1]["lr"], self.epoch)

            if not is_train and i % self.args.display_interval == 0:
                self.logger.info('{}, {} Epoch, {} Iter, Loss: {:.4f}, Data time: {:.4f}, Batch time: {:.4f}'.format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  self.epoch,
                        i, 
                        self.meters['loss'].value(), self.meters['data_time'].value(), 
                        self.meters['batch_time'].value()))
                if self.args.summary_writer:
                    self.summary_writer.add_scalar('val_loss', self.meters['loss'].value(), self.epoch)

        return self.on_end_epoch(is_train=is_train)

    def validate(self):
        self.initialization(is_train=False)

        map = self.run_iteration(self.val_loader, is_train=False)

        model_dir = os.path.dirname(self.args.resume)
        assert os.path.exists(model_dir) == True
        self.save_result(model_dir, is_best=False)

        return map
    

    def load_checkpoint(self):
        self.logger.info("* Loading checkpoint '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        self.start_epoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        self.best_score = checkpoint['best_score']
        # self.best_score = 0.0
        # for k, v in checkpoint.items():
        #     print(k)
        # checkpoint['state_dict'] = checkpoint['model_state_dict']

        if self.start_epoch + 1 == 40 or self.start_epoch + 1 == 39:
            self.start_epoch = 1
            self.best_score = 0.0
            self.logger.info("=========== start_epoch ===========")
        self.start_epoch = 1
        self.best_score = 0.0
        
        model_dict = self.model.state_dict()
        for k, v in checkpoint['state_dict'].items():
            if k in model_dict and v.shape == model_dict[k].shape:
                model_dict[k] = v
            else:
                self.logger.info('\tMismatched layers: {}'.format(k))
        self.model.load_state_dict(model_dict)
        del checkpoint
        del model_dict
        torch.cuda.empty_cache()


    def save_checkpoint(self, checkpoint, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        filename = 'Epoch-{}.pth'.format(self.epoch)
        # filename = 'checkpoint.pth'
        res_path = os.path.join(model_dir, filename)
        self.logger.info('Save checkpoint to {}'.format(res_path))
        torch.save(checkpoint, res_path)
        if is_best:
            filename_best = 'checkpoint_best.pth'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)


    def save_result(self, model_dir, is_best=False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # filename = 'results.csv' if not is_best else 'best_results.csv'
        filename = 'results.csv'
        res_path = os.path.join(model_dir, filename)
        self.logger.info('Save results to {}'.format(res_path))
        with open(res_path, 'w') as fid:
            for i in range(self.meters['ap_meter'].scores.shape[0]):
                fid.write('{},{},{}\n'.format(self.meters['ap_meter'].filenames[i], 
                    ','.join(map(str,self.meters['ap_meter'].scores[i].numpy())), 
                    ','.join(map(str,self.meters['ap_meter'].targets[i].numpy()))))
        
        if is_best:
            filename_best = 'output_best.csv'
            res_path_best = os.path.join(model_dir, filename_best)
            shutil.copyfile(res_path, res_path_best)