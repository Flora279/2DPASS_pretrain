import torch, yaml, json, os
import numpy as np
import random, math
import torch.nn as nn
import torch.nn.functional as F

from network.basic_block import Lovasz_loss
from network.spvcnn import get_model as SPVCNN

import pytorch_lightning as pl

from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from utils.metric_util import fast_hist_crop_torch, per_class_iu
from utils.schedulers import cosine_schedule_with_warmup
from dataloader.pc_dataset import get_SemKITTI_label_name

class get_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        seed=123 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.save_hyperparameters(config)   

        self.baseline_only = config.baseline_only
        self.num_classes = config.model_params.num_classes
        self.hiden_size = config.model_params.hiden_size
        self.lambda_seg2d = config.train_params.lambda_seg2d
        self.lambda_xm = config.train_params.lambda_xm
        self.scale_list = config.model_params.scale_list
        self.num_scales = len(self.scale_list)
        self.gpu_num = len(config.train_params.gpus)
        self.config = config
        SemKITTI_label_name = get_SemKITTI_label_name(config['dataset_params']["label_mapping"])
        self.unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
        self.unique_label_str = [SemKITTI_label_name[x] for x in self.unique_label + 1]


        self.model_3d = SPVCNN(config)        
        self.ignore_label = config['dataset_params']['ignore_label']
        self.train_batch_size = config['dataset_params']['train_data_loader']['batch_size']

        if self.gpu_num > 1:
            self.effective_lr = math.sqrt((self.gpu_num)*(self.train_batch_size)) * config.train_params.base_lr
        else:
            self.effective_lr = math.sqrt(self.train_batch_size) * config.train_params.base_lr

        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})        
        self.training_step_outputs = {'loss_list': [], 'first_step': None}
        self.eval_step_outputs = {'loss_list': [], 'hist_sum': None}

    def setup(self, stage=None):
        """
        make datasets & seeding each worker separately
        """
        ##############################################
        # NEED TO SET THE SEED SEPARATELY HERE
        seed = self.global_rank
       
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print('local seed:', seed)
        ##############################################      

    def configure_optimizers(self):
        if self.config['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.effective_lr)
        elif self.config['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.effective_lr,
                                        momentum=self.config['train_params']["momentum"],
                                        weight_decay=self.config['train_params']["weight_decay"],
                                        nesterov=self.config['train_params']["nesterov"])
        else:
            raise NotImplementedError

        if self.config['train_params']["lr_scheduler"] == 'StepLR':
            lr_scheduler = StepLR(
                optimizer,
                step_size=self.config['train_params']["decay_step"],
                gamma=self.config['train_params']["decay_rate"]
            )
        elif self.config['train_params']["lr_scheduler"] == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.config['train_params']["decay_rate"],
                patience=self.config['train_params']["decay_step"],
                verbose=True
            )
        elif self.config['train_params']["lr_scheduler"] == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['train_params']['max_num_epochs'] - 4,
                eta_min=1e-5,
            )
        elif self.config['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts':
            from functools import partial
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=partial(
                    cosine_schedule_with_warmup,
                    num_epochs=self.config['train_params']['max_num_epochs'],
                    batch_size=self.config['dataset_params']['train_data_loader']['batch_size'],
                    dataset_size=self.config['dataset_params']['training_size'],
                    num_gpu=self.gpu_num
                ),
            )
        else:
            raise NotImplementedError

        scheduler = {
            'scheduler': lr_scheduler,
            'interval': 'step' if self.config['train_params']["lr_scheduler"] == 'CosineAnnealingWarmRestarts' else 'epoch',
            'frequency': 1
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': self.config.monitor,
        }

    def forward(self, data_dict):
        # 3D network
        data_dict = self.model_3d(data_dict)

        return data_dict

    def training_step(self, data_dict, batch_idx):
        data_dict = self.forward(data_dict)
        loss = data_dict['loss']
        self.training_step_outputs['loss_list'].append(loss.cpu().detach())
        if self.training_step_outputs['first_step'] == None:
            self.training_step_outputs['first_step'] = self.global_step

        if self.global_rank == 0:
            self.log('train_loss_step', loss, prog_bar=True, rank_zero_only=True, logger=False) 

        return loss
    
    def on_train_epoch_end(self):
        loss_list = torch.stack(self.training_step_outputs['loss_list']).to(self.device)
        loss_list = self.all_gather(loss_list)
        if self.gpu_num > 1:
            loss_step_mean = loss_list.mean(dim=0)
        else:
            loss_step_mean = loss_list

        if self.global_rank == 0:
            first_step = self.training_step_outputs['first_step']
            for loss, step in zip(loss_step_mean, range(first_step, first_step+len(loss_step_mean))):
                self.logger.experiment.add_scalar('train_loss_step', loss, step)   
            
            self.logger.experiment.add_scalar('train_loss_epoch', loss_step_mean.mean(), self.current_epoch)

        del loss_list, loss_step_mean
        self.training_step_outputs['loss_list'].clear()
        self.training_step_outputs['first_step'] = None

    def eval_share_step(self, data_dict, batch_idx):
        indices = data_dict['indices']
        raw_labels = data_dict['raw_labels'].squeeze(1).cpu()
        origin_len = data_dict['origin_len']
        vote_logits = torch.zeros((len(raw_labels), self.num_classes))
        data_dict = self.forward(data_dict)
        vote_logits = data_dict['logits'].cpu()
        raw_labels = data_dict['labels'].squeeze(0).cpu()
        prediction = vote_logits.argmax(1)

        if self.ignore_label != 0:
            prediction = prediction[raw_labels != self.ignore_label]
            raw_labels = raw_labels[raw_labels != self.ignore_label]
            prediction += 1
            raw_labels += 1

        loss = data_dict['loss']
        self.eval_step_outputs['loss_list'].append(loss.cpu().detach())

        hist_sum = fast_hist_crop_torch(prediction, raw_labels, self.unique_label).cpu().detach()

        if self.eval_step_outputs['hist_sum'] == None:
            self.eval_step_outputs['hist_sum'] = hist_sum
        else:
            self.eval_step_outputs['hist_sum'] = sum([self.eval_step_outputs['hist_sum'], hist_sum])        

    def validation_step(self, batch, batch_idx):
        self.eval_share_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        loss_mean = torch.stack(self.eval_step_outputs['loss_list']).to(self.device).mean()
        loss_mean = self.all_gather(loss_mean).mean()

        hist_sum = self.eval_step_outputs['hist_sum'].to(self.device)
        if self.gpu_num > 1:
            hist_sum = sum(self.all_gather(hist_sum))

        if self.global_rank == 0:
            self.log('val_loss', loss_mean, prog_bar=True, logger=False)
            self.logger.experiment.add_scalar('val_loss', loss_mean, self.global_step)   

            iou = per_class_iu(hist_sum.cpu().numpy())
            val_miou = torch.tensor(np.nanmean(iou) * 100).to(self.device)  
            self.logger.experiment.add_scalar('val_miou', val_miou, self.global_step)   
        else:
            val_miou = torch.tensor(0).to(self.device)  
        
        if self.gpu_num > 1:
            val_miou = sum(self.all_gather(val_miou))

        self.log('val_miou', val_miou, prog_bar=True, logger=False)

        del loss_mean, hist_sum
        self.eval_step_outputs['loss_list'].clear()
        self.eval_step_outputs['hist_sum'] = None

    def test_step(self, batch, batch_idx):
        self.eval_share_step(batch, batch_idx)

    def on_test_epoch_end(self):
        hist_sum = self.eval_step_outputs['hist_sum'].to(self.device)
        if self.gpu_num > 1:
            hist_sum = sum(self.all_gather(hist_sum))

        if self.global_rank == 0:

            iou = per_class_iu(hist_sum.numpy())
            print('Validation per class iou: ')
            for class_name, class_iou in zip(self.unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))            
            print('%s : %.2f%%' % ('miou', np.nanmean(iou) * 100))      

        del hist_sum
        self.eval_step_outputs['loss_list'].clear()
        self.eval_step_outputs['hist_sum'] = None