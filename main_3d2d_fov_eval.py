#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: main.py
@time: 2021/12/7 22:21
'''

import os, datetime, torch, random, math, h5py
import numpy as np
from easydict import EasyDict

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
from network.spvcnn import get_model as SPVCNN
from network.basic_block import ResNetFCN
from network.xModalKD import xModalKD
from dataloader.pc_dataset import get_SemKITTI_label_name
from dataloader.dataset import get_model_class as get_model_class_3d2d, get_collate_class as get_collate_class_3d2d
from dataloader.dataset_3d import get_model_class as get_model_class_3d, get_collate_class as get_collate_class_3d
from dataloader.pc_dataset_fov import get_pc_model_class as get_pc_model_class_3d2d
from dataloader.pc_dataset_3d import get_pc_model_class as get_pc_model_class_3d
from utils.metric_util import fast_hist_crop_torch, per_class_iu
from utils.schedulers import cosine_schedule_with_warmup
from utils.config import  parse_config

import warnings
warnings.filterwarnings("ignore")


class FusionDistTrainer(pl.LightningModule):
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
        self.model_2d = ResNetFCN(
            backbone=config.model_params.backbone_2d,
            pretrained=config.model_params.pretrained2d,
            config=config
        )
        self.fusion = xModalKD(config)        
        self.ignore_label = config['dataset_params']['ignore_label']
        self.train_batch_size = config['dataset_params']['train_data_loader']['batch_size']

        pc_dataset_3d2d = get_pc_model_class_3d2d(config['dataset_params']['pc_dataset_type'])
        pc_dataset_3d = get_pc_model_class_3d(config['dataset_params']['pc_dataset_type'])
        self.dataset_type_3d2d = get_model_class_3d2d(config['dataset_params']['dataset_type'])
        self.dataset_type_3d = get_model_class_3d(config['dataset_params']['dataset_type'])
        self.train_config = config['dataset_params']['train_data_loader']
        self.val_config = config['dataset_params']['val_data_loader']
        self.train_pt_dataset = pc_dataset_3d2d(config, data_path=self.train_config['data_path'], imageset='train')
        self.val_pt_dataset = pc_dataset_3d(config, data_path=self.val_config['data_path'], imageset='val')

        # if self.gpu_num > 1:
        #     self.effective_lr = math.sqrt((self.gpu_num)*(self.train_batch_size)) * config.train_params.base_lr
        # else:
        #     self.effective_lr = math.sqrt(self.train_batch_size) * config.train_params.base_lr
        self.effective_lr =  config.train_params.base_lr

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

    def train_dataloader(self):          
        return torch.utils.data.DataLoader(
            dataset=self.dataset_type_3d2d(self.train_pt_dataset, self.config, self.train_config),
            batch_size=self.train_config["batch_size"],
            collate_fn=get_collate_class_3d2d(self.config['dataset_params']['collate_type']),
            shuffle=self.train_config["shuffle"],
            num_workers=self.train_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_type_3d(self.val_pt_dataset, self.config, self.val_config, num_vote=1),
            batch_size=self.val_config["batch_size"],
            collate_fn=get_collate_class_3d(self.config['dataset_params']['collate_type']),
            shuffle=self.val_config["shuffle"],
            pin_memory=True,
            num_workers=self.val_config["num_workers"]
        )

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

        if self.training:
            data_dict = self.model_2d(data_dict)
            data_dict = self.fusion(data_dict)

        return data_dict
    
    def test_step(self, batch, batch_idx):
        data_dict = self.forward(data_dict)
        vote_logits = data_dict['logits'].cpu()
        raw_labels = data_dict['labels'].squeeze(0).cpu()
        prediction = vote_logits.argmax(1)
        loss = data_dict['loss']
        self.eval_step_outputs['loss_list'].append(loss.cpu().detach())

        hist_sum = fast_hist_crop_torch(prediction, raw_labels, self.unique_label).cpu().detach()

        if self.eval_step_outputs['hist_sum'] == None:
            self.eval_step_outputs['hist_sum'] = hist_sum
        else:
            self.eval_step_outputs['hist_sum'] = sum([self.eval_step_outputs['hist_sum'], hist_sum])       




if __name__ == '__main__':
    # parameters
    configs = parse_config()

    # setting
    path = '/public/home/wudj/2DPASS_train/logs/SemanticKITTI/2dpass_multiple/epoch=73-step=23532-val_miou=65.106.ckpt'
    checkpoint = torch.load(path)
    configs = EasyDict(checkpoint['hyper_parameters'])
    
    configs['train_params']['gpus'] = [0, 1, 2, 3, 4, 5]    
    configs['dataset_params']['val_data_loader']['num_workers'] = 10
    configs['dataset_params']['val_data_loader']['batch_size'] = 10

    train_params = configs['train_params']
    gpus = train_params['gpus']
    gpu_num = len(gpus)
    print(gpus)

    if train_params['mixed_fp16']:
        precision = '16'
    else:
        precision = '32'


    my_model = FusionDistTrainer.load_from_checkpoint(path, config=configs)

    if gpu_num > 1:
        strategy='ddp'
        use_distributed_sampler = True
    else:
        strategy = 'auto'
        use_distributed_sampler = False

    trainer = pl.Trainer(
        max_epochs=train_params['max_num_epochs'],
        devices=gpus,
        num_nodes=1,
        precision = precision,
        accelerator='gpu',
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler
    )


    trainer.test(my_model,
                dataloaders=my_model.val_dataloader()
                )    
    