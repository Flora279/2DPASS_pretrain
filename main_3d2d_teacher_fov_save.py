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
from network.spvcnn_teacher import get_model as SPVCNN
from network.spvcnn_teacher import criterion, PartialConsistencyLoss
from network.basic_block import ResNetFCN
from network.xModalKD import xModalKD
from dataloader.pc_dataset import get_SemKITTI_label_name
from dataloader.dataset_3d_test import get_model_class as get_model_class_3d, get_collate_class as get_collate_class_3d
from dataloader.pc_dataset_3d_fov_test import get_pc_model_class as get_pc_model_class_3d
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


        self.student_model_3d = SPVCNN(config)        
        self.teacher_model_3d = SPVCNN(config)       
        self.initialize_teacher()

        self.model_2d = ResNetFCN(
            backbone=config.model_params.backbone_2d,
            pretrained=config.model_params.pretrained2d,
            config=config
        )
        self.fusion = xModalKD(config)        
        self.ignore_label = config['dataset_params']['ignore_label']
        self.train_batch_size = config['dataset_params']['train_data_loader']['batch_size']

        self.student_criterion = criterion(config)
        self.consistent_criterion = PartialConsistencyLoss(num_scales=self.num_scales, scale_list=self.scale_list, ignore_index=0)
        # self.consistent_criterion = PartialConsistencyLoss(ignore_index=0)

        pc_dataset_3d = get_pc_model_class_3d(config['dataset_params']['pc_dataset_type'])
        self.dataset_type_3d = get_model_class_3d(config['dataset_params']['dataset_type'])
        self.train_config = config['dataset_params']['train_data_loader']
        self.val_config = config['dataset_params']['val_data_loader']
        self.train_pt_dataset = pc_dataset_3d(config, data_path=self.train_config['data_path'], imageset='train')
        self.val_pt_dataset = pc_dataset_3d(config, data_path=self.val_config['data_path'], imageset='val')

        self.effective_lr =  config.train_params.base_lr

        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})        
        self.training_step_outputs = {'loss_list': [], 'first_step': None}
        self.eval_step_outputs = {'hist_sum': None}


    def train_dataloader(self):          
        return torch.utils.data.DataLoader(
            dataset=self.dataset_type_3d(self.train_pt_dataset, self.config, self.train_config),
            batch_size=self.train_config["batch_size"],
            collate_fn=get_collate_class_3d(self.config['dataset_params']['collate_type']),
            shuffle=False,
            num_workers=self.train_config["num_workers"],
            pin_memory=True,
            drop_last=False
        )
    
    def _load_save_file(self, save_path):
        self.f = h5py.File(save_path, 'w')
    

    def initialize_teacher(self) -> None:
        self.alpha = 0.99 # TODO: Move to config
        for p in self.teacher_model_3d.parameters(): p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher_model_3d.parameters(), self.student_model_3d.parameters()):
            tp.data.mul_(alpha).add_(sp.data, alpha=1 - alpha)

    def student_forward(self, student_data_dict):
        # 3D network
        student_data_dict = self.student_model_3d(student_data_dict)

        if self.training:
            student_data_dict = self.model_2d(student_data_dict)
            student_data_dict = self.fusion(student_data_dict)

        return student_data_dict
    
    def teacher_forward(self, teacher_data_dict):
        # 3D network
        teacher_data_dict = self.teacher_model_3d(teacher_data_dict)

        return teacher_data_dict
    
    def test_step(self, data_dict, batch_idx):
        data_dict = self.teacher_forward(data_dict)
        vote_logits = data_dict['logits'].cpu()
        conf, pred = torch.max(vote_logits.softmax(1), dim=1)
        conf = conf.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()             
        for i, point_num in enumerate(data_dict['point_num']):
            if i==0:
                start_idx = 0
            else:
                start_idx = end_idx

            end_idx = start_idx+point_num
            conf_sub = conf[start_idx:end_idx]
            pred_sub = pred[start_idx:end_idx]

            key = data_dict['label_path'][i]
            conf_key, pred_key = os.path.join(key, 'conf'), os.path.join(key, 'pred')
            self.f.create_dataset(conf_key, data=conf_sub)
            self.f.create_dataset(pred_key, data=pred_sub)       



if __name__ == '__main__':
    # parameters
    configs = parse_config()
    # setting
    path = '/public/home/wudj/2DPASS_train/logs/SemanticKITTI/2dpass_multiple_fov_teacher/epoch=53-step=17172-val_miou=56.866.ckpt'
    checkpoint = torch.load(path)
    configs = EasyDict(checkpoint['hyper_parameters'])
    
    configs['train_params']['gpus'] = [1]    
    configs['dataset_params']['train_data_loader']['num_workers'] = 20
    configs['dataset_params']['train_data_loader']['batch_size'] = 20
    
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

    my_model._load_save_file(os.path.join('/public/home/wudj/2DPASS_train/output', '3d2d_teacher_fov_results.h5'))

    trainer.test(my_model,
                dataloaders=my_model.train_dataloader()
                )    
    my_model.f.close()
    