#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: main.py
@time: 2021/12/7 22:21
'''

import os, datetime, torch, random, math
import numpy as np

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
from dataloader.dataset_teacher import get_model_class as get_model_class_3d2d, get_collate_class as get_collate_class_3d2d
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

        pc_dataset_3d2d = get_pc_model_class_3d2d(config['dataset_params']['pc_dataset_type'])
        pc_dataset_3d = get_pc_model_class_3d(config['dataset_params']['pc_dataset_type'])
        self.dataset_type_3d2d = get_model_class_3d2d(config['dataset_params']['dataset_type'])
        self.dataset_type_3d = get_model_class_3d(config['dataset_params']['dataset_type'])
        self.train_config = config['dataset_params']['train_data_loader']
        self.val_config = config['dataset_params']['val_data_loader']
        self.train_pt_dataset = pc_dataset_3d2d(config, data_path=self.train_config['data_path'], imageset='train')
        self.val_pt_dataset = pc_dataset_3d(config, data_path=self.val_config['data_path'], imageset='val')

        self.effective_lr =  config.train_params.base_lr

        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})        
        self.training_step_outputs = {'loss_list': [], 'first_step': None}
        self.eval_step_outputs = {'hist_sum': None}

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
        params = list(self.student_model_3d.parameters()) + list(self.model_2d.parameters()) + list(self.fusion.parameters())

        if self.config['train_params']['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(params,
                                         lr=self.effective_lr)
        elif self.config['train_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(params,
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

    def training_step(self, data_dict, batch_idx):
        self.update_teacher()        
        student_dict = self.student_forward(data_dict['student'])
        teacher_dict = self.teacher_forward(data_dict['teacher'])

        fusion_loss = student_dict['loss']
        student_loss = self.student_criterion(student_dict)
        # consistent_loss = self.consistent_criterion(student_dict['logits'], teacher_dict['logits'], student_dict['labels'].long())
        consistent_loss = self.consistent_criterion(student_dict, teacher_dict, student_dict['labels'].long())
        loss = fusion_loss+ student_loss + consistent_loss

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
        data_dict = self.teacher_forward(data_dict)
        vote_logits = data_dict['logits'].cpu()
        raw_labels = data_dict['labels'].squeeze(0).cpu()
        prediction = vote_logits.argmax(1)

        hist_sum = fast_hist_crop_torch(prediction, raw_labels, self.unique_label).cpu().detach()

        if self.eval_step_outputs['hist_sum'] == None:
            self.eval_step_outputs['hist_sum'] = hist_sum
        else:
            self.eval_step_outputs['hist_sum'] = sum([self.eval_step_outputs['hist_sum'], hist_sum])        

    def validation_step(self, batch, batch_idx):
        self.eval_share_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        hist_sum = self.eval_step_outputs['hist_sum'].to(self.device)
        if self.gpu_num > 1:
            hist_sum = sum(self.all_gather(hist_sum))

        if self.global_rank == 0:

            iou = per_class_iu(hist_sum.cpu().numpy())
            val_miou = torch.tensor(np.nanmean(iou) * 100).to(self.device)  
            self.logger.experiment.add_scalar('val_miou', val_miou, self.global_step)   
        else:
            val_miou = torch.tensor(0).to(self.device)  
        
        if self.gpu_num > 1:
            val_miou = sum(self.all_gather(val_miou))

        self.log('val_miou', val_miou, prog_bar=True, logger=False)

        del hist_sum
        self.eval_step_outputs['hist_sum'] = None


if __name__ == '__main__':
    # parameters
    configs = parse_config()
    train_params = configs['train_params']

    # setting
    gpus = train_params['gpus']
    gpu_num = len(gpus)
    print(gpus)

    if train_params['mixed_fp16']:
        precision = '16'
    else:
        precision = '32'

    # output path
    log_folder = 'logs/' + configs['dataset_params']['pc_dataset_type']
    tb_logger = pl_loggers.TensorBoardLogger(log_folder, name=configs.train_params.log_dir, default_hp_metric=False)
    os.makedirs(f'{log_folder}/{configs.train_params.log_dir}', exist_ok=True)
    profiler = SimpleProfiler(dirpath = f'{log_folder}/{configs.train_params.log_dir}',
                              filename= 'profiler.txt')
    np.set_printoptions(precision=4, suppress=True)

    pl.seed_everything(configs.seed)
    
    checkpoint = ModelCheckpoint(dirpath=f'{log_folder}/{configs.train_params.log_dir}',
                                 save_last=True,
                                 save_top_k=5,
                                 mode='max',
                                 every_n_epochs=1,
                                 filename="{epoch:02d}-{step}-{val_miou:.3f}",
                                 monitor="val_miou")

    # save the backup files
    backup_dir = os.path.join(log_folder, configs.train_params.log_dir, 'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    if not configs['test']:
        os.makedirs(backup_dir, exist_ok=True)
        os.system('cp main_3d2d_teacher_fov.py {}'.format(backup_dir))
        os.system('cp dataloader/dataset_teacher.py {}'.format(backup_dir))
        os.system('cp dataloader/dataset_3d.py {}'.format(backup_dir))
        os.system('cp dataloader/pc_dataset_fov.py {}'.format(backup_dir))
        os.system('cp dataloader/pc_dataset_3d.py {}'.format(backup_dir))
        os.system('cp {} {}'.format(configs.config_path, backup_dir))
        os.system('cp network/spvcnn_teacher.py {}'.format(backup_dir))
        os.system('cp network/basic_block.py {}'.format(backup_dir))
        os.system('cp network/xModalKD.py {}'.format(backup_dir))

    # reproducibility
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(configs.seed)
    config_path = configs.config_path

    my_model = FusionDistTrainer(configs)

    if gpu_num > 1:
        strategy='ddp'
        use_distributed_sampler = True
    else:
        strategy = 'auto'
        use_distributed_sampler = False

    trainer = pl.Trainer(
        # sync_batchnorm=True,
        max_epochs=train_params['max_num_epochs'],
        devices=gpus,
        num_nodes=1,
        precision = precision,
        accelerator='gpu',
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval='step'),
        ],
        profiler=profiler,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        logger=tb_logger,
    )


    # ckpt_path = '/public/home/wudj/Cylinder3D_spconv_v2/logs_multiple_gpu/epoch=27-step=8344-val_miou=61.860.ckpt'
    trainer.fit(my_model
                # ckpt_path=ckpt_path
                )    
    