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
from network.spvcnn import get_model as SPVCNN
from dataloader.pc_dataset import get_SemKITTI_label_name
from dataloader.dataset_3d import get_model_class, get_collate_class
from dataloader.pc_dataset_3d import get_pc_model_class
from utils.metric_util import fast_hist_crop_torch, per_class_iu
from utils.schedulers import cosine_schedule_with_warmup
from utils.config import parse_config

import warnings
import shutil
warnings.filterwarnings("ignore")


class Backbone3DTrainer(pl.LightningModule):
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

        pc_dataset = get_pc_model_class(config['dataset_params']['pc_dataset_type'])
        self.dataset_type = get_model_class(config['dataset_params']['dataset_type'])
        self.train_config = config['dataset_params']['train_data_loader']
        self.val_config = config['dataset_params']['val_data_loader']
        self.train_pt_dataset = pc_dataset(config, data_path=self.train_config['data_path'], imageset='train')
        self.val_pt_dataset = pc_dataset(config, data_path=self.val_config['data_path'], imageset='val')

        #if self.gpu_num > 1:
        #    self.effective_lr = math.sqrt((self.gpu_num)*(self.train_batch_size)) * config.train_params.base_lr
        #else:
        #    self.effective_lr = math.sqrt(self.train_batch_size) * config.train_params.base_lr
        self.effective_lr = config.train_params.base_lr
        print(self.effective_lr)
        self.save_hyperparameters({'effective_lr': self.effective_lr})        
        self.training_step_outputs = {'loss_list': [], 'first_step': None}
        self.eval_step_outputs = {'loss_list': [], 'hist_sum': None}
        print('self.parameters()', self.parameters())
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
            dataset=self.dataset_type(self.train_pt_dataset, self.config, self.train_config),
            batch_size=self.train_config["batch_size"],
            collate_fn=get_collate_class(self.config['dataset_params']['collate_type']),
            shuffle=self.train_config["shuffle"],
            num_workers=self.train_config["num_workers"],
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.dataset_type(self.val_pt_dataset, self.config, self.val_config, num_vote=1),
            batch_size=self.val_config["batch_size"],
            collate_fn=get_collate_class(self.config['dataset_params']['collate_type']),
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
            print('original iou list', iou)
            print('log folder to save', log_folder)
            val_miou = torch.tensor(np.nanmean(iou) * 100).to(self.device)

            file = open(os.path.join(log_folder, configs.train_params.log_dir, 'val_iou_{0}.txt'.format(configs['train_params']['base_lr'])), "a")
            for classs, class_iou in zip(self.unique_label_str, iou):
                print('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('training step:{0}'.format(self.global_step))
                file.write('%s : %.2f%%' % (classs, class_iou * 100))
                file.write('\n')
            file.write('meanIOU={0}'.format(val_miou))
            file.write('\n')
            file.close()
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

            iou = per_class_iu(hist_sum.cpu().numpy())
            print('Validation per class iou: ')
            for class_name, class_iou in zip(self.unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))            
            print('%s : %.2f%%' % ('miou', np.nanmean(iou) * 100))      

        del hist_sum
        self.eval_step_outputs['loss_list'].clear()
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

    backup_dir = os.path.join(log_folder, configs.train_params.log_dir,
                              'backup_files_%s' % str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    if not configs['test']:
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy('main_3d.py', backup_dir)
        shutil.copy('network/voxel_fea_generator.py', backup_dir)
        shutil.copy('dataloader/pc_dataset_3d.py', backup_dir)
        shutil.copy(configs.config_path, backup_dir)
        # os.system('cp network/base_model.py {}'.format(backup_dir))
        shutil.copy('network/spvcnn.py', backup_dir)

    # reproducibility
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(configs.seed)
    config_path = configs.config_path

    my_model = Backbone3DTrainer(configs)

    #for name, param in my_model.named_parameters():
    #    if name == 'model_3d.spv_enc.3.v_enc.1.layers.4.weight' or name == 'model_3d.spv_enc.2.p_enc.layer_out.2.weight':
    #        print('Initialization stage: name', name, 'param', param)
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
        callbacks=[
            checkpoint,
            LearningRateMonitor(logging_interval='step'),
        ],
        profiler=profiler,
        strategy=strategy,
        use_distributed_sampler=use_distributed_sampler,
        logger=tb_logger,
    )
    '''
    # Loading partial ckpt (exclude deconv)
    # if scale=16, load
    ckpt_path = '/public/home/zhangjy/weaksup/2DPASS/logs/SemanticKITTI/MAE_scale_16_deconv2/[0.9, 0.7, 0.5]/epoch=04-step=765-val_MAE_loss=0.013.ckpt' # mask=[0.9,0.7,0.5](0.688), [0.7,0.5,0.3](0.684)
    ckpt = torch.load(ckpt_path)
    #print('ckpt',ckpt.keys())
    pretrained_dict = ckpt['state_dict']
    #print('ckpt state-dict, includes deconv layer', pretrained_dict.keys())
    model_dict = my_model.state_dict()

    # 1. filter out unnecessary keys
    # skip classifier layer, exclude deconv layer
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)  # 避免keyerror when model_dict存在pretrain_dict没有的key
    # 3. load the new state dict
    my_model.load_state_dict(model_dict)
    # NO NEED to load opitimizer state_dict for pretraining usage
    #for name, param in my_model.named_parameters():
    #    if name == 'model_3d.spv_enc.3.v_enc.1.layers.4.weight' or name == 'model_3d.spv_enc.2.p_enc.layer_out.2.weight':
    #        print('After loading ckpt: name', name, 'param', param)
    '''
    trainer.fit(my_model
                )    
    