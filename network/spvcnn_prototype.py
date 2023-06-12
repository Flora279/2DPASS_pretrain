#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: spvcnn.py
@time: 2021/12/16 22:41
'''
import torch
import torch_scatter
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.basic_block import Lovasz_loss
import pytorch_lightning as pl
from network.basic_block import SparseBasicBlock
from network.voxel_fea_generator import voxel_3d_generator, voxelization


class point_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super(point_encoder, self).__init__()
        self.scale = scale
        self.layer_in = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(out_channels // 2),
            nn.Linear(out_channels // 2, out_channels),
            nn.LeakyReLU(0.1, True),
        )
        self.layer_out = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Linear(out_channels, out_channels))

    @staticmethod
    def downsample(coors, p_fea, scale=2):
        batch = coors[:, 0:1]
        coors = coors[:, 1:] // scale
        inv = torch.unique(torch.cat([batch, coors], 1), return_inverse=True, dim=0)[1]
        return torch_scatter.scatter_mean(p_fea, inv, dim=0), inv

    def forward(self, features, data_dict):
        output, inv = self.downsample(data_dict['coors'], features)
        identity = self.layer_in(features)
        output = self.PPmodel(output)[inv]
        output = torch.cat([identity, output], dim=1)

        v_feat = torch_scatter.scatter_mean(
            self.layer_out(output[data_dict['coors_inv']]),
            data_dict['scale_{}'.format(self.scale)]['coors_inv'],
            dim=0
        )
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']

        return v_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape):
        super(SPVBlock, self).__init__()
        self.scale = scale
        self.indice_key = indice_key
        self.layer_id = indice_key.split('_')[1]
        self.last_scale = last_scale
        self.spatial_shape = spatial_shape
        self.v_enc = spconv.SparseSequential(
            SparseBasicBlock(in_channels, out_channels, self.indice_key),
            SparseBasicBlock(out_channels, out_channels, self.indice_key),
        )
        self.p_enc = point_encoder(in_channels, out_channels, scale)

    def forward(self, data_dict):
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']

        # voxel encoder
        v_fea = self.v_enc(data_dict['sparse_tensor'])
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        # point encoder
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features+v_fea.features,
            data_dict=data_dict
        )

        # fusion and pooling
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=p_fea+v_fea_inv,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv]


class get_model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.input_dims = config['model_params']['input_dims']
        self.hiden_size = config['model_params']['hiden_size']
        self.num_classes = config['model_params']['num_classes']
        self.scale_list = config['model_params']['scale_list']
        self.num_scales = len(self.scale_list)
        min_volume_space = config['dataset_params']['min_volume_space']
        max_volume_space = config['dataset_params']['max_volume_space']
        self.coors_range_xyz = [[min_volume_space[0], max_volume_space[0]],
                                [min_volume_space[1], max_volume_space[1]],
                                [min_volume_space[2], max_volume_space[2]]]
        self.spatial_shape = np.array(config['model_params']['spatial_shape'])
        self.strides = [int(scale / self.scale_list[0]) for scale in self.scale_list]

        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape
        )

        # encoder layers
        self.spv_enc = nn.ModuleList()
        for i in range(self.num_scales):
            self.spv_enc.append(SPVBlock(
                in_channels=self.hiden_size,
                out_channels=self.hiden_size,
                indice_key='spv_'+ str(i),
                scale=self.scale_list[i],
                last_scale=self.scale_list[i-1] if i > 0 else 1,
                # spatial_shape=np.int32(self.spatial_shape // self.strides[i]).tolist())
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist())
            )
        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128), # self.hiden_size * self.num_scales #-> 256
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )

        # loss
        self.criterion = criterion(config)

    def forward(self, data_dict, prototype):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)

        data_dict = self.voxel_3d_generator(data_dict)

        enc_feats = []
        for i in range(self.num_scales):
            enc_feats.append(self.spv_enc[i](data_dict))

        output = torch.cat(enc_feats, dim=1)
        data_dict['logits'] = self.classifier(output)
        data_dict['embeddings'] = F.normalize(output, p=2, dim=1)

        data_dict = self.criterion(data_dict, prototype)

        return data_dict


class criterion(nn.Module):
    def __init__(self, config):
        super(criterion, self).__init__()
        self.config = config
        self.lambda_lovasz = self.config['train_params'].get('lambda_lovasz', 0.1)
        self.lambda_prototype = self.config['train_params']['lambda_prototype']
        if 'seg_labelweights' in config['dataset_params']:
            seg_num_per_class = config['dataset_params']['seg_labelweights']
            seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
            seg_labelweights = torch.Tensor(np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0))
        else:
            seg_labelweights = None

        self.ce_loss = nn.CrossEntropyLoss(
            weight=seg_labelweights,
            ignore_index=config['dataset_params']['ignore_label']
        )
        self.lovasz_loss = Lovasz_loss(
            ignore=config['dataset_params']['ignore_label']
        )
        self.num_class = config['model_params']['num_classes']
        hiden_size = config['model_params']['hiden_size']
        num_scales = len(config['model_params']['scale_list'])
        self.embedding_size = hiden_size * num_scales #128*4-> 256        

    def forward(self, data_dict, prototype):
        loss_main_ce = self.ce_loss(data_dict['logits'], data_dict['labels'].long())
        loss_main_lovasz = self.lovasz_loss(F.softmax(data_dict['logits'], dim=1), data_dict['labels'].long())
        loss_prototype, step_embedding = self.cal_loss_prototype(data_dict['embeddings'], prototype, data_dict['labels'].long())
        loss_main = loss_main_ce + loss_main_lovasz * torch.tensor(self.lambda_lovasz).type(torch.FloatTensor) \
                    + loss_prototype * torch.tensor(self.lambda_prototype).type(torch.FloatTensor)
        
        data_dict['loss'] = loss_main
        data_dict['step_embedding'] = step_embedding
        del data_dict['embeddings']

        return data_dict

    def cal_loss_prototype(self, point_embedding, prototype, label):
        device = point_embedding.device
        useful_mask = [label != 0]
        point_embedding = point_embedding[useful_mask]
        label = label[useful_mask]        

        point_emb_prototype_sim = torch.exp(torch.matmul(point_embedding, prototype.permute(1, 0))/0.1) 
        # point_embedding (N, 256) * prototype(256, C) -> point_emb_prototype_sim (N, C (category num))
        sim_sum_by_class = point_emb_prototype_sim.sum(axis=1) # N, 1

        label_count_by_class  = torch.bincount(label, minlength=self.num_class) # C,
        weight_by_class = (-1/torch.sqrt(label_count_by_class)) # C,
        weight_by_point = weight_by_class[label] # N,

        label_mask = torch.arange(0, (self.num_class-1)).unsqueeze(0).repeat(point_embedding.shape[0], 1).to(device) == (label-1).unsqueeze(1) # N, C
        # (label-1) -> shift label to 0, self.num_class-1 no need for ignore index
        prototype_loss = (weight_by_point*torch.log(point_emb_prototype_sim[label_mask]/sim_sum_by_class)).mean()

        with torch.no_grad():
            label_mask_tensor_T = label_mask.transpose(0, 1).float() # C, N
            class_embedding_tensor_sum = torch.matmul(label_mask_tensor_T, point_embedding) # (C, N) x (N, 256) -> (C, 256)
            class_embedding_tensor = (class_embedding_tensor_sum/label_count_by_class[1:].repeat(self.embedding_size, 1).transpose(0, 1)) # sum/label_count -> mean
            # label_count_by_class[1:] first one is ignored index

            no_update_prototype_id = torch.where(label_count_by_class==0)[0][1:]-1 
            # [1:] beacuse 0 is always removed by useful_mask at beginning, (label-1) -> shift label to 0
            update_prototype_id = torch.where(label_count_by_class!=0)[0]-1

            step_embedding = torch.zeros((self.num_class-1), self.embedding_size).to(device) # self.num_class-1 no need for ignore index
            step_embedding[update_prototype_id] = class_embedding_tensor[update_prototype_id]
            step_embedding[no_update_prototype_id] = prototype[no_update_prototype_id]

        return prototype_loss, step_embedding