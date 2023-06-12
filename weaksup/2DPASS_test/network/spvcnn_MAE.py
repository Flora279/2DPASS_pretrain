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
from network.voxel_fea_generator_MAE import voxel_3d_generator, voxelization

import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        #print('coors shape', coors.shape, coors)
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
        )        # backproject to point-wise feature -> take mean to obtain new scale voxel feature
        data_dict['coors'] = data_dict['scale_{}'.format(self.scale)]['coors']
        data_dict['coors_inv'] = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        data_dict['full_coors'] = data_dict['scale_{}'.format(self.scale)]['full_coors']
        data_dict['unmask_index'] = data_dict['scale_{}'.format(self.scale)]['unmask_index']
        return v_feat


class SPVBlock(nn.Module):
    def __init__(self, in_channels, out_channels, indice_key, scale, last_scale, spatial_shape, loss_func):
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

        # logits spconv, use the same logit layer for different scale
        # self.logits = spconv.SubMConv3d(in_channels, 1, indice_key=indice_key.replace('spv', 'logit'), kernel_size=3, stride=1,
        #                                     padding=1,
        #                                     bias=True)
        # loss
        self.criterion_mae = loss_func
        self.deconv = nn.ConvTranspose3d(64, 1, 3, padding=1, stride=1, bias=False)

    def forward(self, data_dict):
        # last-scale [1,2,4,8]
        # self.scale [2,4,8,16]
        print('current scale in spvcnn', self.scale, data_dict['sparse_tensor'].spatial_shape)
        coors_inv_last = data_dict['scale_{}'.format(self.last_scale)]['coors_inv']
        coors_inv = data_dict['scale_{}'.format(self.scale)]['coors_inv']
        # self.v_enc为6层SubMConv 不改变shape
        v_fea = self.v_enc(data_dict['sparse_tensor'])      # full voxel feature
        v_fea_inv = torch_scatter.scatter_mean(v_fea.features[coors_inv_last], coors_inv, dim=0)

        v_partial_fea = self.v_enc(data_dict['partial_sparse_tensor'])  # last scale's feature [1,64,60,1000,1000]
        # subManifoldConv
        # sp.spconv
        print('v_partial_fea.dense() shape', v_partial_fea.dense().shape)

        decode_feat = self.deconv(v_partial_fea.dense())
        data_dict['mae_logits'] = decode_feat   # [1,1,60,1000,1000]
        #print('decode feat device', decode_feat.device, decode_feat)
        # loss at each scale
        loss = self.criterion_mae(data_dict['mae_logits'], data_dict['input_sp_tensor_ones'].dense()) #【1,1,60,1000,1000】
        print('loss', loss)
        data_dict['layer_{}'.format(self.layer_id)] = {}
        data_dict['layer_{}'.format(self.layer_id)]['pts_feat'] = v_fea.features
        data_dict['layer_{}'.format(self.layer_id)]['full_coors'] = data_dict['full_coors']

        # point encoder, 更新了coors & coors_inv & full_coors to current scale
        p_fea = self.p_enc(
            features=data_dict['sparse_tensor'].features + v_fea.features,
            data_dict=data_dict
        )  # return current scale's voxel feature (updated) [41141, 64] (current scale N_vox)

        # current scale:
        # fusion and pooling, voxel shape changes & updates here
        full_voxel_feat = p_fea + v_fea_inv
        print('full feature shape', full_voxel_feat.shape)

        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=full_voxel_feat,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )  # 此时的sparse_tensor才对应着self.scale, full voxel feature

        print("data_dict['coors'][data_dict['slect_index'],:]", data_dict['coors'][data_dict['unmask_index'],:].shape)
        data_dict['partial_sparse_tensor'] = spconv.SparseConvTensor(
            features=full_voxel_feat[data_dict['unmask_index'],:],
            indices=data_dict['coors'][data_dict['unmask_index'],:],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )
        nums = data_dict['coors'].shape[0]  # 每轮都在更新
        voxel_features_all_one = torch.ones(nums, 1).to(p_fea.device)

        data_dict['input_sp_tensor_ones'] = spconv.SparseConvTensor(
            features=voxel_features_all_one,
            indices=data_dict['coors'],
            spatial_shape=self.spatial_shape,
            batch_size=data_dict['batch_size']
        )

        return p_fea[coors_inv], loss   # return point-wise feature, 每个scale都做一次voxel MAE loss


class get_model(pl.LightningModule):    # SPVCNN
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
        self.masked_ratio = config['dataset_params']['masked_ratio']
        self.BCE_loss = nn.BCEWithLogitsLoss()


        # voxelization
        self.voxelizer = voxelization(
            coors_range_xyz=self.coors_range_xyz,
            spatial_shape=self.spatial_shape,
            scale_list=self.scale_list
        )

        # input processing
        self.voxel_3d_generator = voxel_3d_generator(
            in_channels=self.input_dims,
            out_channels=self.hiden_size,   # hidden-size=64
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
                spatial_shape=np.int32(self.spatial_shape // self.strides[i])[::-1].tolist(),
            loss_func = self.BCE_loss)
            )

        # decoder layer
        self.classifier = nn.Sequential(
            nn.Linear(self.hiden_size * self.num_scales, 128),
            nn.ReLU(True),
            nn.Linear(128, self.num_classes),
        )


    def forward(self, data_dict):
        with torch.no_grad():
            data_dict = self.voxelizer(data_dict)
        data_dict = self.voxel_3d_generator(data_dict)
        # data_dict: 1. coords: non-empty voxel coords / grid idx
        #            2. sparse_tensor: input voxel features N_vox*9, mean of pts feat
        #print('voxel data_dict[coords] shape', data_dict['coors'].shape) # [68224, 4]

        loss_MAE = torch.tensor(0.0, device=self.device)

        for i in range(self.num_scales):
            scale = self.scale_list[i]   # [2,4,8,16]
            p_fea, loss = self.spv_enc[i](data_dict)   # 里面更新了dict partial feature, coors, coors_inv, etc.
            # v_fea 经过了6层SubMConv3d学习
            loss_MAE += loss   # 每个scale的MAE_loss

        data_dict['loss_MAE'] = loss_MAE
        return data_dict