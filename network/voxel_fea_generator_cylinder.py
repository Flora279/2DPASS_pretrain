#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: voxel_fea_generator.py
@time: 2021/8/4 13:36
'''
import torch
import torch_scatter
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv

def cart2polar(input_xyz):
    rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = torch.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)    


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * torch.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * torch.sin(input_xyz_polar[1])
    return torch.stack((x, y, input_xyz_polar[2]), axis=0)

class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list):
        super(voxelization, self).__init__()
        self.register_buffer('scale_list', torch.tensor(scale_list + [1]))
        self.register_buffer('coors_range_xyz', torch.tensor(coors_range_xyz))
        self.register_buffer('crop_range', torch.tensor(coors_range_xyz[:, 1] - coors_range_xyz[:, 0]))
        self.register_buffer('spatial_shape', torch.tensor(spatial_shape))

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        xyz_pol = cart2polar(pc)

        for idx, scale in enumerate(self.scale_list):

            grid_ind = (torch.floor((torch.clip(xyz_pol, self.coors_range_xyz[:, 0], self.coors_range_xyz[:, 1]) - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)-1))).long()
            # grid_ind = (torch.floor((torch.clip(xyz_pol, self.coors_range_xyz[:, 0], self.coors_range_xyz[:, 1]) - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)))).long()
            # grid_ind = (torch.floor((xyz_pol - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)-1))).long()
            bxyz_indx = torch.stack([data_dict['batch_idx'], grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32)
            }
        return data_dict


class voxel_3d_generator(nn.Module):
    def __init__(self, in_channels, out_channels, coors_range_xyz, spatial_shape):
        super(voxel_3d_generator, self).__init__()
        self.register_buffer('coors_range_xyz', torch.tensor(coors_range_xyz))
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        # self.register_buffer('intervals', torch.tensor(crop_range / (spatial_shape - 1)))
        self.register_buffer('intervals', torch.tensor(crop_range / (spatial_shape)))
        self.spatial_shape = spatial_shape

        self.PPmodel = nn.Sequential(
            # nn.Linear(in_channels + 9, out_channels),
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        point_pol = cart2polar(point[:, :3])

        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        voxel_centers = (grid_ind.type(torch.float32) + 0.5) * self.intervals + self.coors_range_xyz[:, 0]
        center_to_point = (point_pol - voxel_centers).type(torch.float32)

        # pc_feature = torch.cat((point, point_pol, nor_pc, center_to_point), dim=1) # 4 + 3+ 3 + 3
        pc_feature = torch.cat((point[:, -1].reshape(-1, 1), point_pol, nor_pc, center_to_point), dim=1) # 1 (intensity) + 3 + 3 + 3
        # pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1) # 4 + 3 + 3
        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=data_dict['scale_1']['coors'].int(),
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict