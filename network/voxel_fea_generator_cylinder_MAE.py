#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: voxel_fea_generator.py
@time: 2021/8/4 13:36
'''
import torch, random
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
    def __init__(self, coors_range_xyz, spatial_shape, scale_list, mask_ratio):
        super(voxelization, self).__init__()
        self.register_buffer('scale_list', torch.tensor(scale_list + [1]))
        self.register_buffer('coors_range_xyz', torch.tensor(coors_range_xyz))
        self.register_buffer('crop_range', torch.tensor(coors_range_xyz[:, 1] - coors_range_xyz[:, 0]))
        self.register_buffer('spatial_shape', torch.tensor(spatial_shape))
        self.coors_range_r = self.coors_range_xyz[0]
        self.mask_ratio = mask_ratio

    def select_idx(self, coors_r, spatial_shape_r):
        # coors_range_rphi = [[0,50],[-pi,pi]]. coors_r: cylinder grid ind N*1, only know radius coordinate is enough

        #print('self.coors_range_r [0,50]?', self.coors_range_r)
        #print('spatial_shape_r 480?', spatial_shape_r)
        voxel_size_r = (self.coors_range_r[1] - self.coors_range_r[0]) / spatial_shape_r    #voxel size in polar coors, in radius, radius = distance, satial_shape_r=480
        device = coors_r.device
        #print('max coors_r = 480?', torch.max(coors_r))
        # Add 0.5 to get the center of each voxel, self.coors_range_r[0]=0
        center_xy_distance = (coors_r.float() + 0.5) * voxel_size_r.to(device) + self.coors_range_r[0].to(device)

        #print('voxel center r = voxel center dist', center_xy_distance)
        #print('max dist = 50?', torch.max(center_xy_distance))
        select_30 = center_xy_distance[:] <= 20
        select_30to50 = (center_xy_distance[:] > 20) & (center_xy_distance[:] <= 40)
        select_50 = center_xy_distance[:] > 40

        id_list_all = torch.arange(coors_r.shape[0])

        id_list_select_30 = id_list_all[select_30].tolist()
        id_list_select_30to50 = id_list_all[select_30to50].tolist()
        id_list_select_50 = id_list_all[select_50].tolist()

        random.shuffle(id_list_select_30)  # 必须要.tolist() 如果是torch.tensor的话random.shuffle会有重复提取的问题
        shuffle_id_list_select_30 = torch.tensor(id_list_select_30).to(device)

        random.shuffle(id_list_select_30to50)
        shuffle_id_list_select_30to50 = torch.tensor(id_list_select_30to50).to(device)

        random.shuffle(id_list_select_50)
        shuffle_id_list_select_50 = torch.tensor(id_list_select_50).to(device)

        mask_index = torch.cat((shuffle_id_list_select_30[:int((self.mask_ratio[0]) * len(shuffle_id_list_select_30))],
                                shuffle_id_list_select_30to50[
                                :int((self.mask_ratio[1]) * len(shuffle_id_list_select_30to50))],
                                shuffle_id_list_select_50[:int(self.mask_ratio[2] * len(shuffle_id_list_select_50))]),
                               0)

        id_list_all = id_list_all.to(device)
        unmask_index = id_list_all[~torch.isin(id_list_all, mask_index)]
        return mask_index, unmask_index

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]     # all points in current scan
        xyz_pol = cart2polar(pc)

        for idx, scale in enumerate(self.scale_list):
            # cylinder grid ind
            #print('self.coors_range_xyz', self.coors_range_xyz)
            #print('xyz_pol', xyz_pol, 'min -3.14?', torch.min(xyz_pol[:,1]))
            grid_ind = (torch.floor((torch.clip(xyz_pol, self.coors_range_xyz[:, 0], self.coors_range_xyz[:, 1]) - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)-1))).long()
            # grid_ind = (torch.floor((torch.clip(xyz_pol, self.coors_range_xyz[:, 0], self.coors_range_xyz[:, 1]) - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)))).long()
            # grid_ind = (torch.floor((xyz_pol - self.coors_range_xyz[:, 0]) / self.crop_range *(torch.ceil(self.spatial_shape / scale)-1))).long()
            bxyz_indx = torch.stack([data_dict['batch_idx'], grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq_r = unq.clone()[:, 1]
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)

            mask_index, unmask_index = self.select_idx(unq_r, torch.ceil(self.spatial_shape[0] / scale))
            data_dict['scale_{}'.format(scale)] = {
                'full_coors': bxyz_indx,
                'coors_inv': unq_inv,
                'coors': unq.type(torch.int32),
                'unmask_index': unmask_index.type(torch.long)
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
        # pc_feature = torch.cat((center_to_point, point_pol, point[:, :2], point[:, -1].reshape(-1, 1)),
        #                       dim=1)  # 3 + 3 + 2 + sig: cylinder feat

        return pc_feature

    def forward(self, data_dict):
        pt_fea = self.prepare_input(
            data_dict['points'],
            data_dict['scale_1']['full_coors'][:, 1:],
            data_dict['scale_1']['coors_inv']
        )
        pt_fea = self.PPmodel(pt_fea)

        features = torch_scatter.scatter_mean(pt_fea, data_dict['scale_1']['coors_inv'], dim=0)
        coors = data_dict['scale_1']['coors'].int()
        data_dict['sparse_tensor'] = spconv.SparseConvTensor(
            features=features,
            indices=coors,
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )


        unmask_index = data_dict['scale_1']['unmask_index']
        # partial_feature, partial_coors = features.clone()[unmask_index, :], coors.clone()[unmask_index, :]
        partial_feature, partial_coors = features[unmask_index, :], coors[unmask_index, :]


        # voxel_features_partial = voxel_features_partial.data.cpu().cuda()
        # voxel_coords_partial = voxel_coords_partial.data.cpu().cuda()
        data_dict['partial_sparse_tensor'] = spconv.SparseConvTensor(
            features=partial_feature,
            indices=partial_coors,
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        voxel_fratures_all_one = torch.ones(coors.shape[0], 1).to(features.device)
        data_dict['input_sp_tensor_ones'] = spconv.SparseConvTensor(
            features=voxel_fratures_all_one,
            indices=coors,
            spatial_shape=np.int32(self.spatial_shape)[::-1].tolist(),
            batch_size=data_dict['batch_size']
        )

        data_dict['coors'] = data_dict['scale_1']['coors']
        data_dict['coors_inv'] = data_dict['scale_1']['coors_inv']
        data_dict['full_coors'] = data_dict['scale_1']['full_coors']

        return data_dict