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
import pdb


class voxelization(nn.Module):
    def __init__(self, coors_range_xyz, spatial_shape, scale_list, mask_ratio):
        super(voxelization, self).__init__()
        self.spatial_shape = spatial_shape
        self.scale_list = scale_list + [1]
        self.coors_range_xyz = coors_range_xyz
        self.coors_range_xy = torch.Tensor(self.coors_range_xyz)[:2, :]
        self.mask_ratio = mask_ratio


    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = (spatial_shape-1) * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        # (pc - coors_range[0])/((coors_range[1] - coors_range[0])/spatial_shape) -> (coors_range[1] - coors_range[0])/spatial_shape = resolution
        return idx.long()       # .long() = .floor()

    def select_idx(self, coors, spatial_shape):
        voxel_size = (self.coors_range_xy[:, 1] - self.coors_range_xy[:, 0]) / spatial_shape
        device = coors.device
        center_xy = (coors.float() + 0.5)*voxel_size.to(device) + self.coors_range_xy[:, 0].to(device)  # Add 0.5 to get the center of each voxel
        center_xy_distance = torch.linalg.vector_norm(center_xy, dim=1)

        select_30 = center_xy_distance[:] <= 30
        select_30to50 = (center_xy_distance[:] > 30) & (center_xy_distance[:] <= 50)
        select_50 = center_xy_distance[:] > 50

        id_list_all = torch.arange(coors.shape[0])
        
        id_list_select_30 = id_list_all[select_30].tolist()
        id_list_select_30to50 = id_list_all[select_30to50].tolist()
        id_list_select_50 = id_list_all[select_50].tolist()

        random.shuffle(id_list_select_30)       # 必须要.tolist() 如果是torch.tensor的话random.shuffle会有重复提取的问题
        shuffle_id_list_select_30 = torch.tensor(id_list_select_30).to(device)

        random.shuffle(id_list_select_30to50)
        shuffle_id_list_select_30to50 = torch.tensor(id_list_select_30to50).to(device)

        random.shuffle(id_list_select_50)
        shuffle_id_list_select_50 = torch.tensor(id_list_select_50).to(device)

        mask_index = torch.cat((shuffle_id_list_select_30[:int((self.mask_ratio[0]) * len(shuffle_id_list_select_30))],
                                shuffle_id_list_select_30to50[:int((self.mask_ratio[1]) * len(shuffle_id_list_select_30to50))],
                                shuffle_id_list_select_50[:int(self.mask_ratio[2] * len(shuffle_id_list_select_50))]), 0)

        id_list_all = id_list_all.to(device)
        unmask_index = id_list_all[~torch.isin(id_list_all, mask_index)]
        return mask_index, unmask_index

    def forward(self, data_dict):
        pc = data_dict['points'][:, :3]
        # device = pc.device
        # pc = torch.clip(pc, self.coors_range_xyz_[:, 0].to(device), self.coors_range_xyz_[:, 1].to(device))

        for idx, scale in enumerate(self.scale_list):

            xidx = self.sparse_quantize(pc[:, 0], self.coors_range_xyz[0], np.ceil(self.spatial_shape[0] / scale))
            yidx = self.sparse_quantize(pc[:, 1], self.coors_range_xyz[1], np.ceil(self.spatial_shape[1] / scale))
            zidx = self.sparse_quantize(pc[:, 2], self.coors_range_xyz[2], np.ceil(self.spatial_shape[2] / scale))

            bxyz_indx = torch.stack([data_dict['batch_idx'], xidx, yidx, zidx], dim=-1).long()
            unq, unq_inv, unq_cnt = torch.unique(bxyz_indx, return_inverse=True, return_counts=True, dim=0)
            unq_xy = unq.clone()[:, 1:3]
            unq = torch.cat([unq[:, 0:1], unq[:, [3, 2, 1]]], dim=1)    # [bzyx]
            
            mask_index, unmask_index = self.select_idx(unq_xy, np.ceil(self.spatial_shape[:2] / scale))

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
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = coors_range_xyz
        self.PPmodel = nn.Sequential(
            nn.Linear(in_channels + 6, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels)
        )

    def prepare_input(self, point, grid_ind, inv_idx):
        pc_mean = torch_scatter.scatter_mean(point[:, :3], inv_idx, dim=0)[inv_idx]
        nor_pc = point[:, :3] - pc_mean

        coors_range_xyz = torch.Tensor(self.coors_range_xyz)
        cur_grid_size = torch.Tensor(self.spatial_shape)
        crop_range = coors_range_xyz[:, 1] - coors_range_xyz[:, 0]
        intervals = (crop_range / cur_grid_size).to(point.device)
        voxel_centers = grid_ind * intervals + coors_range_xyz[:, 0].to(point.device)
        center_to_point = point[:, :3] - voxel_centers

        pc_feature = torch.cat((point, nor_pc, center_to_point), dim=1) # 4 + 3 + 3
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