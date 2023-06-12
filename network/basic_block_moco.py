#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: basic_block.py
@time: 2021/12/16 20:34
'''
import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet34
import torchvision.models as models
from utils.lovasz_loss import lovasz_softmax


class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))




class ResNetMocoEncoder(nn.Module):
    def __init__(self, moco_path):
        super(ResNetMocoEncoder, self).__init__()

        model = models.__dict__['resnet50']()

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            param.requires_grad = False
        checkpoint = torch.load(moco_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                # remove prefix
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        

    def forward(self, x):

        layer1_out = self.conv1(x)
        x = self.bn1(layer1_out)
        x = self.relu(x)
        x = self.maxpool(x)

        layer2_out = self.layer1(x)
        layer3_out = self.layer2(layer2_out)
        layer4_out = self.layer3(layer3_out)

        return layer1_out, layer2_out, layer3_out, layer4_out
    
class ResNetFCNMoco(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, config=None, moco_path='/public/home/wudj/SLidR/weights/moco_v2.pt'):
        super(ResNetFCNMoco, self).__init__()


        self.moco_encoder = ResNetMocoEncoder(moco_path)
        self.hiden_size = config['model_params']['hiden_size']
        # Decoder
            # nn.Conv2d(in1, config["model_n_out"], 1),
            # nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),        
        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(64, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2),
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(256, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=4),
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(512, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=8),
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(1024, self.hiden_size, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=16),
        )
        # self.deconv_layer3 = nn.Sequential(
        #     nn.Conv2d(512, 64, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=4),
        # )
        # self.deconv_layer4 = nn.Sequential(
        #     nn.Conv2d(1024, 64, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, self.hiden_size, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=4),
        # )

    def forward(self, data_dict):
        x = data_dict['img']
        h, w = x.shape[2], x.shape[3]
        if h % 16 != 0 or w % 16 != 0:
            assert False, "invalid input size: {}".format(x.shape)

        # Encoder
        layer1_out, layer2_out, layer3_out, layer4_out = self.moco_encoder(x)

        # Deconv
        layer1_out = self.deconv_layer1(layer1_out)
        layer2_out = self.deconv_layer2(layer2_out)
        layer3_out = self.deconv_layer3(layer3_out)
        layer4_out = self.deconv_layer4(layer4_out)
        # to the same torch.Size([bs, 64, 320, 480])
        data_dict['img_scale2'] = layer1_out
        data_dict['img_scale4'] = layer2_out
        data_dict['img_scale8'] = layer3_out
        data_dict['img_scale16'] = layer4_out

        process_keys = [k for k in data_dict.keys() if k.find('img_scale') != -1]
        img_indices = data_dict['img_indices']

        temp = {k: [] for k in process_keys}

        for i in range(x.shape[0]):
            for k in process_keys:
                temp[k].append(data_dict[k].permute(0, 2, 3, 1)[i][img_indices[i][:, 0], img_indices[i][:, 1]])

        for k in process_keys:
            data_dict[k] = torch.cat(temp[k], 0)

        return data_dict

class Lovasz_loss(nn.Module):
    def __init__(self, ignore=None):
        super(Lovasz_loss, self).__init__()
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, ignore=self.ignore)