import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from backbones.adm import Artifact_Detection_Module
from backbones.resnet import resnet18, resnet34, resnet50
from .pub_mod import *


class Discriminator(nn.Module):
    def __init__(self, max_iter):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(256, 256)
        # self.fc2 = nn.Linear(256, 19)
        self.fc2 = nn.Linear(256, 4)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL(max_iter)

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out


class FASBranch(nn.Module):

    def __init__(self, backbone='resnet50'):
        super(FASBranch, self).__init__()

        self.backbone = backbone

        if backbone == 'resnet18':
            self.base_model = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.base_model = resnet34(pretrained=True)
        elif backbone == 'resnet50':
            self.base_model = resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported Backbone!")

        self.inplanes = self.base_model.out_num_features

        self.adm = Artifact_Detection_Module(self.inplanes)

        # texture/style/contents
        self.input_layer = nn.Sequential(
            self.base_model.conv1,
            self.base_model.bn1,
            self.base_model.relu,
            self.base_model.maxpool
        )
        self.layer1 = self.base_model.layer1
        self.layer2 = self.base_model.layer2
        self.layer3 = self.base_model.layer3
        self.layer4 = self.base_model.layer4
        self.avgpool = self.base_model.avgpool

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.init_layer(self.layer5)

        ada_num = 2
        self.adaIN_layers = nn.ModuleList([ResnetAdaINBlock(256) for i in range(ada_num)])

        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        self.init_layer(self.conv_final)

        self.gamma = nn.Linear(256, 256, bias=False)
        self.beta = nn.Linear(256, 256, bias=False)

        self.FC = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ada_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.init_layer(self.ada_conv1)

        self.ada_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.init_layer(self.ada_conv2)

        self.ada_conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256)
        )
        self.init_layer(self.ada_conv3)

        max_iter = 4000
        self.dis = Discriminator(max_iter)


    def init_layer(self, layer):
        for m in layer.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def cal_gamma_beta(self, x1):
        x1 = self.input_layer(x1)
        x1_1 = self.layer1(x1)
        x1_2 = self.layer2(x1_1)
        x1_3 = self.layer3(x1_2)

        x1_4 = self.layer4(x1_3)
        global_feat = self.avgpool(x1_4)

        x1_5 = self.layer5(x1_3)

        x1_add = x1_1
        x1_add = self.ada_conv1(x1_add) + x1_2
        x1_add = self.ada_conv2(x1_add) + x1_3
        x1_add = self.ada_conv3(x1_add)

        gmp = torch.nn.functional.adaptive_max_pool2d(x1_add, 1)
        gmp_ = self.FC(gmp.view(gmp.shape[0], -1))
        gamma, beta = self.gamma(gmp_), self.beta(gmp_)

        domain_invariant = torch.nn.functional.adaptive_avg_pool2d(x1_5, 1).reshape(x1_5.shape[0], -1)

        return x1_5, gamma, beta, domain_invariant, x1_4, global_feat


    def forward(self, x):
        # x, global_feat = self.base_model(x)
        x1, gamma1, beta1, domain_invariant, x4_1, global_feat_1 = self.cal_gamma_beta(x)
        fea_x1_x1 = x1
        for i in range(len(self.adaIN_layers)):
            fea_x1_x1 = self.adaIN_layers[i](fea_x1_x1, gamma1, beta1)
        fea_x1_x1 = self.conv_final(fea_x1_x1)
        fea_x1_x1 = torch.nn.functional.adaptive_avg_pool2d(fea_x1_x1, 1)
        fea_x1_x1 = fea_x1_x1.reshape(fea_x1_x1.shape[0], -1)

        # location result, confidence of each anchor, final feature map of adm.
        loc1, cof1, adm_final_feat1 = self.adm(x4_1)
        final_cls_feat = global_feat_1 + adm_final_feat1
        final_cls_feat = final_cls_feat.reshape(final_cls_feat.shape[0], -1)

        final_cls_feat = final_cls_feat + fea_x1_x1

        dis_invariant = self.dis(domain_invariant)

        if self.training:
            return loc1, cof1, final_cls_feat, dis_invariant

        return final_cls_feat
