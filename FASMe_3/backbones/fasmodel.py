import torch
import torch.nn as nn
from backbones.fasbranch34 import FASBranch34, NUM_DOMAIN
from backbones.fasbranch50 import FASBranch50
import cv2
import numpy as np

class FASModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet34'):
        super(FASModel, self).__init__()

        self.num_classes = num_classes

        if backbone == 'resnet18':
            self.branch1 = FASBranch34(backbone=backbone)
            self.branch2 = FASBranch34(backbone=backbone)
        elif backbone == 'resnet34':
            self.branch1 = FASBranch34(backbone=backbone)
            self.branch2 = FASBranch34(backbone=backbone)
        elif backbone == 'resnet50':
            self.branch1 = FASBranch50(backbone=backbone)
            self.branch2 = FASBranch50(backbone=backbone)
        else:
            raise ValueError("Unsupported Backbone!")

        self.inplanes = 512 * 2
        self.mlp = nn.Sequential(
            nn.Linear(self.inplanes, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        self.init_layer(self.mlp)

        self.mlp_domain = nn.Sequential(
            nn.Linear(NUM_DOMAIN*2, NUM_DOMAIN),
        )
        self.init_layer(self.mlp_domain)

        self.softmax = nn.Softmax(dim=-1)


    def init_layer(self, layer):
        for m in layer.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def cal_feats(self, x):

        xs = torch.split(x, split_size_or_sections=256, dim=3)

        loc0, conf0, feat0, domain_invariant0 = self.branch1(xs[0])
        loc1, conf1, feat1, domain_invariant1 = self.branch2(xs[1])

        final_cls_feat = torch.cat([feat0, feat1], dim=1)

        loc = torch.cat([loc0.unsqueeze(1), loc1.unsqueeze(1)], dim=1)
        conf = torch.cat([conf0.unsqueeze(1), conf1.unsqueeze(1)], dim=1)
        domain_invariant_cat = torch.cat([domain_invariant0, domain_invariant1], dim=1)

        domain_invariant = self.mlp_domain(domain_invariant_cat)

        return final_cls_feat, loc, conf, domain_invariant


    def forward(self, x1, x2):

        if self.training:
            final_feat1, loc, conf, domain_invariant = self.cal_feats(x1)
            final_feat2, _, _, _ = self.cal_feats(x2)

            final_cls = self.mlp(final_feat1)

            return loc, conf, domain_invariant, final_cls, final_feat1, final_feat2
        else:
            xs = torch.split(x1, split_size_or_sections=256, dim=3)
            feat0 = self.branch1(xs[0])
            feat1 = self.branch2(xs[1])

            final_cls_feat = torch.cat([feat0, feat1], dim=1)

            final_cls = self.mlp(final_cls_feat)
            # final_cls = self.mlp(final_cls_feat.view(batch_num, -1))
            return final_cls


if __name__ == "__main__":
    model = FASModel(3, backbone='resnet34')

    image0 = cv2.imread('../images/test.jpg')
    image1 = image0.copy()
    image2 = image0.copy()
    image3 = image0.copy()
    image4 = image0.copy()

    image = cv2.hconcat([image0, image1, image2, image3, image4])
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)

    img = torch.Tensor(image)

    img = torch.randn(2, 3, 256, 1280)

    # model.train()
    # loc, conf, domain_invariant, final_cls, final_feat1, final_feat2 = model(img, img)
    model.eval()
    final_cls = model(img, img)
    final_cls = final_cls


