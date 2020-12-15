import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from models.resnet_conv1 import resnet50


# RGB Stream (VGG16)
class RGB_Stream(nn.Module):
    def __init__(self):
        super(RGB_Stream, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.toplayer = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2048, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb):
        rgb = self.backbone.relu1(self.backbone.bn1(self.backbone.conv1(rgb)))
        rgb = self.backbone.relu2(self.backbone.bn2(self.backbone.conv2(rgb)))
        rgb = self.backbone.relu3(self.backbone.bn3(self.backbone.conv3(rgb)))
        rgb1 = rgb
        rgb = self.backbone.maxpool(rgb)
        rgb2 = self.backbone.layer1(rgb)
        rgb3 = self.backbone.layer2(rgb2)
        rgb4 = self.backbone.layer3(rgb3)
        rgb5 = self.backbone.layer4(rgb4)
        rgb6 = self.toplayer(rgb5)

        return [rgb1, rgb2, rgb3, rgb4, rgb5, rgb6]


# Depth Stream (VGG16)
class Dep_Stream(nn.Module):
    def __init__(self):
        super(Dep_Stream, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.toplayer = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(2048, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, dep):
        dep = self.backbone.relu1(self.backbone.bn1(self.backbone.conv1(dep)))
        dep = self.backbone.relu2(self.backbone.bn2(self.backbone.conv2(dep)))
        dep = self.backbone.relu3(self.backbone.bn3(self.backbone.conv3(dep)))
        dep1 = dep
        dep = self.backbone.maxpool(dep)
        dep2 = self.backbone.layer1(dep)
        dep3 = self.backbone.layer2(dep2)
        dep4 = self.backbone.layer3(dep3)
        dep5 = self.backbone.layer4(dep4)
        dep6 = self.toplayer(dep5)
        return [dep1, dep2, dep3, dep4, dep5, dep6]


class Pred_Layer(nn.Module):
    def __init__(self, in_c=32):
        super(Pred_Layer, self).__init__()
        self.enlayer = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.outlayer = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        x = self.enlayer(x)
        x = self.outlayer(x)
        return x


# BAM
class BAM(nn.Module):
    def __init__(self, in_c):
        super(BAM, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32 * 2)

    def forward(self, rgb_feat, dep_feat, pred):
        feat = torch.cat((rgb_feat, dep_feat), 1)
        feat = self.reduce(feat)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return new_pred


# FF
class FF(nn.Module):
    def __init__(self, in_c):
        super(FF, self).__init__()
        self.reduce = nn.Conv2d(in_c, 32, 1)
        self.ff_conv = nn.Sequential(
            nn.Conv2d(32, 32, k, 1, k // 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, rgb_feat, dep_feat, pred):
        feat = torch.cat((rgb_feat, dep_feat), 1)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        ff_feat = self.ff_conv(feat * pred)
        new_pred = self.rgbd_pred_layer(ff_feat)
        return new_pred


# BF
class BF(nn.Module):
    def __init__(self, in_c):
        super(BF, self).__init__()
        self.reduce = nn.Conv2d(in_c * 2, 32, 1)
        self.bf_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.rgbd_pred_layer = Pred_Layer(32)

    def forward(self, rgb_feat, dep_feat, pred):
        feat = torch.cat((rgb_feat, dep_feat), 1)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_pred = self.rgbd_pred_layer(bf_feat)
        return new_pred


# ASPP for MBAM
class ASPP(nn.Module):
    def __init__(self, in_c):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_c * 2, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_c * 2, 32, 3, 1, padding=3, dilation=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_c * 2, 32, 3, 1, padding=5, dilation=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_c * 2, 32, 3, 1, padding=7, dilation=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


# MBAM
class MBAM(nn.Module):
    def __init__(self, in_c):
        super(MBAM, self).__init__()
        self.ff_conv = ASPP(in_c)
        self.bf_conv = ASPP(in_c)
        self.rgbd_pred_layer = Pred_Layer(32 * 8)

    def forward(self, rgb_feat, dep_feat, pred):
        feat = torch.cat((rgb_feat, dep_feat), 1)
        [_, _, H, W] = feat.size()
        pred = torch.sigmoid(
            F.interpolate(pred,
                          size=(H, W),
                          mode='bilinear',
                          align_corners=True))

        ff_feat = self.ff_conv(feat * pred)
        bf_feat = self.bf_conv(feat * (1 - pred))
        new_pred = self.rgbd_pred_layer(torch.cat((ff_feat, bf_feat), 1))
        return new_pred


class BiANet(nn.Module):
    def __init__(self):
        super(BiANet, self).__init__()

        # two-streams
        self.rgb_stream = RGB_Stream()
        self.dep_stream = Dep_Stream()

        # Global Pred
        self.rgb_global = Pred_Layer(32)
        self.dep_global = Pred_Layer(32)
        self.rgbd_global = Pred_Layer(32 * 2)

        # Shor-Conection
        self.bams = nn.ModuleList([
            BAM(128),
            BAM(256),
            MBAM(512),
            MBAM(1024),
            MBAM(2048),
        ])

    def _upsample_add(self, x, y):
        [_, _, H, W] = y.size()
        return F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, rgb, dep):
        [_, _, H, W] = rgb.size()
        rgb_feats = self.rgb_stream(rgb)
        dep_feats = self.dep_stream(dep)

        # Gloabl Prediction
        rgb_pred = self.rgb_global(rgb_feats[5])
        dep_pred = self.dep_global(dep_feats[5])
        rgbd_pred = self.rgbd_global(torch.cat((rgb_feats[5], dep_feats[5]),
                                               1))
        preds = [
            torch.sigmoid(
                F.interpolate(rgb_pred,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)),
            torch.sigmoid(
                F.interpolate(dep_pred,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)),
            torch.sigmoid(
                F.interpolate(rgbd_pred,
                              size=(H, W),
                              mode='bilinear',
                              align_corners=True)),
        ]

        p = rgbd_pred
        for idx in [4, 3, 2, 1, 0]:
            _p = self.bams[idx](rgb_feats[idx], dep_feats[idx], p)
            p = self._upsample_add(p, _p)
            preds.append(
                torch.sigmoid(
                    F.interpolate(p,
                                  size=(H, W),
                                  mode='bilinear',
                                  align_corners=True)))
        return preds
