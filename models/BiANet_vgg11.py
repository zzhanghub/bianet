import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

backbone = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


# VGG16
def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


# VGG16 with Side Outputs
class VGG_Sout(nn.Module):
    def __init__(self, extract=[1, 4, 9, 14, 19]):
        super(VGG_Sout, self).__init__()
        self.vgg = nn.ModuleList(vgg(cfg=backbone['vgg11']))
        self.extract = extract

    def forward(self, x):
        souts = []
        for idx in range(len(self.vgg)):
            x = self.vgg[idx](x)
            if idx in self.extract:
                souts.append(x)

        return souts, x


# Global Sliency (A new block following VGG-16 for predict global saliency map)
class GSLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(GSLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, channel, 1)
        self.convs = nn.Sequential(nn.Conv2d(channel, channel, k, 1, k // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channel, channel, k, 1, k // 2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(channel, channel, k, 1, k // 2),
                                   nn.ReLU(inplace=True))
        self.out_layer = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.convs(x)
        out = self.out_layer(x)
        return out


# Original Attention
class OriAtt(nn.Module):
    def __init__(self):
        super(OriAtt, self).__init__()

    def forward(self, sout, pred):
        return sout.mul(torch.sigmoid(pred))


# Reverse Attention
class RevAtt(nn.Module):
    def __init__(self):
        super(RevAtt, self).__init__()

    def forward(self, sout, pred):
        return sout.mul(1 - torch.sigmoid(pred))


# ASPP block
class ASPP(nn.Module):
    def __init__(self, in_channel, channel):
        super(ASPP, self).__init__()

        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channel, channel, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, 1, padding=3, dilation=3),
            nn.ReLU(inplace=True),
        )

        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, 1, padding=5, dilation=5),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channel, channel, 3, 1, padding=7, dilation=7),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # x4 = F.interpolate(x4, size=x3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x


# Output residual (Dual-stream Attention)
class ResiLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(ResiLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channel, channel, 1)

        self.rev_att = RevAtt()
        self.rev_conv = nn.Sequential(
            nn.Conv2d(channel, channel, k, 1, k // 2),
            # nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.ori_att = OriAtt()
        self.ori_conv = nn.Sequential(
            nn.Conv2d(channel, channel, k, 1, k // 2),
            # nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(channel * 2, channel, k, 1, k // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, 3, 1, 1),
        )

    def forward(self, sout, pred):
        sout = self.conv1x1(sout)

        sout_rev = self.rev_att(sout, pred)
        sout_rev = self.rev_conv(sout_rev)

        sout_ori = self.ori_att(sout, pred)
        sout_ori = self.ori_conv(sout_ori)

        return self.out_layer(torch.cat((sout_ori, sout_rev), 1))


# Multi-Scaled Attention Residual Prediction
class PResiLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(PResiLayer, self).__init__()
        # self.conv1x1 = nn.Conv2d(in_channel, channel, 1)

        self.rev_att = RevAtt()
        self.ori_att = OriAtt()

        self.ori_aspp = ASPP(in_channel, channel)
        self.rev_aspp = ASPP(in_channel, channel)

        self.out_layer = nn.Sequential(
            nn.Conv2d(channel * 8, channel, k, 1, k // 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(channel, 1, 3, 1, 1),
        )

    def forward(self, sout, pred):
        # sout = self.conv1x1(sout)

        sout_rev = self.rev_att(sout, pred)
        sout_rev = self.rev_aspp(sout_rev)

        sout_ori = self.ori_att(sout, pred)
        sout_ori = self.ori_aspp(sout_ori)

        sout_cat = torch.cat((sout_ori, sout_rev), 1)

        return self.out_layer(sout_cat)


# Top-Down Stream for dual att
class TDLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(TDLayer, self).__init__()
        self.resi_layer = ResiLayer(in_channel, channel, k)

    def forward(self, sout, pred):
        pred = nn.functional.interpolate(pred,
                                         size=sout.size()[2:],
                                         mode='bilinear',
                                         align_corners=True)
        residual = self.resi_layer(sout, pred)
        return pred + residual


# Top-Down Stream for Multi-scaled Bi att
class PTDLayer(nn.Module):
    def __init__(self, in_channel, channel, k):
        super(PTDLayer, self).__init__()
        self.resi_layer = PResiLayer(in_channel, channel, k)

    def forward(self, sout, pred):
        pred = nn.functional.interpolate(pred,
                                         size=sout.size()[2:],
                                         mode='bilinear',
                                         align_corners=True)
        residual = self.resi_layer(sout, pred)
        return pred + residual


# CANet Modele
class BiANet(nn.Module):
    def __init__(self):
        super(BiANet, self).__init__()
        self.rgb_sout = VGG_Sout()
        self.rgb_gs = GSLayer(512, 256, k=5)

        self.dep_sout = VGG_Sout()
        self.dep_gs = GSLayer(512, 256, k=5)

        self.rgbd_gs = GSLayer(1024, 256, k=5)

        self.td_layers = nn.ModuleList([
            PTDLayer(1024, 32, 3),
            PTDLayer(1024, 32, 3),
            PTDLayer(512, 32, 3),
            TDLayer(256, 32, 3),
            TDLayer(128, 32, 3),
        ])

    def forward(self, rgb, dep):
        [_, _, h, w] = rgb.size()

        rgb_souts, rgb_x = self.rgb_sout(rgb)
        dep_souts, dep_x = self.dep_sout(dep)

        rgb_pred = self.rgb_gs(rgb_x)  # global saliency
        dep_pred = self.dep_gs(dep_x)  # global saliency

        rgbd_souts = []  # cat rgb_souts and dep_souts
        for idx in range(len(rgb_souts)):
            rgbd_souts.append(torch.cat((rgb_souts[idx], dep_souts[idx]), 1))

        rgbd_preds = []
        rgbd_preds.append(self.rgbd_gs(torch.cat((rgb_x, dep_x),
                                                 1)))  # global saliency

        for idx in range(len(rgbd_souts)):
            rgbd_preds.append(self.td_layers[idx](rgbd_souts[-(idx + 1)],
                                                  rgbd_preds[idx]))

        scaled_preds = []
        scaled_preds.append(
            torch.sigmoid(
                nn.functional.interpolate(rgb_pred,
                                          size=(h, w),
                                          mode='bilinear',
                                          align_corners=True)))
        scaled_preds.append(
            torch.sigmoid(
                nn.functional.interpolate(dep_pred,
                                          size=(h, w),
                                          mode='bilinear',
                                          align_corners=True)))

        for idx in range(len(rgbd_preds) - 1):
            scaled_preds.append(
                torch.sigmoid(
                    nn.functional.interpolate(rgbd_preds[idx],
                                              size=(h, w),
                                              mode='bilinear',
                                              align_corners=True)))
        scaled_preds.append(torch.sigmoid(rgbd_preds[-1]))

        # rgb_gs, dep_gs, rgbd(from top to down), final pred is scaled_preds[-1]
        return scaled_preds


# weight init
def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
