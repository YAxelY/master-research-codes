"""Patch encoder: truncated ResNet-18, as in the original ips_net.py n_res_blocks = 4 : cut after layer4, downsampling /32, output 512 n_res_blocks = 2 : cut after layer2, downsampling /8,  output 128"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class PatchEncoder(nn.Module):

    def __init__(self, n_chan_in=3, pretrained=True, n_res_blocks=4):
        super().__init__()
        if n_res_blocks not in (2, 4):
            raise ValueError("n_res_blocks must be 2 or 4")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        if n_chan_in != 3:
            old = net.conv1
            new = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    new.weight.copy_(mean_w.repeat(1, n_chan_in, 1, 1))
            net.conv1 = new

        layers = [net.conv1, net.bn1, net.relu, net.maxpool,
                  net.layer1, net.layer2]
        self.out_dim = 128
        if n_res_blocks == 4:
            layers += [net.layer3, net.layer4]
            self.out_dim = 512
        layers.append(net.avgpool)
        self.net = nn.Sequential(*layers)
        self.downsample = 8 if n_res_blocks == 2 else 32

    def forward(self, x):
        return torch.flatten(self.net(x), 1)

    def feature_map_size(self, patch_size):
        """Exact side of the map before average pooling (1 = nothing to pool)."""
        h = (patch_size + 2 * 3 - 7) // 2 + 1      # conv1  k7 s2 p3
        h = (h + 2 * 1 - 3) // 2 + 1               # maxpool k3 s2 p1
        h = (h + 2 * 1 - 3) // 2 + 1               # layer2 s2 (layer1 s1)
        if self.downsample == 32:
            h = (h + 2 * 1 - 3) // 2 + 1           # layer3 s2
            h = (h + 2 * 1 - 3) // 2 + 1           # layer4 s2
        return max(1, h)


class ScaleEmbedding(nn.Module):
    """One learned vector per scale, added to the projected features."""

    def __init__(self, D, n_scales):
        super().__init__()
        self.emb = nn.Embedding(n_scales, D)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, feats, scale_ids):
        return feats + self.emb(scale_ids)
