import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate, rot_pred=False):
        super().__init__()
        self.rot_pred = rot_pred
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate))
        blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten())
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)
        if self.rot_pred:
            self.rotpred_fc = nn.Linear(hidden_size[-1], 6)
        self.old = False

    def forward(self, x, x_rot=None, x_s=None):
        if not self.rot_pred:
            x_out = self.blocks(x)
            output = self.classifier(x_out)
            return output
        else:
            if self.old:
                x_out = self.blocks(x)
                output = self.classifier(x_out)
                if x_rot is not None:
                    x_out_rot = self.blocks(x_rot)
                    rot_output = self.rotpred_fc(x_out_rot)
                    return output, rot_output
                else:
                    return output
            else:
                x_out_w = self.blocks(x)
                output_w = self.classifier(x_out_w)
                x_out_s = self.blocks(x_s)
                output_s = self.classifier(x_out_s)
                x_out_rot = self.blocks(x_rot)
                rot_output = self.rotpred_fc(x_out_rot)
                return output_w, output_s, rot_output, x_out_w, x_out_s, x_out_rot


def wresnet28x2(num_classes, init_param=None, make_batchnorm=None, use_sbn=False, rot_pred=False):
    data_shape = [3, 32, 32]
    model = WideResNet(data_shape=data_shape, num_classes=num_classes, depth=28, widen_factor=2, drop_rate=0.0, rot_pred=rot_pred)
    model.apply(init_param)
    if use_sbn:
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
    return model

def wresnet28x8(num_classes, init_param=None, make_batchnorm=None, use_sbn=False, rot_pred=False):
    data_shape = [3, 32, 32]
    model = WideResNet(data_shape=data_shape, num_classes=num_classes, depth=28, widen_factor=8, drop_rate=0.0, rot_pred=rot_pred)
    model.apply(init_param)
    if use_sbn:
        model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=False))
    return model