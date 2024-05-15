from models.wideresnet import wresnet28x2
from torch import nn
import torch


def kaiming_normal(w):
    if isinstance(w, nn.Linear) or isinstance(w, nn.Conv2d):
        nn.init.kaiming_normal_(w.weight)

def init_param(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum=None, track_running_stats=False):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def get_model(model_name, use_sbn=False, rot_pred=False):
    dataset_name, model_name = model_name.split('_')
    num_classes = 10
    if dataset_name == "cifar100":
        num_classes = 100

    if dataset_name == "cifar10" or dataset_name == "cifar100" or dataset_name == "svhn":
        num_channels = 3
    else:
        print(f"Dataset name: {dataset_name} is not okay.")
        exit(-1)
    if model_name == "wresnet28x2":
        return wresnet28x2(num_classes=num_classes, init_param=init_param, make_batchnorm=make_batchnorm, use_sbn=use_sbn, rot_pred=rot_pred)
    else:
        raise NotImplementedError(f'Model {dataset_name}_{model_name} not implemented yet')
