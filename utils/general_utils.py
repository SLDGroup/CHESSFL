import os
import random
import torch
from torch import nn
import numpy as np


def get_hw_info(hw_type):
    if hw_type.split('_')[0] == "files":
        # password, username, path of project + files/
        return "pwd", "usr", f"/home/allen-admin/PycharmProjects/FL/files/files{hw_type.split('_')[1]}_{hw_type.split('_')[2]}/"
    else:
        print("[!] ERROR wrong device type.")


def seed_everything(seed=42):
    """
    Sets seed for all the possible sources of randomness to ensure reproducibility.

    Parameters:
        seed (int): The seed value to be set.
        include_np (bool): To include the numpy random or not
    """

    # Set the seed for Python's random module
    random.seed(seed)
    np.random.seed(seed)
    # Set the seed for Python's hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set the seed for PyTorch operations
    torch.manual_seed(seed)

    # Set the seed for PyTorch GPU operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        # Ensuring deterministic behavior in PyTorch for certain cudnn algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # print(f"[+] Seeds for all randomness sources set to: {seed}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count