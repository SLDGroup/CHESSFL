import torch
import copy
from models.get_model import make_batchnorm
from utils.dataset_utils import load_data


def make_batchnorm_stats(model, cuda_name, dataset_name, seed=42, edge_server_num=-1, edge=False, labeled=4000, chessfl=False):
    device = torch.device(cuda_name)
    with torch.no_grad():
        sbn_model = copy.deepcopy(model)
        sbn_model = sbn_model.to(device)
        sbn_model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=True))
        sbn_model.train(True)
        if not chessfl:
            if not edge:
                data_loader = load_data(seed=seed, lab_dataset=True, batch_size=250, labeled=labeled, dataset_name=dataset_name)
                for img, label in data_loader:
                    img = img.to(device)
                    sbn_model(img)
            else:
                data_loader = load_data(dev_idx=edge_server_num, seed=seed, is_edge_server=True, batch_size=250, labeled=labeled, dataset_name=dataset_name)
                for img, label in data_loader:
                    img = img.to(device)
                    sbn_model(img)
        else:
            if edge:
                data_loader = load_data(dev_idx=edge_server_num, seed=seed, is_edge_server=True, batch_size=250, labeled=labeled, rotate=True, dataset_name=dataset_name)
                for img_w, img_s, label, img_rot, tf_type in data_loader:
                    img_w, img_s, label = img_w.to(device), img_s.to(device), label.to(device)
                    img_rot, tf_type = img_rot.to(device), tf_type.to(device)
                    sbn_model(img_w, img_rot, img_s)
    return sbn_model
