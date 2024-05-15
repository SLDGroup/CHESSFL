import os
import copy
import torch
import argparse
import json
from models.get_model import get_model
from utils.train_test import test, edge_train, edge_train_chessfl, edge_clustering
from utils.general_utils import get_hw_info, seed_everything
from utils.fl_utils import aggregate_avg, aggregate_cos
import time
from device_handler import DeviceHandler
import numpy as np
from os import path
from utils.cloud_utils import make_batchnorm_stats

class Cloud:
    def __init__(self, cloud_cfg, dev_cfg, seed):
        self.cloud_cfg = cloud_cfg
        self.dev_cfg = dev_cfg
        self.seed = seed
        os.makedirs("logs", exist_ok=True)
        seed_everything(seed=seed)

    def federated_learning(self):
        total_time_start = time.time()
        with open(self.cloud_cfg, 'r') as cfg:
            dat = json.load(cfg)

            exp = dat["experiment"]
            r = dat["run"]
            log_file = f"log_{exp}_{r}.csv"

            files_number = dat["files_number"]

            cloud_ip = dat["cloud_ip"]
            cloud_port = dat["cloud_port"]
            cloud_cuda_name = dat["cloud_cuda_name"]
            cloud_pwd, cloud_usr, cloud_path = get_hw_info(hw_type=files_number)
            os.makedirs(cloud_path, exist_ok=True)

            model_name = dat["model_name"]  # {dataset_name}_{model_name}
            train_type = dat["train_type"]
            comm_rounds = dat["comm_rounds"]
            verbose = dat["verbose"]

            num_users = dat["num_users"]
            dev_fraction = dat["dev_fraction"]
            learning_rate = dat["learning_rate"]
            data_iid = dat["data_iid"]

            # new data
            hierarchical = dat["hierarchical"]
            num_edge_servers = dat["num_edge_servers"]
            edge_server_epochs = dat["edge_server_epochs"]
            k2 = dat["k2"]
            labeled = dat["labeled"]
            # sbn = dat["sbn"]
            dataset_name = model_name.split('_')[0]
            if train_type == "semifl":
                sbn = "edge_true_false"
            else:
                sbn = "edge_false_false"
            sbn_data = sbn.split('_')[0]
            use_sbn = True if sbn.split("_")[1] == "true" else False
            use_sbn = use_sbn and not (model_name.split('_')[1] == 'conv5')
            sbn_other = True if sbn.split("_")[2] == "true" else False  # use sBN for other approaches than SemiFL
            save_opt = True
            comm_round_time = 0
            chessfl = True if train_type == "chessfl" else False
        if train_type == "semifl" or train_type == "fixmatch":
            net_glob = get_model(model_name=f"{model_name}", use_sbn=use_sbn)
        else:
            net_glob = get_model(model_name=f"{model_name}", use_sbn=use_sbn, rot_pred=True)

        rot_pred = True if train_type == "chessfl" else False
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=comm_rounds, eta_min=0)
        if sbn_data == "server" and (train_type == "semifl" or sbn_other) and use_sbn:
            sbn_model = make_batchnorm_stats(model=net_glob, cuda_name=cloud_cuda_name, seed=self.seed, dataset_name=dataset_name, labeled=labeled)
            net_glob.load_state_dict(sbn_model.state_dict(), strict=False)

        if hierarchical:
            edge_server_models = []
            for edge_idx in range(num_edge_servers):
                if train_type == "semifl" or train_type == "fixmatch":
                    net_edge = get_model(model_name=f"{model_name}", use_sbn=use_sbn)
                else:
                    net_edge = get_model(model_name=f"{model_name}", use_sbn=use_sbn, rot_pred=True)
                net_edge.load_state_dict(net_glob.state_dict(), strict=False)
                edge_server_models.append(net_edge)

                if sbn_data == "edge" and (train_type == "semifl" or sbn_other) and use_sbn:
                    sbn_model = make_batchnorm_stats(model=edge_server_models[edge_idx], cuda_name=cloud_cuda_name, seed=self.seed,
                                                     edge_server_num=edge_idx, edge=True, chessfl=chessfl, dataset_name=dataset_name, labeled=labeled)
                    edge_server_models[edge_idx].load_state_dict(sbn_model.state_dict(), strict=False)

                if train_type == "chessfl":
                    edge_opt_path = path.join(cloud_path, f"optimizer_{edge_idx}.pth")
                    edge_server_models[edge_idx], edge_loss, edge_acc = (
                        edge_train_chessfl(model=edge_server_models[edge_idx], edge_server_idx=edge_idx,
                                   cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                   seed=self.seed, edge_server_epochs=edge_server_epochs, learning_rate=learning_rate,
                                   comm_round=0, edge_opt_path=edge_opt_path, save_opt=save_opt,
                                   labeled=labeled, rot_pred=rot_pred))

                    edge_clustering(model=edge_server_models[edge_idx], dataset_name=dataset_name,
                                    seed=self.seed, cuda_name=cloud_cuda_name, edge_server_idx=edge_idx,
                                    dev_path=cloud_path, labeled=labeled, rot_pred=rot_pred)
                torch.save(edge_server_models[edge_idx].state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))

            clients_per_edge_server = 50 // num_edge_servers
            edge_server_client_dict = {edge_server: [] for edge_server in range(num_edge_servers)}
            for client_idx in range(50):
                edge_server_idx = client_idx // clients_per_edge_server
                edge_server_client_dict[edge_server_idx].append(client_idx)

        torch.save(net_glob.state_dict(), path.join(cloud_path, f"global_weights.pth"))
        loss_func = torch.nn.CrossEntropyLoss()

        loss_test, acc_test = test(model=net_glob, loss_func=loss_func, cuda_name=cloud_cuda_name,
                                   dataset_name=dataset_name, seed=self.seed, verbose=verbose, labeled=labeled, rot_pred=rot_pred)
        prev_loss_test, prev_acc_test = loss_test, acc_test
        print(f"Initial Accuracy: {acc_test:.2f}%; Initial Loss: {loss_test:.4f}")

        with open(path.join("logs", log_file), 'w') as logger:
            logger.write(f"CommRound,Acc,Loss\n0,{acc_test},{loss_test}\n")
        if hierarchical:
            with open(path.join("logs", f"edge_{log_file}"), 'w') as logger:
                logger.write(f"CommRound,EdgeIdx,TrainAcc,TrainLoss,TestAcc,TestLoss\n")

        for comm_round in range(comm_rounds):
            start_time = time.time()
            remaining_time = comm_round_time * (comm_rounds-(comm_round+1))
            print(f"{exp}, {r}, {train_type}, {model_name}:{cloud_cuda_name}, {labeled}, "
                  f"lr={learning_rate}, {'IID' if data_iid else 'non-IID'}, "
                  f"TIME:{int(comm_round_time // 60)}m {int(comm_round_time % 60)}s, "
                  f"Approx remaining time: {int(remaining_time // 3600)}h {int((remaining_time % 3600) // 60)}m {int(remaining_time % 60)}s")
            global_weights = torch.load(path.join(cloud_path, f"global_weights.pth"))
            net_glob.load_state_dict(global_weights)
            with open(self.dev_cfg, 'r') as f:
                dt = json.load(f)
            num_devices = dt["num_devices"]
            if verbose:
                print(f"Number of devices: {num_devices}")

            for edge_idx in range(num_edge_servers):
                edge_server_models[edge_idx].load_state_dict(torch.load(path.join(cloud_path, f"edge_weights_{edge_idx}.pth")), strict=False)

            # split available_devices_idx in series of <num_devices> to run in parallel
            mycnt = 0
            mydictcnt = 0
            avail_devs_idx_dict = {}
            # Obtain new number of devices to consider for this communication round
            if not hierarchical:
                available_devices = np.random.choice(range(num_users), size=int(dev_fraction * num_users), replace=False)
                if verbose:
                    print(f"Available devices: {available_devices}")
            elif hierarchical:
                available_devices = range(num_users)

            for myidx in available_devices:
                if mycnt == 0:
                    avail_devs_idx_dict[mydictcnt] = [myidx]
                else:
                    avail_devs_idx_dict[mydictcnt].append(myidx)
                mycnt += 1
                if mycnt == num_devices:
                    mycnt = 0
                    mydictcnt += 1

            for mydict_key in avail_devs_idx_dict.keys():
                device_handler_list = []
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    dev = dt[f"dev{idx + 1}"]
                    dev_type = dev["hw_type"]
                    local_epochs = dev["local_epochs"]
                    dev_host = dev["hostname"]
                    dev_port = dev["port"]
                    dev_cuda_name = dev["cuda_name"]
                    dev_model_name = dev["model_name"]

                    if not hierarchical:
                        net_local = copy.deepcopy(net_glob).to(torch.device(dev_cuda_name))
                    else:
                        net_local = copy.deepcopy(edge_server_models[i // 10]).to(torch.device(dev_cuda_name))
                    dev_model_filename = f"dev_{i}.pth"
                    torch.save(net_local.state_dict(), path.join(cloud_path, dev_model_filename))
                    dev_pwd, dev_usr, dev_path = get_hw_info(dev_type)
                    edge_s_idx = client_idx // clients_per_edge_server if hierarchical else 0
                    """
                    setup_message = {cloud_info},{device_info},{data}

                    where

                    {cloud_info}  = {cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}
                    {device_info} = {dev_type};{data_iid};{cuda_name};{verbose};{real_device_idx}
                    {data}        = {comm_round};{model_name};{filename};{local_epochs};{learning_rate};{train_type};{save_opt};{use_sbn};{labeled}
                    """
                    cloud_info = f"{cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}"
                    device_info = f"{dev_type};{data_iid};{dev_cuda_name};{verbose};{i}"
                    data = f"{comm_round};{dev_model_name};{dev_model_filename};{local_epochs};{learning_rate};{train_type};{save_opt};{use_sbn};{labeled};{edge_s_idx}"

                    setup_message = f"{cloud_info},{device_info},{data}"

                    device_handler_list.append(
                        DeviceHandler(cloud_path=cloud_path, dev_idx=i, dev_host=dev_host, dev_port=dev_port,
                                      dev_usr=dev_usr, dev_pwd=dev_pwd, dev_path=dev_path,
                                      dev_model_filename=dev_model_filename, setup_message=setup_message,
                                      verbose=verbose)
                    )
                if verbose:
                    print(f"[+] Started all clients")
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    device_handler_list[idx].start()
                if verbose:
                    print(f"[+] Wait until clients to finish their job")
                value = []
                for idx, i in enumerate(avail_devs_idx_dict[mydict_key]):
                    value.append(device_handler_list[idx].join())
            if verbose:
                print(" \n[+] Joined all clients")

            if hierarchical:
                for edge_idx in range(num_edge_servers):
                    local_weights = []
                    for i in available_devices:
                        if i in edge_server_client_dict[edge_idx]:
                            dev_model_filename = f"dev_{i}.pth"
                            if train_type == "semifl" or train_type == "fixmatch":
                                net_local = get_model(model_name=f"{model_name}", use_sbn=use_sbn)
                            else:
                                net_local = get_model(model_name=f"{model_name}", use_sbn=use_sbn, rot_pred=True)
                            net_local.load_state_dict(torch.load(path.join(cloud_path, dev_model_filename), map_location=torch.device(cloud_cuda_name)), strict=False)
                            local_weights.append(net_local.to(torch.device(cloud_cuda_name)).state_dict())

                    if train_type == "chessfl":
                        w_edge = edge_server_models[edge_idx].to(torch.device(cloud_cuda_name)).state_dict()
                        w_edge = aggregate_cos(sigma=0.1, global_weights=w_edge, local_weights=local_weights, device=torch.device(cloud_cuda_name))
                    else:
                        w_edge = aggregate_avg(local_weights=local_weights)
                    torch.save(w_edge, path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))
                    edge_server_models[edge_idx].load_state_dict(w_edge)

                    edge_opt_path = path.join(cloud_path, f"optimizer_edge{edge_idx}.pth")

                    if train_type == "chessfl":
                        # Step 1 Fine tune aggregated model
                        edge_server_models[edge_idx], edge_loss, edge_acc = (
                            edge_train_chessfl(model=edge_server_models[edge_idx], edge_server_idx=edge_idx,
                                       cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                       seed=self.seed, edge_server_epochs=edge_server_epochs, learning_rate=learning_rate,
                                       comm_round=comm_round, edge_opt_path=edge_opt_path, save_opt=save_opt, labeled=labeled, rot_pred=rot_pred))

                        edge_clustering(model=edge_server_models[edge_idx], dataset_name=dataset_name,
                                        seed=self.seed, cuda_name=cloud_cuda_name, edge_server_idx=edge_idx,
                                        dev_path=cloud_path, labeled=labeled, rot_pred=rot_pred)
                    else:
                        edge_server_models[edge_idx], edge_loss, edge_acc = (
                            edge_train(model=edge_server_models[edge_idx], edge_server_idx=edge_idx,
                                       cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                       seed=self.seed, edge_server_epochs=edge_server_epochs, learning_rate=learning_rate,
                                       comm_round=comm_round, edge_opt_path=edge_opt_path, save_opt=save_opt, labeled=labeled))
                    torch.save(edge_server_models[edge_idx].state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))

                if (comm_round + 1) % k2 == 0:
                    # 1. Aggregate edge models
                    edge_weights = []
                    for edge_idx in range(num_edge_servers):
                        edge_weights.append(edge_server_models[edge_idx].state_dict())

                    if train_type == "chessfl":
                        w_glob = net_glob.to(torch.device(cloud_cuda_name)).state_dict()
                        w_glob = aggregate_cos(sigma=0.1, global_weights=w_glob, local_weights=edge_weights, device=torch.device(cloud_cuda_name))
                    else:
                        w_glob = aggregate_avg(local_weights=edge_weights)

                    torch.save(w_glob, path.join(cloud_path, f"global_weights.pth"))
                    net_glob.load_state_dict(w_glob)
                    loss_test, acc_test = test(model=net_glob, loss_func=loss_func, cuda_name=cloud_cuda_name,
                                               dataset_name=dataset_name, seed=self.seed, verbose=verbose, labeled=labeled, rot_pred=rot_pred)
                    prev_loss_test, prev_acc_test = loss_test, acc_test

                    if sbn_data == "server" and (train_type == "semifl" or sbn_other) and use_sbn:
                        sbn_model = make_batchnorm_stats(model=net_glob, cuda_name=cloud_cuda_name, seed=self.seed, dataset_name=dataset_name, labeled=labeled)
                        net_glob.load_state_dict(sbn_model.state_dict(), strict=False)
                        torch.save(net_glob.state_dict(), path.join(cloud_path, f"global_weights.pth"))

                    if sbn_data == "edge" and (train_type == "semifl" or sbn_other) and use_sbn:
                        for edge_idx in range(num_edge_servers):
                            sbn_model = make_batchnorm_stats(model=net_glob, cuda_name=cloud_cuda_name,
                                                             seed=self.seed, edge_server_num=edge_idx, edge=True, chessfl=chessfl, dataset_name=dataset_name, labeled=labeled)
                            edge_server_models[edge_idx].load_state_dict(sbn_model.state_dict(), strict=False)
                            torch.save(edge_server_models[edge_idx].state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))

                    if train_type == "chessfl":
                        for edge_idx in range(num_edge_servers):
                            torch.save(net_glob.state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))
                            edge_server_models[edge_idx].load_state_dict(torch.load(path.join(cloud_path, f"edge_weights_{edge_idx}.pth")), strict=False)
                            edge_server_models[edge_idx], edge_loss, edge_acc = (
                                edge_train_chessfl(model=edge_server_models[edge_idx], edge_server_idx=edge_idx,
                                                   cuda_name=cloud_cuda_name, dataset_name=dataset_name,
                                                   seed=self.seed, edge_server_epochs=edge_server_epochs,
                                                   learning_rate=learning_rate,
                                                   comm_round=comm_round, edge_opt_path=edge_opt_path,
                                                   save_opt=save_opt, labeled=labeled, rot_pred=rot_pred))
                            edge_clustering(model=edge_server_models[edge_idx], dataset_name=dataset_name,
                                            seed=self.seed, cuda_name=cloud_cuda_name, edge_server_idx=edge_idx,
                                            dev_path=cloud_path, labeled=labeled, rot_pred=rot_pred)
                            torch.save(edge_server_models[edge_idx].state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))
                    elif train_type == "fixmatch":
                        for edge_idx in range(num_edge_servers):
                            torch.save(net_glob.state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))

                else:
                    if sbn_data == "edge" and (train_type == "semifl" or sbn_other) and use_sbn:
                        for edge_idx in range(num_edge_servers):
                            sbn_model = make_batchnorm_stats(model=edge_server_models[edge_idx], cuda_name=cloud_cuda_name, seed=self.seed,
                                                             edge_server_num=edge_idx, edge=True, chessfl=chessfl, dataset_name=dataset_name, labeled=labeled)
                            edge_server_models[edge_idx].load_state_dict(sbn_model.state_dict(), strict=False)
                            torch.save(edge_server_models[edge_idx].state_dict(), path.join(cloud_path, f"edge_weights_{edge_idx}.pth"))
                    loss_test, acc_test = prev_loss_test, prev_acc_test

            scheduler.step()
            learning_rate = optimizer.param_groups[0]['lr']

            with open(path.join("logs", log_file), 'a+') as logger:
                logger.write(f"{comm_round+1},{acc_test},{loss_test}\n")
            print(f"CommRound: {comm_round+1}; Accuracy: {acc_test:.2f}%; Loss: {loss_test:.4f}\n")
            comm_round_time = time.time() - start_time

        print(f"Total time for experiment: {time.time() - total_time_start}")

        with open(path.join("logs", "time.csv"), 'a+') as logger:
            logger.write(f"{exp},{r},{time.time() - total_time_start}\n")

        self.end_experiment(verbose=verbose)

    def end_experiment(self, verbose):
        if verbose:
            print("[+] Closing everything")
        device_handler_list = []
        with open(self.dev_cfg, 'r') as f:
            dt = json.load(f)
        for i in range(dt["num_devices"]):
            dev = dt[f"dev{i + 1}"]
            device_handler_list.append(
                DeviceHandler(dev_host=dev["hostname"], dev_port=dev["port"], setup_message="end", verbose=verbose)
            )

        if verbose:
            print("[+] Closing all clients...")

        for i in range(dt["num_devices"]):
            device_handler_list[i].start()

        if verbose:
            print("[+] Wait until clients close")

        for i in range(dt["num_devices"]):
            device_handler_list[i].join()

        if verbose:
            print("[+] Closed all clients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_cfg', type=str, default="configs/cloud_cfg_exp1.json",
                        help='Cloud configuration file name')
    parser.add_argument('--dev_cfg', type=str, default="configs/dev_cfg_exp1.json",
                        help='Device configuration file name')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    cloud = Cloud(cloud_cfg=args.cloud_cfg, dev_cfg=args.dev_cfg, seed=args.seed)
    cloud.federated_learning()
