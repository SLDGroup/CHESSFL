import argparse
import json
from os import path, makedirs

parser = argparse.ArgumentParser()
parser.add_argument('--cloud_config_filename', type=str, help='File name for cloud configs')
parser.add_argument('--dev_config_filename', type=str, help='File name for device configs')

# Not in the config.bash
parser.add_argument('--cloud_ip', type=str, default="127.0.0.1", help='Cloud IP')
parser.add_argument('--cloud_port', type=int, default=22, help='Cloud port')
parser.add_argument('--cloud_cuda_name', type=str, default="cuda:0", help='Cloud cuda_name')

# In the config.bash
parser.add_argument('--exec_name', type=str, default="exec_exp1_run1.bash", help='Executable script name')
parser.add_argument('--experiment', type=int, default=1, help='Experiment number')
parser.add_argument('--run', type=int, default=1, help='Run number')
parser.add_argument('--seed', type=int, help='Seed')

parser.add_argument('--model_name', type=str, default="conv5", help='Model name')
parser.add_argument('--train_type', type=str, default="fixmatch", help='What training type to use')
parser.add_argument('--comm_rounds', type=int, default=500, help='Communication rounds')
parser.add_argument('--verbose', type=str, default='false', help='Verbosity [true, false]')

parser.add_argument('--num_devices', type=int, default=10, help='Number of devices actually running on hardware')
parser.add_argument('--dataset_name', type=str, default="cifar10", help='Dataset name')
parser.add_argument('--num_users', type=int, help='Total number of users')
parser.add_argument('--dev_fraction', type=float, default=0.1, help='Fraction of num_users to run rand per comm round')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--files_number', type=str, default="files_1_1", help='Files folder number')
parser.add_argument('--data_iid', type=str, help='IID [true] or non-IID [false]')

parser.add_argument('--dev_hw_type', type=str, default="files_1", help='Hardware type for all device')
parser.add_argument('--ports', nargs='+', default=[22, 22], help='Ports for devices')
parser.add_argument('--cuda_names', nargs='+', default=['cpu', 'cpu'], help='Cuda_name for each device [cuda, cpu]')
parser.add_argument('--dev_local_epochs', type=int, default=5, help='Local epochs for all device')

# New semi-supervised
parser.add_argument('--hierarchical', type=str, default='true', help='True if HFL')
parser.add_argument('--num_edge_servers', type=int, default=5, help='Number of edge servers')
parser.add_argument('--edge_server_epochs', type=int, default=5, help='Number of edge server finetuning epochs')
parser.add_argument('--k2', type=int, default=5, help='Number of edge aggregations')
parser.add_argument('--labeled', type=int, default=5, help='Number of labeled images')
parser.add_argument('--sbn', type=str, default="server_labeled", help='Static BN type')
parser.add_argument('--save_opt', type=str, default='false', help='Save optimizer for each entity?')


args = parser.parse_args()
cloud_config_filename = args.cloud_config_filename
dev_config_filename = args.dev_config_filename

cloud_ip = args.cloud_ip
cloud_port = args.cloud_port
cloud_cuda_name = args.cloud_cuda_name

exec_name = args.exec_name
experiment = args.experiment  # exp1
run = args.run  # run1
seed = args.seed

model_name = args.model_name
train_type = args.train_type
comm_rounds = args.comm_rounds
verbose = args.verbose
if verbose == "true":
    verbose = True
else:
    verbose = False

num_devices = args.num_devices
dataset_name = args.dataset_name
num_users = args.num_users
dev_fraction = args.dev_fraction
learning_rate = args.learning_rate
files_number = args.files_number
data_iid = args.data_iid
if data_iid == "true":
    data_iid = True
else:
    data_iid = False


num_edge_servers = args.num_edge_servers
edge_server_epochs = args.edge_server_epochs
k2 = args.k2
labeled = args.labeled
save_opt = args.save_opt
if save_opt == "true":
    save_opt = True
else:
    save_opt = False
sbn = args.sbn
hierarchical = args.hierarchical
if hierarchical == "true":
    hierarchical = True
else:
    hierarchical = False

config_dict = {}
makedirs("configs", exist_ok=True)
with open(path.join("configs", cloud_config_filename), 'w') as file:
    config_dict["cloud_ip"] = cloud_ip
    config_dict["cloud_port"] = cloud_port
    config_dict["cloud_cuda_name"] = cloud_cuda_name

    config_dict["experiment"] = cloud_config_filename.split("_")[2]
    config_dict["run"] = cloud_config_filename.split("_")[3].split('.')[0]

    config_dict["model_name"] = f"{dataset_name}_{model_name}"
    config_dict["train_type"] = train_type
    config_dict["comm_rounds"] = comm_rounds
    config_dict["verbose"] = verbose

    config_dict["num_users"] = num_users
    config_dict["dev_fraction"] = dev_fraction
    config_dict["learning_rate"] = learning_rate
    config_dict["files_number"] = files_number
    config_dict["data_iid"] = data_iid

    config_dict["hierarchical"] = hierarchical
    config_dict["num_edge_servers"] = num_edge_servers
    config_dict["edge_server_epochs"] = edge_server_epochs
    config_dict["k2"] = k2
    config_dict["labeled"] = labeled
    config_dict["sbn"] = sbn
    config_dict["save_opt"] = save_opt

    json.dump(config_dict, file, indent=2)

# Device CONFIGS
dev_hw_type = args.dev_hw_type
ports = args.ports
cuda_names = args.cuda_names
dev_local_epochs = args.dev_local_epochs

aux = []
for k in ports[0].split(" "):
    aux.append(int(k))
ports = aux

aux = []
for k in cuda_names[0].split(" "):
    aux.append(k)
cuda_names = aux

config_dict = {}
with open(path.join("configs", dev_config_filename), 'w') as file:
    config_dict["num_devices"] = num_devices
    for dev_idx in range(num_devices):
        dev_dict = {"hw_type": dev_hw_type,
                    "hostname": "127.0.0.1",
                    "port": ports[dev_idx],
                    "cuda_name": cuda_names[dev_idx],
                    "model_name": f"{dataset_name}_{model_name}",
                    "local_epochs": dev_local_epochs}
        config_dict[f"dev{dev_idx + 1}"] = dev_dict
    json.dump(config_dict, file, indent=2)

# Writing execution file exec.bash
makedirs("execs", exist_ok=True)
with open(path.join("execs", exec_name), 'w') as file:
    file.write("#!/bin/bash\n\ntrap \"kill 0\" EXIT\n\ndeclare -a elems=(\n")
    for dev_idx in range(num_devices):
        file.write(f"\t\"127.0.0.1 {ports[dev_idx]} {dev_idx}\"\n")
    file.write(")\n\nfor elem in \"${elems[@]}\"\ndo\n\tread -a tuple <<< \"$elem\"\n\tpython device.py ")
    file.write("--host=\"${tuple[0]}\" --port=\"${tuple[1]}\" --dev_idx=\"${tuple[2]}\" ")
    file.write(f"--exp={(cloud_config_filename.split('_')[2]).split('p')[1]} ")
    file.write(f"--r={cloud_config_filename.split('_')[3].split('.')[0].split('n')[1]} --seed={seed} &")
    file.write(f"\ndone\n\n")
    file.write(f"python cloud.py --cloud_cfg {path.join('configs',cloud_config_filename)} ")
    file.write(f"--dev_cfg {path.join('configs',dev_config_filename)} --seed {seed}")
