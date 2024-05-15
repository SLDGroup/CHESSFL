#!/bin/bash

PORT=10000
PORT_INCR=50
seeds=(42 182 342)
comm_rounds=500
verbose='false'
num_devices=15 # number of simulated devices at the same time
hierarchical='true'
learning_rate=0.03

declare -a experiment_configs=(
# experiment 0| runs 1| data_iid 2| device_cuda_name 3| dev_local_epochs 4| edge_server_epochs 5| k2 6| train_type 7|
# model_name 8| dataset_name 9| labeled 10
  ############ MAX LABELS 4000/10000  ###########
  ####################  IID  ####################
  "100 3 true  cuda:0 1 1 2 semifl    wresnet28x2 cifar10   4000"
  "200 3 true  cuda:2 1 1 2 fixmatch  wresnet28x2 cifar100  10000"
  "300 3 true  cuda:4 1 1 2 chessfl   wresnet28x2 svhn      4000"
  ##################  Non-IID  ##################
  "101 3 false cuda:1 1 1 2 semifl    wresnet28x2 cifar10   4000"
  "201 3 false cuda:3 1 1 2 fixmatch  wresnet28x2 cifar100  10000"
  "301 3 false cuda:5 1 1 2 chessfl   wresnet28x2 svhn      4000"
)

num_users=50 # total number of devices (device indexes)
num_edge_servers=5
dev_fraction=0.1 # taken into account if not hierarchical
sbn="edge_true_false"
save_opt="true"

for experiment_config in "${experiment_configs[@]}"
do
  read -ra exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  runs="${exp_config[1]}"
  data_iid="${exp_config[2]}"
  device_cuda_name="${exp_config[3]}"
  dev_local_epochs="${exp_config[4]}"
  edge_server_epochs="${exp_config[5]}"
  k2="${exp_config[6]}"
  train_type="${exp_config[7]}"
  model_name="${exp_config[8]}"
  dataset_name="${exp_config[9]}"
  labeled="${exp_config[10]}"

  for (( run=1; run<=runs; run++ ))
  do
    files_number="files_${experiment}_${run}"
    seed=${seeds[$((run-1))]}
    for i in $(seq 0 $((num_devices-1))) # number of simulated devices at the same time
    do
      cuda_names[$i]="${device_cuda_name}"
      ports[$i]=$((PORT+i))
    done

    cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
    dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"
    exec_name="exec_exp${experiment}_run${run}.bash"

    echo "${experiment}", "${run}"
    python generate_configs.py \
    --cloud_config_filename "${cloud_config_filename}" \
    --dev_config_filename "${dev_config_filename}" \
    --cloud_cuda_name "${device_cuda_name}" \
    --exec_name "${exec_name}" \
    --experiment "${experiment}" \
    --run "${run}" \
    --seed "${seed}" \
    --model_name "${model_name}" \
    --train_type "${train_type}" \
    --comm_rounds "${comm_rounds}" \
    --verbose "${verbose}" \
    --num_devices "${num_devices}" \
    --dataset_name "${dataset_name}" \
    --num_users "${num_users}" \
    --dev_fraction "${dev_fraction}" \
    --learning_rate "${learning_rate}" \
    --files_number "${files_number}" \
    --data_iid "${data_iid}" \
    --dev_hw_type "${files_number}" \
    --ports "${ports[*]}" \
    --cuda_names "${cuda_names[*]}" \
    --dev_local_epochs "${dev_local_epochs}" \
    --cloud_cuda_name "${device_cuda_name}" \
    --hierarchical "${hierarchical}" \
    --num_edge_servers "${num_edge_servers}" \
    --edge_server_epochs "${edge_server_epochs}" \
    --k2 "${k2}" \
    --sbn "${sbn}" \
    --save_opt "${save_opt}" \
    --labeled "${labeled}"

    PORT=$((PORT+PORT_INCR))
  done
done
