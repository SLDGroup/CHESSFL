from os import path
import os
import argparse
from utils.communication import receive_msg, send_msg, close_connection
from utils.train_test import local_training
from utils.general_utils import get_hw_info,seed_everything
import socket


class Device:
    def __init__(self, connection, exp, r, dev_idx, seed, verbose=False):
        seed_everything(seed=seed)
        self.seed = seed
        self.connection = connection
        self.dev_idx = dev_idx
        self.comm_log_file = f"dev{self.dev_idx}_log_exp{exp}_run{r}.csv"
        self.verbose = verbose

    def run(self):
        """
        Main method for the Device. The Cloud sends a message to the Device in the following format:

        setup_message = {cloud_info},{device_info},{data}

        where

        {cloud_info}  = {target_ip};{target_port};{target_path};{target_usr};{target_pwd}
        {device_info} = {hw_type};{data_iid};{cuda_name};{verbose}
        {data}        = {comm_round};{model_name};{filename};{local_epochs};{learning_rate};{train_type};{save_opt};{use_sbn};{labeled}
        """
        # Step 1. Receive setup message from Cloud.
        setup_message = receive_msg(self.connection, verbose=self.verbose)

        # Exit if there is no message.
        if setup_message is None:
            print(f"[!] No message received from the Cloud on device {self.dev_idx}.")
            return False

        if setup_message == "end":
            close_connection(connection=self.connection, verbose=self.verbose)
            return True

        cloud_info, device_info, data = setup_message.split(',')

        target_host, target_port, target_path, target_usr, target_pwd = cloud_info.split(';')

        hw_type, data_iid, cuda_name, self.verbose, real_device_idx = device_info.split(';')
        dev_pwd, dev_usr, dev_path = get_hw_info(hw_type)
        os.makedirs(dev_path, exist_ok=True)
        data_iid = bool(int(data_iid == 'True'))
        self.verbose = bool(int(self.verbose == 'True'))

        comm_round, model_name, model_filename, local_epochs, learning_rate, train_type, save_opt, use_sbn,labeled,edge_server_idx = data.split(';')
        comm_round = int(comm_round)
        labeled = int(labeled)
        edge_server_idx = int(edge_server_idx)
        local_epochs = int(local_epochs)
        learning_rate = float(learning_rate)
        save_opt = bool(int(save_opt == 'True'))
        use_sbn = bool(int(use_sbn == 'True'))

        model_path = path.join(dev_path, model_filename)
        opt_path = path.join(dev_path, f"optimizer_{real_device_idx}.pth")

        # Step 2. Synch with Cloud with msg "done_setup"
        send_msg(connection=self.connection, msg="done_setup", verbose=self.verbose)

        # Step 3. Train the model
        train_loss, train_acc, train_time = \
            local_training(model_name=model_name, train_type=train_type, model_path=model_path,
                           cuda_name=cuda_name, learning_rate=learning_rate, local_epochs=local_epochs,
                           dev_idx=real_device_idx, verbose=self.verbose, data_iid=data_iid, seed=self.seed,
                           comm_round=comm_round, opt_path=opt_path, save_opt=save_opt, use_sbn=use_sbn,labeled=labeled, dev_path=dev_path, edge_server_idx=edge_server_idx)

        # Step 4. Sync with Cloud with "done_training" message
        send_msg(connection=self.connection, msg=f"done_training;{train_time}", verbose=self.verbose)

        close_connection(connection=self.connection, verbose=self.verbose)
        return False


def start_device(host, port, dev_idx, exp, r, seed, verbose=False):
    """
    Starts the device, handles incoming connections, and records power and temperature (optional).

    Parameters:
        host (str): Hostname or IP to bind the device to.
        port (int): Port to bind the device to.
        dev_idx (int): Index of the device.
        exp (int): Experiment number.
        r (int): Run number.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        None
    """

    os.makedirs("logs", exist_ok=True)

    # Create a socket object
    soc = socket.socket()
    # Set the socket to reuse the address
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind the socket to the host and port
    soc.bind((host, port))

    if verbose:
        print(f"[+] Socket created on device {dev_idx}")

    try:
        while True:
            # Listen for incoming connections
            soc.listen(5)
            if verbose:
                print(f"[+] Device {dev_idx} is listening on host:port {host}:{port}")
            # Accept a connection
            connection, _ = soc.accept()
            # Create a Device instance with the accepted connection
            device = Device(connection=connection, exp=exp, r=r, dev_idx=dev_idx, seed=seed, verbose=verbose)
            # Run the device
            if device.run():
                break
    except BaseException as e:
        print(f'[!] Device {dev_idx} ERROR: {e} Socket closed due to no incoming connections or error.')
    finally:
        # Close the socket
        soc.close()


parser = argparse.ArgumentParser(description="Device arguments")
parser.add_argument('--host', default="127.0.0.1", help='IP address of the device')
parser.add_argument('--port', type=int, default=10000, help='Port for the device to listen on')
parser.add_argument('--dev_idx', type=int, default=0, help='Index of the device')
parser.add_argument('--exp', type=int, default=0,  help='Experiment number')
parser.add_argument('--r', type=int, default=0, help='Run number')
parser.add_argument('--seed', type=int, default=42, help='Seed number')
parser.add_argument('--verbose', action='store_true', help='If verbose or not')
args = parser.parse_args()

start_device(host=args.host, port=args.port, dev_idx=args.dev_idx, exp=args.exp, r=args.r, seed=args.seed,
             verbose=args.verbose)
