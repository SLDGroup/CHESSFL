import pickle
import time
from os import path
import os
import socket
import zipfile
import paramiko
from scp import SCPClient
from os.path import basename


def connect(host, port=22, verbose=False):
    """
    Establishes a connection to a given host and port.

    Parameters:
        host (str): The hostname of the device to connect to.
        port (int, optional): The port number on the device to connect to. Defaults to 22.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        socket.socket: A socket object representing the connection to the host.
    """

    # Instantiate a new socket object.
    soc = socket.socket()
    if verbose:
        print("[+] Status: Socket successfully created.")

    # Status message for the attempted connection.
    if verbose:
        print(f"[+] Status: Attempting to connect to {host} on port {port}...")

    while True:
        try:
            # Attempt to establish a connection to the host.
            soc.connect((host, port))

            # Connection is successful.
            if verbose:
                print(f"[+] Success: Connection established to {host} on port {port}.")
            return soc

        except (socket.error, ConnectionError) as e:
            # An error occurred while attempting to connect. Wait for 5 seconds before retrying.
            print(f"[!] Error: {e}")
            print("[+] Status: Connection failed. Retrying in 5 seconds...")
            time.sleep(5)


def close_connection(connection, verbose=False):
    """
    Closes an established socket connection.

    Parameters:
        connection (socket.pyi): The socket object representing the connection to be closed.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
    """

    # Display status message before closing the connection if verbose is set to True
    if verbose:
        print("[+] Status: Preparing to close the socket connection...")

    # Close the socket connection
    connection.close()

    # Display status message after closing the connection if verbose is set to True
    if verbose:
        print("[+] Success: Socket connection closed.")


def send_file(connection, target, target_host, target_port, target_usr, target_pwd, target_path, filename, source_path,
              verbose=False):
    """
    Sends a locally trained model to a target.

    This function notifies the target that the model is ready for transfer. The notification format is 
    "{filename};{filesize}". It then waits for a response from the target. If the response is "Confirm", 
    it closes the connection. If the response is "Resend", it reattempts to send the file until a "Confirm"
    response is received.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        target (str): The target's identifier.
        target_host (str): The IP address of the target.
        target_port (int): The port number at the target.
        target_usr (str): The username for authentication at the target.
        target_pwd (str): The password for authentication at the target.
        target_path (str): The directory on the target where the file will be uploaded.
        filename (str): The name of the file to be sent.
        source_path (str): The directory on the local device where the file is located.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        int: The size of the file sent.
    """

    if verbose:
        print(f"[+] Preparing to send local model to {target}")

    zip_filename, filesize = zip_file(filename=filename, target_path=source_path, verbose=verbose)

    if verbose:
        print(f"[+] Initiating file transfer: {zip_filename} to {target_host}")

    while True:
        scp_file(target_host=target_host, target_usr=target_usr, target_pwd=target_pwd,
                 target_path=target_path, zip_filename=zip_filename, source_path=source_path, verbose=verbose)
        send_msg(connection=connection, msg=f"{filename};{filesize}", verbose=verbose)

        if verbose:
            print(f"[+] Waiting for confirmation from {target}")

        received_data = receive_msg(connection=connection, verbose=verbose)

        if received_data == "Confirm":
            if verbose:
                print(f"[+] Successful file transfer to {target} ({target_host})")
            break
        elif received_data == "Resend":
            if verbose:
                print(f"[!] Failed file transfer, attempting to resend: {zip_filename} to {target_host}")

    return filesize


def receive_file(connection, target, target_host, local_path, verbose=False, msg="zip"):
    """
    Waits for a file from the target and validates its transmission.
    Continually attempts to receive a file (global model) from a target until the file is successfully received.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        target (str): The target's string identifier.
        target_host (str): The IP address of the target.
        local_path (str): The path where the received file will be saved on the local device.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
        msg (str, optional): The type of file being transferred. Defaults to "zip".

    Returns:
        int: The size of the successfully received file.
    """

    if verbose:
        print(f"[+] Starting {msg} transfer from {target}")

    while True:
        filename, filesize = get_notification_transfer_done(connection=connection, target=target, verbose=verbose)
        zipped_file = f"{filename.split('.')[0]}.zip"

        unzip_file(connection=connection, zip_filename=zipped_file,  target_path=local_path, verbose=verbose)

        file_path = path.join(local_path, zipped_file)

        if path.exists(file_path) and os.path.getsize(file_path) == int(filesize):
            if verbose:
                print(f"[+] Successful file reception from {target_host}")
            send_msg(connection=connection, msg="Confirm", verbose=verbose)
            break

        error_msg = (f"[!] File {zipped_file} size mismatch: {os.path.getsize(file_path)} != {filesize}"
                     if path.exists(file_path)
                     else f"[!] Failed to locate {zipped_file} in {local_path}")

        if verbose:
            print(error_msg)
        send_msg(connection=connection, msg="Resend", verbose=verbose)

    if verbose:
        print(f"[+] Successful transfer of {msg} from {target}")
    return int(filesize)


def scp_file(target_host, target_usr, target_pwd, target_path, zip_filename, source_path, verbose=False):
    """
    Securely copies a file from the local host to a remote host via SCP.

    Parameters:
        target_host (str): The IP address of the remote host.
        target_usr (str): The username for authentication on the remote host.
        target_pwd (str): The password for authentication on the remote host.
        target_path (str): The directory path on the remote host where the file will be copied.
        zip_filename (str): The name of the file to be copied.
        source_path (str): The directory path on the local host where the file resides.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
    """

    if verbose:
        print(f"[+] Status: Initiating file transfer of {zip_filename} to the remote host.")

    while True:
        try:
            # Use paramiko to handle the SCP file transfer.
            with paramiko.SSHClient() as client:
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(hostname=target_host, username=target_usr, password=target_pwd, port=22,
                               auth_timeout=2000, banner_timeout=2000)

                with SCPClient(client.get_transport()) as scp:
                    scp.put(path.join(source_path, zip_filename), remote_path=target_path)

                # If the file transfer is successful, exit the loop.
                break

        except Exception as e:
            # An error occurred during the file transfer. Print the error and retry after 5 seconds.
            if verbose:
                print(f"[!] Error: {e}")
                print(f"[!] Failed to transfer the file. Retrying...")
            time.sleep(5)

    if verbose:
        print("[+] Success: File transfer completed.")


def get_notification_transfer_done(connection, target, verbose=False):
    """
    Waits for a message from the client indicating the completion of file transfer.

    The expected message format from the client is "{filename};{filesize}".

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        target (str): The target IP or hostname from which the file is received.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        tuple: The name and size of the file received.
    """
    msg = receive_msg(connection=connection, verbose=verbose)
    assert msg is not None, f"[!] Error: No input received from '{target}'."

    filename, filesize = msg.split(';')

    return filename, int(filesize)


def zip_file(filename, target_path, verbose=False):
    """
    Zips the specified file.

    Parameters:
        filename (str): The name of the file to zip.
        target_path (str): The path where the zip file should be created.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        tuple: The name and size of the zip file.
    """
    zip_filename = filename.split(".")[0] + ".zip"

    if verbose:
        print(f"[+] Status: Compressing file '{filename}' into '{zip_filename}'...")

    with zipfile.ZipFile(path.join(target_path, zip_filename), 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(path.join(target_path, filename), basename(path.join(target_path, filename)))

    if verbose:
        print(f"[+] Success: Created zip file '{zip_filename}' in '{target_path}'")

    filesize = os.path.getsize(path.join(target_path, zip_filename))

    if verbose:
        print(f"[+] Info: Size of '{zip_filename}' is {filesize / 1000000} MB")

    return zip_filename, filesize


def unzip_file(connection, zip_filename, target_path, verbose=False):
    """
    Unzips the specified file. If an error occurs during extraction, a 'Resend' message is sent to the client.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        zip_filename (str): The name of the zip file to extract.
        target_path (str): The path where the file should be extracted.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
    """
    if verbose:
        print(f"[+] Status: Extracting file '{zip_filename}'...")

    with zipfile.ZipFile(path.join(target_path, zip_filename), 'r') as zip_ref:
        try:
            zip_ref.extractall(path=target_path)
        except Exception as e:
            if verbose:
                print(f"[!] Error: Failed to extract zip file due to: {e}")
            send_msg(connection, "Resend", verbose)

    if verbose:
        print(f"[+] Success: Extracted file '{zip_filename}' to '{target_path}'")


def send_msg(connection, msg, verbose=False):
    """
    Sends a message to the client through a given connection.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        msg (str): The message to be sent.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
    """

    if verbose:
        print(f"[+] Status: Sending message '{msg}' ...")

    msg = pickle.dumps(msg)
    connection.sendall(msg)

    if verbose:
        print("[+] Success: Message sent.")


def receive_msg(connection, verbose=False):
    """
    Receives a message from the client through a given connection.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        str or None: The received message or None if an error occurred or no response received within the timeout period
    """

    received_data, status = recv(connection=connection, verbose=verbose)

    if status == 0:
        connection.close()
        return None

    if verbose:
        print(f"[+] Success: Received message '{received_data}' from the client.")
    return received_data


def recv(connection, verbose=False):
    """
    Helper function to receive and decode a message from the client.

    Parameters:
        connection (socket.pyi): The socket object representing the connection.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        tuple: Received message (or None if no message was received or an error occurred) and
        status code (0 for failure or 1 for success).
    """

    recv_start_time = time.time()
    received_data = b""
    buffer_size = 1024
    recv_timeout = 10000

    while True:
        try:
            data = connection.recv(buffer_size)
            received_data += data

            if not data:  # No data received from the client.
                if (time.time() - recv_start_time) > recv_timeout:  # Timeout period has passed.
                    if verbose:
                        print(f"[!] Error: Connection closed due to inactivity for {recv_timeout} seconds.")
                    return None, 0
            elif str(data)[-2] == '.':  # Message ends with a period (assumes this indicates the end of the message).
                if verbose:
                    print(f"[+] Status: All data ({len(received_data)} bytes) received.")

                if received_data:
                    try:
                        return pickle.loads(received_data), 1  # Successful receipt and decoding of data.
                    except Exception as e:
                        if verbose:
                            print(f"[!] Error: Failed to decode client's data due to: {e}")
                        return None, 0
            else:  # Data has been received, reset the timeout counter.
                recv_start_time = time.time()

        except Exception as e:
            if verbose:
                print(f"[!] Error: Failed to receive data from the client due to: {e}")
            return None, 0
