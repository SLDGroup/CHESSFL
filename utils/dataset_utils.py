import os
from torchvision import datasets, transforms
import numpy as np
import _pickle as cpickle
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
import random
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from utils.general_utils import seed_everything
from collections import Counter
import torch

class CustomSVHN(SVHN):
    def __init__(self, dataset=None, imgs=None, labels=None, is_unlabeled=False, rotate=False):
        if dataset is not None:
            self.data = [item[0] for item in dataset]
            self.targets = [item[1] for item in dataset]
        else:
            self.data = imgs
            self.targets = labels
        self.is_unlabeled = is_unlabeled
        self.rotate = rotate
        self.transform_weak = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                                                       std=[0.1980, 0.2010, 0.1970])])
        self.transform_strong = transforms.Compose([transforms.RandAugment(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                                                         std=[0.1980, 0.2010, 0.1970])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = transforms.ToPILImage()(self.data[idx])

        classes = ('0', '90', '180', '270', 'hflip', 'vflip')
        tf_type = random.randint(0, len(classes) - 1)
        if tf_type == 0:
            img_rot = img
        elif tf_type == 1:
            img_rot = transforms.functional.rotate(img, 90)
        elif tf_type == 2:
            img_rot = transforms.functional.rotate(img, 180)
        elif tf_type == 3:
            img_rot = transforms.functional.rotate(img, 270)
        elif tf_type == 4:
            img_rot = transforms.functional.hflip(img)
        elif tf_type == 5:
            img_rot = transforms.functional.rotate(img, 180)
            img_rot = transforms.functional.hflip(img_rot)

        img_w = self.transform_weak(img)

        if not self.rotate:
            if not self.is_unlabeled:
                return img_w, target
            else:
                img_s = self.transform_strong(img)
                return img_w, img_s, target
        else:
            if not self.is_unlabeled:
                img_rot = self.transform_weak(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type
            else:
                img_rot = self.transform_strong(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type

class CustomCIFAR100(CIFAR100):
    def __init__(self, dataset=None, imgs=None, labels=None, is_unlabeled=False, rotate=False):
        if dataset is not None:
            self.data = [item[0] for item in dataset]
            self.targets = [item[1] for item in dataset]
        else:
            self.data = imgs
            self.targets = labels
        self.is_unlabeled = is_unlabeled
        self.rotate = rotate
        self.transform_weak = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                                       std=[0.2673, 0.2564, 0.2762])])
        self.transform_strong = transforms.Compose([transforms.RandAugment(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                                                         std=[0.2673, 0.2564, 0.2762])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = transforms.ToPILImage()(self.data[idx])

        classes = ('0', '90', '180', '270', 'hflip', 'vflip')
        tf_type = random.randint(0, len(classes) - 1)
        if tf_type == 0:
            img_rot = img
        elif tf_type == 1:
            img_rot = transforms.functional.rotate(img, 90)
        elif tf_type == 2:
            img_rot = transforms.functional.rotate(img, 180)
        elif tf_type == 3:
            img_rot = transforms.functional.rotate(img, 270)
        elif tf_type == 4:
            img_rot = transforms.functional.hflip(img)
        elif tf_type == 5:
            img_rot = transforms.functional.rotate(img, 180)
            img_rot = transforms.functional.hflip(img_rot)

        img_w = self.transform_weak(img)

        if not self.rotate:
            if not self.is_unlabeled:
                return img_w, target
            else:
                img_s = self.transform_strong(img)
                return img_w, img_s, target
        else:
            if not self.is_unlabeled:
                img_rot = self.transform_weak(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type
            else:
                img_rot = self.transform_strong(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type

class CustomCIFAR10(CIFAR10):
    def __init__(self, dataset=None, imgs=None, labels=None, is_unlabeled=False, rotate=False):
        if dataset is not None:
            self.data = [item[0] for item in dataset]
            self.targets = [item[1] for item in dataset]
        else:
            self.data = imgs
            self.targets = labels
        self.is_unlabeled = is_unlabeled
        self.rotate = rotate
        self.transform_weak = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                       std=[0.2023, 0.1994, 0.2010])])
        self.transform_strong = transforms.Compose([transforms.RandAugment(),
                                                    transforms.RandomCrop(32, padding=4),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                         std=[0.2023, 0.1994, 0.2010])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = transforms.ToPILImage()(self.data[idx])

        classes = ('0', '90', '180', '270', 'hflip', 'vflip')
        tf_type = random.randint(0, len(classes) - 1)
        if tf_type == 0:
            img_rot = img
        elif tf_type == 1:
            img_rot = transforms.functional.rotate(img, 90)
        elif tf_type == 2:
            img_rot = transforms.functional.rotate(img, 180)
        elif tf_type == 3:
            img_rot = transforms.functional.rotate(img, 270)
        elif tf_type == 4:
            img_rot = transforms.functional.hflip(img)
        elif tf_type == 5:
            img_rot = transforms.functional.rotate(img, 180)
            img_rot = transforms.functional.hflip(img_rot)

        img_w = self.transform_weak(img)

        if not self.rotate:
            if not self.is_unlabeled:
                return img_w, target
            else:
                img_s = self.transform_strong(img)
                return img_w, img_s, target
        else:
            if not self.is_unlabeled:
                img_rot = self.transform_weak(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type
            else:
                img_rot = self.transform_strong(img_rot)
                img_s = self.transform_strong(img)
                return img_w, img_s, target, img_rot, tf_type


def split_dataset(dataset_name, num_labeled):
    if dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root=f'data/{dataset_name}', train=True, download=True, transform=transforms.ToTensor())
        num_classes = 10
    elif dataset_name == "cifar100":
        dataset = datasets.CIFAR100(root=f'data/{dataset_name}', train=True, download=True, transform=transforms.ToTensor())
        num_classes = 100
    elif dataset_name == "svhn":
        dataset = datasets.SVHN(root=f'data/{dataset_name}', split="train", download=True, transform=transforms.ToTensor())
        num_classes = 10
    # Build an index for each class
    class_indices = {i: [] for i in range(num_classes)}  # CIFAR10 has 10 classes
    for index, (_, label) in enumerate(dataset):
        class_indices[label].append(index)

    # Sample the desired number of images from each class
    samples_per_class = num_labeled // num_classes
    labeled_indices = []
    for label, indices in class_indices.items():
        labeled_indices.extend(indices[:samples_per_class])

    unlabeled_indices = list(set(range(len(dataset))) - set(labeled_indices))

    lab_dataset = [dataset[i] for i in labeled_indices]
    unlab_dataset = [dataset[i] for i in unlabeled_indices]

    return lab_dataset, unlab_dataset


def sanity_check_split_dataset(lab_dataset, unlab_dataset, dataset_name):
    if dataset_name == "cifar10" or dataset_name == "svhn":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    # Total number of images in labeled and unlabeled datasets
    print(f"Total labeled images: {len(lab_dataset)}")
    print(f"Total unlabeled images: {len(unlab_dataset)}")

    # Initialize class counters for labeled and unlabeled datasets
    class_counts_lab = {i: 0 for i in range(num_classes)}
    class_counts_unlab = {i: 0 for i in range(num_classes)}

    # Count class occurrences in labeled dataset
    for _, label in lab_dataset:
        class_counts_lab[label] += 1

    # Count class occurrences in unlabeled dataset
    for _, label in unlab_dataset:
        class_counts_unlab[label] += 1

    # Display class counts for labeled dataset
    print("Class counts (labeled):")
    for class_num, count in class_counts_lab.items():
        print(f"\tClass {class_num}: {count}")

    # Display class counts for unlabeled dataset
    print("Class counts (unlabeled):")
    for class_num, count in class_counts_unlab.items():
        print(f"\tClass {class_num}: {count}")


def split_labeled_data(labeled_dataset, num_edge_servers, save_path):
    # Get the total count of each class in the dataset
    class_counts = defaultdict(int)
    for _, label in labeled_dataset:
        class_counts[label] += 1

    # Ensure each class has enough samples for a balanced split
    for class_label, count in class_counts.items():
        assert count % num_edge_servers == 0, \
            f"Class {class_label} has {count} samples, which is not divisible by the number of edge servers {num_edge_servers}"

    # Determine the number of samples of each class per server
    samples_per_class_per_server = {class_label: count // num_edge_servers for class_label, count in
                                    class_counts.items()}

    # Create lists of indices for each server
    server_indices = [[] for _ in range(num_edge_servers)]
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(labeled_dataset):
        class_indices[label].append(idx)

    for class_label, indices in class_indices.items():
        # Split the indices of each class among the servers
        samples_per_server = samples_per_class_per_server[class_label]
        for i in range(num_edge_servers):
            start_idx = i * samples_per_server
            end_idx = (i + 1) * samples_per_server
            server_indices[i].extend(indices[start_idx:end_idx])

    # Create a data loader for each server and save to a .pkl file
    for i, indices in enumerate(server_indices):
        edge_server_dataset = Subset(labeled_dataset, indices)
        edge_data = [item[0] for item in edge_server_dataset]
        edge_labels = [item[1] for item in edge_server_dataset]
        with open(f"{save_path}/imgs_train_edge{i}.pkl", 'wb') as f:
            cpickle.dump(edge_data, f)
        with open(f"{save_path}/labels_train_edge{i}.pkl", 'wb') as f:
            cpickle.dump(edge_labels, f)

def split_labeled_data_niid(labeled_dataset, num_edge_servers, dataset_name, base_dir, labeled=4000):
    alpha = 0.1
    if dataset_name == "cifar10":
        n_classes = 10
        num_imgs_max = labeled / num_edge_servers
    elif dataset_name == "cifar100":
        n_classes = 100
        num_imgs_max = labeled / num_edge_servers
    elif  dataset_name == "svhn":
        n_classes = 10
        num_imgs_max = labeled / num_edge_servers
    print("\nSampling configuration:")
    print(f"\tDataset: {dataset_name}")
    print(f"\tNumber of edge servers: {num_edge_servers}")
    print(f"\tDistribute Non-IID")
    print(f"\tWriting data at this location: {base_dir}")
    print(f"\tAlpha for Dirichlet distribution: {alpha}")

    n_samples_train = len(labeled_dataset)

    unlab_data = [sample[0] for sample in labeled_dataset]
    unlab_targets = [sample[1] for sample in labeled_dataset]
    all_ids_train = np.array(unlab_targets)

    class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in range(n_classes)}

    dist_of_client = np.random.dirichlet(np.repeat(alpha, num_edge_servers), size=n_classes).transpose()
    dist_of_client /= dist_of_client.sum()

    for i in range(100):
        s0 = dist_of_client.sum(axis=0, keepdims=True)
        s1 = dist_of_client.sum(axis=1, keepdims=True)
        dist_of_client /= s0
        dist_of_client /= s1

    samples_per_class_train = (np.floor(dist_of_client * n_samples_train))

    # NEW: Adjust samples_per_class_train to ensure num_imgs_max constraint
    for client_num in range(num_edge_servers):
        total_imgs_for_client = samples_per_class_train[client_num].sum()
        if total_imgs_for_client > num_imgs_max:
            scale_factor = num_imgs_max / total_imgs_for_client
            samples_per_class_train[client_num] = np.floor(samples_per_class_train[client_num] * scale_factor)

    start_ids_train = np.zeros((num_edge_servers + 1, n_classes), dtype=np.int32)
    for i in range(0, num_edge_servers):
        start_ids_train[i + 1] = start_ids_train[i] + samples_per_class_train[i]

    print("\nSanity checks:")
    print(f"\tSum of dist. of classes over clients: {dist_of_client.sum(axis=0)}")
    print(f"\tSum of dist. of clients over classes: {dist_of_client.sum(axis=1)}")
    print(f"\tTotal trainset size: {samples_per_class_train.sum()}")

    client_ids = {client_num: [] for client_num in range(num_edge_servers)}
    for client_num in range(num_edge_servers):
        l = np.array([], dtype=np.int32)
        for class_num in range(n_classes):
            start, end = start_ids_train[client_num, class_num], start_ids_train[client_num + 1, class_num]
            l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
        client_ids[client_num] = l

    print("\nDistribution over classes:")
    for client_num in range(num_edge_servers):
        client_dataset = Subset(labeled_dataset, client_ids[client_num])
        client_data = [item[0] for item in client_dataset]
        client_labels = [item[1] for item in client_dataset]
        with open(f"{base_dir}/imgs_train_edge{client_num}_niid.pkl", 'wb') as f:
            cpickle.dump(client_data, f)
        with open(f"{base_dir}/labels_train_edge{client_num}_niid.pkl", 'wb') as f:
            cpickle.dump(client_labels, f)
        print(f"\tClient {client_num}: \n \t\t Train: {samples_per_class_train[client_num]} \n \t\t Total: {samples_per_class_train[client_num].sum()}")
    print("[+] Datasets for each user saved successfully.")


def get_datasets(dataset_name="cifar10", num_users=10, num_edge_servers=5, seed=42, labeled=4000, num_imgs_max=500):
    seed_everything(seed=seed)

    base_dir = f"dataset/{dataset_name}/{seed}_{labeled}"
    os.makedirs(f"{base_dir}/iid", exist_ok=True)
    os.makedirs(f"{base_dir}/niid", exist_ok=True)

    lab_dataset, unlab_dataset = split_dataset(dataset_name=dataset_name, num_labeled=labeled)
    sanity_check_split_dataset(lab_dataset, unlab_dataset, dataset_name)
    lab_data = [item[0] for item in lab_dataset]
    lab_labels = [item[1] for item in lab_dataset]
    with open(f"{base_dir}/lab_imgs.pkl", 'wb') as f:
        cpickle.dump(lab_data, f)
    with open(f"{base_dir}/lab_labels.pkl", 'wb') as f:
        cpickle.dump(lab_labels, f)
    unlab_data = [item[0] for item in unlab_dataset]
    unlab_labels = [item[1] for item in unlab_dataset]
    with open(f"{base_dir}/unlab_imgs.pkl", 'wb') as f:
        cpickle.dump(unlab_data, f)
    with open(f"{base_dir}/unlab_labels.pkl", 'wb') as f:
        cpickle.dump(unlab_labels, f)

    split_labeled_data(lab_dataset, num_edge_servers, save_path=base_dir)

    if dataset_name == "cifar10":
        n_classes = 10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        test_dataset = datasets.CIFAR10(root=f'data/{dataset_name}', train=False, download=True, transform=transform_test)
    elif dataset_name == "cifar100":
        n_classes = 100
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409),
                                 (0.2673, 0.2564, 0.2762))
        ])
        test_dataset = datasets.CIFAR100(root=f'data/{dataset_name}', train=False, download=True, transform=transform_test)
    elif  dataset_name == "svhn":
        n_classes = 10
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970))
        ])
        test_dataset = datasets.SVHN(root=f'data/{dataset_name}', split="test", download=True, transform=transform_test)

    with open(f"{base_dir}/iid/global_test.pkl", 'wb') as f:
        cpickle.dump(test_dataset, f)

    # Get training data split between num_users both iid and niid
    for data_iid in [True, False]:
        alpha = 100 if data_iid else 0.1
        iidtype = 'iid' if data_iid else 'niid'
        os.makedirs(base_dir, exist_ok=True)

        print("\nSampling configuration:")
        print(f"\tDataset: {dataset_name}")
        print(f"\tNumber of clients: {num_users}")
        print(f"\tDistribute IID: {data_iid}")
        print(f"\tWriting data at this location: {base_dir}")
        print(f"\tAlpha for Dirichlet distribution: {alpha}")

        n_samples_train = len(unlab_dataset)

        unlab_data = [sample[0] for sample in unlab_dataset]
        unlab_targets = [sample[1] for sample in unlab_dataset]
        all_ids_train = np.array(unlab_targets)


        class_ids_train = {class_num: np.where(all_ids_train == class_num)[0] for class_num in range(n_classes)}

        dist_of_client = np.random.dirichlet(np.repeat(alpha, num_users), size=n_classes).transpose()
        dist_of_client /= dist_of_client.sum()

        for i in range(100):
            s0 = dist_of_client.sum(axis=0, keepdims=True)
            s1 = dist_of_client.sum(axis=1, keepdims=True)
            dist_of_client /= s0
            dist_of_client /= s1

        samples_per_class_train = (np.floor(dist_of_client * n_samples_train))

        # NEW: Adjust samples_per_class_train to ensure num_imgs_max constraint
        for client_num in range(num_users):
            total_imgs_for_client = samples_per_class_train[client_num].sum()
            if total_imgs_for_client > num_imgs_max:
                scale_factor = num_imgs_max / total_imgs_for_client
                samples_per_class_train[client_num] = np.floor(samples_per_class_train[client_num] * scale_factor)

        start_ids_train = np.zeros((num_users + 1, n_classes), dtype=np.int32)
        for i in range(0, num_users):
            start_ids_train[i + 1] = start_ids_train[i] + samples_per_class_train[i]

        print("\nSanity checks:")
        print(f"\tSum of dist. of classes over clients: {dist_of_client.sum(axis=0)}")
        print(f"\tSum of dist. of clients over classes: {dist_of_client.sum(axis=1)}")
        print(f"\tTotal trainset size: {samples_per_class_train.sum()}")

        client_ids = {client_num: [] for client_num in range(num_users)}
        for client_num in range(num_users):
            l = np.array([], dtype=np.int32)
            for class_num in range(n_classes):
                start, end = start_ids_train[client_num, class_num], start_ids_train[client_num + 1, class_num]
                l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
            client_ids[client_num] = l

        print("\nDistribution over classes:")
        for client_num in range(num_users):
            client_dataset = Subset(unlab_dataset, client_ids[client_num])
            client_data = [item[0] for item in client_dataset]
            client_labels = [item[1] for item in client_dataset]
            with open(f"{base_dir}/{iidtype}/imgs_train_dev{client_num}.pkl", 'wb') as f:
                cpickle.dump(client_data, f)
            with open(f"{base_dir}/{iidtype}/labels_train_dev{client_num}.pkl", 'wb') as f:
                cpickle.dump(client_labels, f)
            print(f"\tClient {client_num}: \n \t\t Train: {samples_per_class_train[client_num]} \n \t\t Total: {samples_per_class_train[client_num].sum()}")
    print("[+] Datasets for each user saved successfully.")
    return "done"


def load_data(data_iid=True, dev_idx=-1, dataset_name="cifar10", seed=42, test_global=False, is_unlabeled=False,
              is_edge_server=False, lab_dataset=False, unlab_dataset=False, batch_size=32, labeled=4000, rotate=False):
    base_dir = f"dataset/{dataset_name}/{seed}_{labeled}"
    if test_global:
        with open(f"{base_dir}/iid/global_test.pkl", 'rb') as f:
            dataset = cpickle.load(f)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2, worker_init_fn=np.random.seed(seed))
    elif is_edge_server:
        with open(f"{base_dir}/imgs_train_edge{dev_idx}.pkl", 'rb') as f:
            imgs = cpickle.load(f)
        with open(f"{base_dir}/labels_train_edge{dev_idx}.pkl", 'rb') as f:
            labels = cpickle.load(f)
        if dataset_name == "cifar10":
            data_loader = DataLoader(CustomCIFAR10(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "cifar100":
            data_loader = DataLoader(CustomCIFAR100(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "svhn":
            data_loader = DataLoader(CustomSVHN(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
    elif lab_dataset:
        with open(f"{base_dir}/lab_imgs.pkl", 'rb') as f:
            imgs = cpickle.load(f)
        with open(f"{base_dir}/lab_labels.pkl", 'rb') as f:
            labels = cpickle.load(f)
        if dataset_name == "cifar10":
            data_loader = DataLoader(CustomCIFAR10(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "cifar100":
            data_loader = DataLoader(CustomCIFAR100(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "svhn":
            data_loader = DataLoader(CustomSVHN(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
    elif unlab_dataset:
        with open(f"{base_dir}/unlab_imgs.pkl", 'rb') as f:
            imgs = cpickle.load(f)
        with open(f"{base_dir}/unlab_labels.pkl", 'rb') as f:
            labels = cpickle.load(f)
        if dataset_name == "cifar10":
            data_loader = DataLoader(CustomCIFAR10(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "cifar100":
            data_loader = DataLoader(CustomCIFAR100(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "svhn":
            data_loader = DataLoader(CustomSVHN(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
    else:
        iidtype = "iid" if data_iid else "niid"
        with open(f"{base_dir}/{iidtype}/imgs_train_dev{dev_idx}.pkl", 'rb') as f:
            imgs = cpickle.load(f)
        with open(f"{base_dir}/{iidtype}/labels_train_dev{dev_idx}.pkl", 'rb') as f:
            labels = cpickle.load(f)
        if dataset_name == "cifar10":
            data_loader = DataLoader(CustomCIFAR10(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "cifar100":
            data_loader = DataLoader(CustomCIFAR100(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))
        elif dataset_name == "svhn":
            data_loader = DataLoader(CustomSVHN(dataset=None, imgs=imgs, labels=labels, is_unlabeled=is_unlabeled, rotate=rotate),
                                     batch_size=batch_size, shuffle=True, num_workers=1, worker_init_fn=np.random.seed(seed))

    return data_loader


def get_fixmix(data_loader, model, cuda_name, batch_size=32, shuffle=True, num_workers=1, seed=42):
    device = torch.device(cuda_name)
    model.eval()  # Set the model to evaluation mode
    all_probs = []
    original_dataset = data_loader.dataset

    with torch.no_grad():
        for images, _, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs)

    all_probs = torch.cat(all_probs)
    # Get the indices of images with max probability >= 0.95
    high_prob_indices = all_probs.max(dim=1).values.ge(0.95).nonzero(as_tuple=True)[0]

    # Safety feature
    if len(high_prob_indices) == 0:
        return None, None

    # Extract data and labels using torch indexing
    high_prob_indices_list = high_prob_indices.tolist()
    fix_data = torch.stack([original_dataset.data[i] for i in high_prob_indices_list])
    fix_labels = torch.stack([torch.tensor(original_dataset.targets[i]) for i in high_prob_indices_list])

    fix_dataset = CustomCIFAR10(imgs=fix_data, labels=fix_labels, is_unlabeled=True)
    fix_dataloader = DataLoader(fix_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=np.random.seed(seed))

    # Creating mix_dataloader using torch.randint for faster index sampling
    sampled_indices = torch.randint(0, len(fix_dataset), (len(fix_dataset),)).tolist()
    subset_dataset = Subset(fix_dataset, sampled_indices)
    mix_dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=np.random.seed(seed))

    return fix_dataloader, mix_dataloader


def sanity_check_load_data(dev_idx):
    data_loader = load_data(dev_idx=dev_idx)

    # Counter for images per class
    class_counter = Counter()

    total_images = 0
    for images, labels in data_loader:
        print(len(images))
        total_images += len(images)
        class_counter.update(labels.numpy())

    print(f"Total number of images: {total_images}")
    print(f"Images per class: {dict(class_counter)}")


