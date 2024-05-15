import copy
import pickle
import time
from os import path
import os
import socket
import zipfile
import paramiko
from scp import SCPClient
import sys
from torchvision import datasets, transforms
from os.path import basename
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from models.get_model import get_model
import _pickle as cPickle
from tqdm import tqdm
from torch.utils.data import random_split
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.stats import mode
from collections import Counter

# Define the loss
class SimLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output1, act1, output2, act2, label, alpha):
        cross = nn.CrossEntropyLoss()
        # Calculate the cosine similarity between the two vectors
        cosine_similarity = F.cosine_similarity(act1, act2)

        # Define the cosine similarity loss as 1 - cosine_similarity
        # This is because the cosine similarity ranges from -1 to 1, where 1 means the vectors are identical
        # So by subtracting it from 1, we get a loss that is 0 when the vectors are identical and greater than 0 otherwise
        cosine_similarity_loss = 1 - cosine_similarity.mean()
        # print(act1.shape)
        # print(act2.shape)
        # print(cosine_similarity)
        sim_loss = alpha * cosine_similarity_loss + (1-alpha)*(cross(output1, label) + cross(output2, label))
        # print(sim_loss)
        return sim_loss


def get_loss_func(loss_func_name):
    if loss_func_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_func_name == "sim_loss":
        return SimLoss()

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        for i in range(len(self.idxs)):
            self.idxs[i] = int(self.idxs[i])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def train(model, loss_func, dev_idx, batch_size, num_workers, model_path, cuda_name, optimizer, local_epochs,
          verbose, dataset_name, seed, data_iid, comm_round):
    labeled = 20
    confidence_level = 5
    e = 599
    device = torch.device(cuda_name)
    if True:  # TODO for some reason it takes them as non-iid Double check in the cloud
        iidtype = "iid"
    else:
        iidtype = "niid"
    with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}/imgs_train_dev{dev_idx}.pkl", 'rb') as f:
        imgs_train = cPickle.load(f)
    with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}/labels_train_dev{dev_idx}.pkl", 'rb') as f:
        labels_train = cPickle.load(f)
    with open(f"dataset/{dataset_name}/iid/seed{seed}/transform_train.pkl", 'rb') as f:
        transform_train = cPickle.load(f)
    data_loader = DataLoader(DatasetSplitDirichlet(image=imgs_train, target=labels_train, transform=transform_train),
                             batch_size=1, shuffle=True, num_workers=num_workers)
    # with open(f"dataset/cifar10/iid/seed42_lab{labeled}/global_centroids_pre{e}_{comm_round}.pkl", 'rb') as f:
    with open(f"dataset/cifar10/iid/seed42_lab{labeled}/global_centroids_pre{e}_{comm_round}.pkl", 'rb') as f:
        mydict = cPickle.load(f)
        labels_pretrain = mydict["labels_pretrain"]
        centroids = mydict["centroids"]

    model.train()
    data_all_classes_client = {i: [] for i in range(10)}
    labels_all_classes_client = {i: [] for i in range(10)}

    # Extract features for all classes
    for image, label in data_loader:
        # image = image.unsqueeze(0)  # add batch dimension
        label = label.detach().cpu().numpy()[0]
        if torch.cuda.is_available():
            image = image.cuda()
        output, feature = model(image)
        feature = feature.detach().cpu().numpy()
        data_all_classes_client[label].append(feature.squeeze())
        labels_all_classes_client[label].append(label)
    data_client = np.concatenate([np.array(data_all_classes_client[i]) for i in range(10)])
    labels_client = np.concatenate([np.array(labels_all_classes_client[i]) for i in range(10)])

    accuracy = []
    # Compute the distance from each local data point to each centroid
    distances = cdist(data_client, centroids)

    # Find the index of the closest centroid for each local data point
    closest_centroid_indices = np.argmin(distances, axis=1)

    # Get the minimum distance for each local data point to its closest centroid
    min_distances = np.min(distances, axis=1)

    # Calculate the threshold for the top confidence_level% smallest distances
    threshold = np.percentile(min_distances, confidence_level)

    # Select only the local data points that are within the top confidence_level% smallest distances
    top_indices = np.where(min_distances <= threshold)

    # Use these indices to select the corresponding pseudo-labels
    top_pseudo_labels = np.array(labels_pretrain)[closest_centroid_indices[top_indices]]

    # # Select only the data_client that had the top pseudo labels
    # top_data_client = data_client[top_indices]
    # kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(top_data_client)
    # for_save_centroids = kmeans.cluster_centers_
    #
    # # Compute the distance from each new centroid to each old centroid
    # distances_centroids = cdist(for_save_centroids, centroids)
    #
    # # Find the index of the closest old centroid for each new centroid
    # closest_old_centroid_indices = np.argmin(distances_centroids, axis=1)
    #
    # # Use these indices to select the corresponding pseudo-labels
    # labels_pretrain = np.array(labels_pretrain)
    # for_save_labels_client = labels_pretrain[closest_old_centroid_indices]
    #
    # # for_save_labels_client = [mode(labels_client[kmeans.labels_ == i])[0][0] for i in range(10)]
    # with open(f"dataset/cifar10/iid/seed42_lab{labeled}/dev{dev_idx}_clusters.pkl", 'wb') as f:
    #     cPickle.dump({"labels_client": for_save_labels_client.tolist(), "centroids_client": for_save_centroids}, f)

    # Compare these to the true labels to get the number of correct pseudo-labels
    correct_labels = np.sum(top_pseudo_labels == labels_client[top_indices])
    total_labels = len(top_pseudo_labels)

    # Compute the accuracy of pseudo-labels
    accuracy.append(correct_labels / total_labels if total_labels > 0 else 0)

    # print(f"Dev{dev_idx} Accuracy of pseudo-labels top {confidence_level}%: {np.mean(accuracy) * 100:.2f} \u00B1 {np.std(accuracy) * 100:.2f}%")

    # Filter the images to retain only those that fall within the top confidence_level% distances
    selected_images = np.array(imgs_train)[top_indices]

    # Convert the selected images and pseudo-labels to torch tensors
    selected_images_tensor = torch.tensor(selected_images, dtype=torch.float32)
    top_pseudo_labels_tensor = torch.tensor(top_pseudo_labels, dtype=torch.long)

    # Create a new dataset with these selected images and their pseudo-labels
    new_dataset = TensorDataset(selected_images_tensor, top_pseudo_labels_tensor)

    # Create a new DataLoader with this new dataset
    new_data_loader = DataLoader(new_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # new_data_loader_after = DataLoader(new_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    # print("before")
    for epoch in range(local_epochs):
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 1
        for batch_idx, (images, labels) in enumerate(new_data_loader):
            images, labels = images.to(device), labels.to(device)

            outputs, act = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        # if verbose:
        # print(f"Dev{dev_idx} Epoch {epoch} loss: {train_loss/(batch_idx+1)} accuracy: {100.*correct/total}")
    # print("after")
    torch.save(model.state_dict(), model_path)

    data_all_classes_client = {i: [] for i in range(10)}
    labels_all_classes_client = {i: [] for i in range(10)}

    # Extract features for all classes
    for image, label in data_loader:
        # image = image.unsqueeze(0)  # add batch dimension
        label = label.detach().cpu().numpy()[0]
        if torch.cuda.is_available():
            image = image.cuda()
        output, feature = model(image)
        feature = feature.detach().cpu().numpy()
        data_all_classes_client[label].append(feature.squeeze())
        labels_all_classes_client[label].append(label)
    data_client = np.concatenate([np.array(data_all_classes_client[i]) for i in range(10)])
    labels_client = np.concatenate([np.array(labels_all_classes_client[i]) for i in range(10)])

    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(data_client)
    for_save_centroids = kmeans.cluster_centers_

    # Compute the distance from each new centroid to each old centroid
    distances_centroids = cdist(for_save_centroids, centroids)

    # Find the index of the closest old centroid for each new centroid
    closest_old_centroid_indices = np.argmin(distances_centroids, axis=1)

    # Use these indices to select the corresponding pseudo-labels
    labels_pretrain = np.array(labels_pretrain)
    for_save_labels_client = labels_pretrain[closest_old_centroid_indices]

    # for_save_labels_client = [mode(labels_client[kmeans.labels_ == i])[0][0] for i in range(10)]
    with open(f"dataset/cifar10/iid/seed42_lab{labeled}/dev{dev_idx}_clusters.pkl", 'wb') as f:
        cPickle.dump({"labels_client": for_save_labels_client.tolist(), "centroids_client": for_save_centroids}, f)

    return train_loss/(batch_idx+1), (100.*correct/total)


def test(model, loss_func, dev_idx, batch_size, num_workers, cuda_name, test_global, verbose,
         dataset_name, seed, data_iid, return_total=False):

    device = torch.device(cuda_name)
    model = model.to(device)

    if test_global:
        with open(f"dataset/{dataset_name}/iid/seed{seed}/global_test.pkl", 'rb') as f:
            dataset_test = cPickle.load(f)
        data_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        if data_iid:
            iidtype = "iid"
        else:
            iidtype = "niid"
        with open(f"dataset/{dataset_name}/iid/seed{seed}/transform_test.pkl", 'rb') as f:
            transform_test = cPickle.load(f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_test_dev{dev_idx}.pkl", 'rb') as f:
            imgs_test = cPickle.load(f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_test_dev{dev_idx}.pkl", 'rb') as f:
            labels_test = cPickle.load(f)
        data_loader = DataLoader(DatasetSplitDirichlet(image=imgs_test, target=labels_test, transform=transform_test),
                                 batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 1

    if not test_global and verbose:
        print(f"[++] Testing on local dataset... ")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs, act = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if verbose:
            print(f'[++] Local model loss {test_loss/(batch_idx+1)}, Local model accuracy: {100.*correct/total}%')
    if not test_global and verbose:
        print(f"[++] Testing on local dataset... ")
        print(f"[++] Finished testing in {time.time() - start_time}")

    if return_total:
        return test_loss/(batch_idx+1), 100.*correct/total, total
    else:
        return test_loss/(batch_idx+1), 100.*correct/total


def local_training(model_name, dataset_name, loss_func, batch_size, num_workers, model_path, local_testing, cuda_name,
                   learning_rate, momentum, local_epochs, log_train_time, dev_idx, verbose, data_iid, seed, comm_round):

    device = torch.device(cuda_name)
    model = get_model(model_name=f"{dataset_name}_{model_name}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    if verbose:
        print(f"[++] Device{dev_idx} training...")
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    if log_train_time:
        start_time = time.time()
    train_loss, train_acc = train(model=model, loss_func=loss_func, batch_size=batch_size, num_workers=num_workers,
                                  model_path=model_path, cuda_name=cuda_name, optimizer=optimizer,
                                  local_epochs=local_epochs, verbose=verbose, dataset_name=dataset_name, seed=seed,
                                  data_iid=data_iid, dev_idx=dev_idx, comm_round=comm_round)
    if log_train_time:
        train_time = time.time() - start_time
    if verbose:
        print(f"[++] Train loss: {train_loss}")

    test_loss, test_acc = None, None
    if local_testing:
        if verbose:
            print("[++] Evaluating local accuracy after training...")
        test_loss, test_acc = test(model=model, loss_func=loss_func, batch_size=batch_size, num_workers=num_workers,
                                   cuda_name=cuda_name, test_global=False, verbose=verbose, dataset_name=dataset_name,
                                   seed=seed, data_iid=data_iid, dev_idx=dev_idx)
        print(f"Train loss {train_loss} Test loss {test_loss} Train acc {train_acc} Test acc {test_acc}")
    return train_loss, train_acc, test_loss, test_acc, train_time


class DatasetSplitDirichlet(Dataset):
    def __init__(self, image, target, transform):
        self.image = image
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index]
        image = image / 255.
        transform = transforms.ToPILImage()
        image = transform(image)
        image = self.transform(image)
        target = self.target[index]
        return image, target


def dirichlet_test(dataset, num_users, images_per_client, alpha, dataset_name):
    num_classes = len(dataset.classes)
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        idx = [torch.where(torch.FloatTensor(dataset.targets) == i) for i in range(num_classes)]
        data = [dataset.data[idx[i][0]] for i in range(num_classes)]
    else:
        idx = [torch.where(dataset.targets == i) for i in range(num_classes)]
        data = [dataset.data[idx[i]] for i in range(num_classes)]
    label = [torch.ones(len(data[i])) * i for i in range(num_classes)]

    s = np.random.dirichlet(np.ones(num_classes) * alpha, num_users)
    data_dist = np.zeros((num_users, num_classes))

    for j in range(num_users):
        data_dist[j] = (
                (s[j] * images_per_client).astype('int') / (s[j] * images_per_client).astype('int').sum() *
                images_per_client).astype('int')
        data_num = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0, high=num_classes)] += ((images_per_client - data_num))
        data_dist = data_dist.astype('int')

    X = []
    Y = []
    for j in range(num_users):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.randint(low=0, high=len(data[i]), size=data_dist[j][i])

                if dataset_name == "cifar10" or dataset_name == "cifar100":
                    x_data.append(torch.from_numpy(data[i][d_index]))
                else:
                    x_data.append(torch.unsqueeze(data[i][d_index],1))
                y_data.append(label[i][d_index])
        x_data = torch.cat(x_data).to(torch.float32)
        y_data = torch.cat(y_data).to(torch.int64)
        if dataset_name == "cifar10" or dataset_name == "cifar100":
            x_data = x_data.permute(0,3,1,2)
        X.append(x_data)
        Y.append(y_data)
    return X, Y


def dirichlet(original_dataset, subset, num_users, images_per_client, alpha, dataset_name):
    num_classes = len(original_dataset.classes)

    # Create a mapping from targets to indices in the subset
    targets = np.array(original_dataset.targets)[subset.indices]

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        idx = [np.where(targets == i)[0] for i in range(num_classes)]
        subset_data = original_dataset.data[subset.indices]
        data = [subset_data[idx[i]] for i in range(num_classes)]
    else:
        idx = [np.where(targets == i)[0] for i in range(num_classes)]
        subset_data = original_dataset.data[subset.indices]
        data = [subset_data[idx[i]] for i in range(num_classes)]
    label = [torch.ones(len(data[i])) * i for i in range(num_classes)]

    s = np.random.dirichlet(np.ones(num_classes) * alpha, num_users)
    data_dist = np.zeros((num_users, num_classes))

    for j in range(num_users):
        data_dist[j] = (
                (s[j] * images_per_client).astype('int') / (s[j] * images_per_client).astype('int').sum() *
                images_per_client).astype('int')
        data_num = data_dist[j].sum()
        data_dist[j][np.random.randint(low=0, high=num_classes)] += ((images_per_client - data_num))
        data_dist = data_dist.astype('int')

    X = []
    Y = []
    for j in range(num_users):
        x_data = []
        y_data = []
        for i in range(num_classes):
            if data_dist[j][i] != 0:
                d_index = np.random.choice(len(data[i]), size=min(len(data[i]), data_dist[j][i]), replace=False)
                if dataset_name == "cifar10" or dataset_name == "cifar100":
                    x_data.append(torch.from_numpy(data[i][d_index]))
                else:
                    x_data.append(torch.unsqueeze(torch.from_numpy(data[i][d_index]),1))
                y_data.append(label[i][d_index])
        x_data = torch.cat(x_data).to(torch.float32)
        y_data = torch.cat(y_data).to(torch.int64)
        if dataset_name == "cifar10" or dataset_name == "cifar100":
            x_data = x_data.permute(0,3,1,2)
        X.append(x_data)
        Y.append(y_data)
    return X, Y


def get_datasets(dataset_name, data_iid=True, num_users=2, global_data=False, seed=42, images_per_client=500,
                 dataset_train=None, labeled=10):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset_test = None
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                   std=[0.2023, 0.1994, 0.2010])])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                                 std=[0.2023, 0.1994, 0.2010])])
        if dataset_train is None:
            dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=False, transform=transform_test)
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                   (0.2675, 0.2565, 0.2761))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                  (0.2675, 0.2565, 0.2761))])
        if dataset_train is None:
            dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=False, transform=transform_test)
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        if dataset_train is None:
            dataset_train = datasets.MNIST('data/mnist', train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform_train)
        dataset_test = datasets.MNIST('data/mnist', train=False, download=False, transform=transform_test)
    elif dataset_name == "emnist":
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Pad(2),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.ToPILImage(),
                                             transforms.Pad(2),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        if dataset_train is None:
            dataset_train = datasets.EMNIST('data/emnist', train=True, download=True,
                                            transform=transform_train, split='byclass')
        else:
            dataset = datasets.EMNIST('data/emnist', train=True, download=True,
                                            transform=transform_train, split='byclass')

        dataset_test = datasets.EMNIST('data/emnist', train=False, download=False,
                                       transform=transform_test, split='byclass')

    if data_iid:
        iidtype = "iid"
    else:
        iidtype = "niid"
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    if not os.path.exists(f"dataset/{dataset_name}"):
        os.mkdir(f"dataset/{dataset_name}")
    if not os.path.exists(f"dataset/{dataset_name}/{iidtype}"):
        os.mkdir(f"dataset/{dataset_name}/{iidtype}")
    if not os.path.exists(f"dataset/{dataset_name}/{iidtype}/seed{seed}"):
        os.mkdir(f"dataset/{dataset_name}/{iidtype}/seed{seed}")
    if not os.path.exists(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}"):
        os.mkdir(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}")

    if global_data:
        # Calculate the number of samples that correspond to 10% and 90% of the data
        num_total = len(dataset_train)
        num_pretrain = int(labeled/100 * num_total)  # 10% of total data
        num_train_fl = num_total - num_pretrain  # 90% of total data

        # Split the dataset
        dataset_pretrain, dataset_train = random_split(dataset_train, [num_pretrain, num_train_fl], generator=generator)

        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/global_pretrain_{labeled}.pkl",'wb') as f:
            cPickle.dump(dataset_pretrain, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/global_train_{100-labeled}.pkl", 'wb') as f:
            cPickle.dump(dataset_train, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/global_test.pkl",'wb') as f:
            cPickle.dump(dataset_test, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/transform_train.pkl",'wb') as f:
            cPickle.dump(transform_train, f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/transform_test.pkl",'wb') as f:
            cPickle.dump(transform_test, f)
        return dataset_train

    if data_iid:
        imgs_train, labels_train = dirichlet(original_dataset=dataset, subset=dataset_train, num_users=num_users,
                                             images_per_client=images_per_client, alpha=100, dataset_name=dataset_name)
        imgs_test, labels_test = dirichlet_test(dataset=dataset_test, num_users=num_users,
                                                images_per_client=images_per_client, alpha=100,
                                                dataset_name=dataset_name)
        iidtype = "iid"
    else:
        imgs_train, labels_train = dirichlet(original_dataset=dataset, subset=dataset_train, num_users=num_users, images_per_client=images_per_client,
                                             alpha=0.1, dataset_name=dataset_name)
        imgs_test, labels_test = dirichlet_test(dataset=dataset_test, num_users=num_users,
                                                images_per_client=images_per_client, alpha=0.1,
                                                dataset_name=dataset_name)
        iidtype = "niid"

    for i in tqdm(range(num_users)):
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}/imgs_train_dev{i}.pkl", 'wb') as f:
            cPickle.dump(imgs_train[i], f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}_lab{labeled}/labels_train_dev{i}.pkl", 'wb') as f:
            cPickle.dump(labels_train[i], f)

        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/imgs_test_dev{i}.pkl", 'wb') as f:
            cPickle.dump(imgs_test[i], f)
        with open(f"dataset/{dataset_name}/{iidtype}/seed{seed}/labels_test_dev{i}.pkl", 'wb') as f:
            cPickle.dump(labels_test[i], f)
    return None


def get_notification_transfer_done(connection, buffer_size, recv_timeout, target, verbose):
    """
    Function for getting notification that the transfer of files between the Server and the Client is done.
    Server receives message from the Client in the following format:

        {data}

        where

        {data} = {filename};{filesize}

    Returns
    -------
    filename : string
        The name of the file received.

    filesize : string
        The size of the file received.
    """
    msg = receive_msg(connection, buffer_size, recv_timeout, verbose)
    assert msg is not None, f"[!] Received no input from {target}"
    filename, filesize = msg.split(';')
    return filename, int(filesize)


def progress_bar(f, size, sent, p):
    progress = sent / size * 100.
    sys.stdout.write(f"({p[0]}:{p[1]}) {f}\'s progress: {progress}\r")


def scp_file(target_ip, target_port, target_usr, target_pwd, target_path, zip_filename, source_path, verbose):
    """
    File for sending a file through SCP.
    """
    if verbose:
        print(f"[+] Server is sending zip file {zip_filename} to the Client.")
    retry = True
    while retry:
        try:
            time.sleep(np.random.randint(2,6))
            policy = paramiko.client.AutoAddPolicy
            with paramiko.SSHClient() as client:
                client.set_missing_host_key_policy(policy)
                client.connect(target_ip, username=target_usr, password=target_pwd, port=22, auth_timeout=200, banner_timeout=200)

                with SCPClient(client.get_transport()) as scp:
                    scp.put(path.join(source_path, zip_filename), remote_path=target_path)
                retry = False

        except BaseException as e:
            print(f"[!] ERROR: {e}")
            print(f"[!] ERROR Connection failed. Could not connect to IP {target_ip} with username "
                  f"{target_usr} and password {target_pwd} for port {target_port}")
            print(f"[!] ERROR: could not put on {source_path} the file {zip_filename} for sending on the "
                  f"remote_path={target_path}")
            retry = True
            print(f"[!] Retrying...")
            time.sleep(5)
            # exit(-1)
    if verbose:
        print("[+] Server sent zip file to the Client.\n")


def unzip_file(connection, zip_filename, target_path, verbose):
    """
    Function for unzipping a file given as parameter. If the zip file cannot be extracted (errors occur), a "Resend"
    message is sent to the Client.
    """
    if verbose:
        print(f"[+] Unzipping file {zip_filename}")
    with zipfile.ZipFile(f"{path.join(target_path, zip_filename)}", 'r') as zip_ref:
        try:
            zip_ref.extractall(path=target_path)
        except BaseException as e:
            if verbose:
                print(f"[!] Encountered error when unzipping: {e}.")
            send_msg(connection, "Resend", verbose)
    if verbose:
        print(f"[+] Extracted file {zip_filename} to {target_path}\n")
        

def zip_file(filename, target_path, verbose):
    """
    Function for zipping a file.
    """
    zip_filename = filename.split(".")[0] + ".zip"
    if verbose:
        print(f"[+] Zipping the file {filename}")
    with zipfile.ZipFile(path.join(target_path, zip_filename), 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(path.join(target_path, filename), basename(path.join(target_path, filename)))
    if verbose:
        print(f"[+] File {zip_filename} zipped in {target_path}")
    filesize = os.path.getsize(path.join(target_path, zip_filename))
    if verbose:
        print(f"[+] File size of {zip_filename} is {filesize / 1000000} MB\n")
    return zip_filename, filesize


def close_connection(connection, verbose):
    """
    Function for closing the connection with the Server.
    """
    if verbose:
        print("[+] Closing the socket.")
    connection.close()
    if verbose:
        print("[+] Socket closed.\n")


def send_msg(connection, msg, verbose):
    """
    Function for sending a string message to the Client.
    """
    if verbose:
        print(f"[+] Server sending message \"{msg}\" to the Client.")
    msg = pickle.dumps(msg)
    connection.sendall(msg)
    if verbose:
        print(f"[+] Message sent.\n")


def receive_msg(connection, buffer_size, recv_timeout, verbose):
    """
    Function for receiving a string message and returning it.

    Returns
    -------
    subject : string
        Returns None if there was an error or if recv_timeout seconds passed with unresponsive Client.
        Returns the received message otherwise.
    """
    received_data, status = recv(connection, buffer_size, recv_timeout, verbose)
    if status == 0:
        connection.close()
        if verbose:
            print(f"[!] Connection closed either due to inactivity for {recv_timeout} seconds or due "
                  f"to an error.")
        return None

    if verbose:
        print(f"[+] Server received message from the Client: {received_data}\n")
    return received_data


def recv(connection, buffer_size, recv_timeout, verbose):
    """
    Function for receiving a string message and returning it.

    Returns
    -------
    received_data : string
        If there is no data received for recv_timeout seconds or if there is an exception returns None.
        If the message is received, it is decoded and returned

    status : int
        Returns 0 if the connection is no longer active and it should be closed.
        Returns 1 if the message was received successfully.
    """
    recv_start_time = time.time()
    received_data = b""
    while True:
        status = 0
        try:
            data = connection.recv(buffer_size)
            received_data += data

            if data == b"":  # Nothing received from the client.
                received_data = b""
                # If still nothing received for a number of seconds specified by the recv_timeout attribute, return
                # with status 0 to close the connection.
                if (time.time() - recv_start_time) > recv_timeout:
                    return None, status
            elif str(data)[-2] == '.':
                if verbose:
                    print(f"[+] All data ({len(received_data)} bytes) received.")

                if len(received_data) > 0:
                    try:
                        # Decoding the data (bytes).
                        received_data = pickle.loads(received_data)
                        # Returning the decoded data.
                        status = 1
                        return received_data, status

                    except BaseException as e:
                        if verbose:
                            print(f"[!] Error decoding the Client's data: {e}.\n")
                        return None, status
            else:
                # In case data is received from the client, update the recv_start_time to the current time to reset
                # the timeout counter.
                recv_start_time = time.time()

        except BaseException as e:
            if verbose:
                print(f"[!] Error receiving data from the Client: {e}.\n")
            return None, 0
