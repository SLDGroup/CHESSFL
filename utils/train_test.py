import time
import torch
from models.get_model import get_model
from utils.dataset_utils import load_data, get_fixmix
import _pickle as cpickle
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from utils.general_utils import AverageMeter
from torch.utils.data import Dataset
import torch.nn.functional as F
import copy
from collections import Counter
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from os import path
from scipy.stats import mode

def train_fixmatch(model, dev_idx, model_path, cuda_name, optimizer, local_epochs, data_iid, dataset_name, seed, verbose=False, labeled=4000):
    device = torch.device(cuda_name)
    data_loader = load_data(data_iid=data_iid, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed, is_unlabeled=True, labeled=labeled)

    for epoch in range(local_epochs):
        losses = AverageMeter()
        accuracies = AverageMeter()
        for img_weak, img_strong, label in data_loader:
            img_weak, img_strong, label = img_weak.to(device), img_strong.to(device), label.to(device)
            optimizer.zero_grad()
            # 1. Get the pseudo-labels using the weakly augmented data
            with torch.no_grad():
                outputs_weak = model(img_weak)
                pseudo_labels = torch.argmax(outputs_weak, dim=1)
            # high_confidence_mask = torch.max(F.softmax(outputs_weak, dim=1), dim=1)[0] > 0.95
            probs = F.softmax(outputs_weak, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            threshold = 0.95  # This can be adjusted
            mask = max_probs.ge(threshold).float()

            outputs_strong = model(img_strong)

            if mask.sum() > 0:
                loss = F.cross_entropy(outputs_strong, pseudo_labels, reduction='none')
                loss = (loss * mask).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.update(loss.item(), mask.sum())
                pred = outputs_strong.argmax(dim=1, keepdim=True)
                correct = pred.eq(pseudo_labels.view_as(pred)).sum().item()
                accuracies.update(correct / img_strong.size(0), img_strong.size(0))

        if verbose:
            print(f"[+] Device {dev_idx} Epoch {epoch+1}: Loss - {losses.avg}, Accuracy - {np.round(100. * accuracies.avg,2)}%")

    torch.save(model.state_dict(), model_path)

    return losses.avg, (100. * accuracies.avg)


def train_semifl(model, dev_idx, model_path, cuda_name, optimizer, local_epochs, data_iid, dataset_name, seed,
                 verbose=False, labeled=4000):
    device = torch.device(cuda_name)
    data_loader = load_data(data_iid=data_iid, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed, is_unlabeled=True, labeled=labeled)

    # print("Loaded data")
    fix_dataloader, mix_dataloader = get_fixmix(data_loader=data_loader, model=model, cuda_name=cuda_name, seed=seed)

    # print("Mixed data")
    if fix_dataloader is None:
        # print("Fix data loader is None")
        return 0.0, 0.0

    lmbda = 1.0
    beta = torch.distributions.beta.Beta(torch.tensor([0.75]), torch.tensor([0.75]))
    for epoch in range(local_epochs):
        losses = AverageMeter()
        accuracies = AverageMeter()

        # For simplicity, let's assume your DataLoader can return two batches of data at a time for Dfix and Dmix
        for (img_weak_fix, img_strong_fix, label_fix), (img_weak_mix, img_strong_mix, label_mix) in zip(fix_dataloader, mix_dataloader):
            img_weak_fix, img_strong_fix, label_fix = img_weak_fix.to(device), img_strong_fix.to(device), label_fix.to(device)
            img_weak_mix, img_strong_mix, label_mix = img_weak_mix.to(device), img_strong_mix.to(device), label_mix.to(device)

            outputs_strong_fix = model(img_strong_fix)
            loss_fix = F.cross_entropy(outputs_strong_fix, label_fix)

            lam_mix = beta.sample()[0]
            x_mix = lam_mix * img_weak_fix + (1 - lam_mix) * img_weak_mix
            outputs_mix = model(x_mix)


            loss_mix = (lam_mix * F.cross_entropy(outputs_mix, label_fix) +
                        (1 - lam_mix) * F.cross_entropy(outputs_mix, label_mix))
            loss = loss_fix + lmbda * loss_mix
            optimizer.zero_grad()


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.update(loss.item(), img_weak_fix.size(0))
            pred = outputs_strong_fix.argmax(dim=1, keepdim=True)
            correct = pred.eq(label_fix.view_as(pred)).sum().item()
            accuracies.update(correct / img_weak_fix.size(0), img_weak_fix.size(0))


        if verbose:
            print(f"[+] Device {dev_idx} Epoch {epoch + 1}: Loss - {losses.avg}, Accuracy - {np.round(100. * accuracies.avg, 2)}%")

    torch.save(model.state_dict(), model_path)
    # print("End training....")
    return losses.avg, (100. * accuracies.avg)


def spectral_loss(z1, z2, mu=1.0):
    mask1 = (torch.norm(z1, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    mask2 = (torch.norm(z2, p=2, dim=1) < np.sqrt(mu)).float().unsqueeze(1)
    z1 = mask1 * z1 + (1-mask1) * F.normalize(z1, dim=1) * np.sqrt(mu)
    z2 = mask2 * z2 + (1-mask2) * F.normalize(z2, dim=1) * np.sqrt(mu)
    loss_part1 = -2 * torch.mean(z1 * z2) * z1.shape[1]
    square_term = torch.matmul(z1, z2.T) ** 2
    loss_part2 = torch.mean(torch.triu(square_term, diagonal=1) + torch.tril(square_term, diagonal=-1)) * \
                 z1.shape[0] / (z1.shape[0] - 1)
    return (loss_part1 + loss_part2) / mu


def centroid_kl_loss(local_features, pseudo_labels, edge_centroids, device, temperature=4):
    # Convert pseudo_labels tensor to CPU and to a list
    pseudo_labels_list = pseudo_labels.cpu().tolist()

    # Get the associated centroids for each pseudo label and apply softmax with temperature
    selected_centroids = torch.stack([edge_centroids[label] for label in pseudo_labels_list]).to(device)
    prob_selected_centroids = F.softmax(selected_centroids / temperature, dim=1)

    # Apply log_softmax with temperature to local features
    log_prob_local_features = F.log_softmax(local_features / temperature, dim=1)

    # Compute KL divergence
    loss = F.kl_div(log_prob_local_features, prob_selected_centroids, reduction='batchmean')

    return loss


def distance_pseudo_labeling(edge_centroids, local_features, confidence_threshold=0.70):
    # Ensure that the local_features is on the same device as the centroids
    device = next(iter(edge_centroids.values())).device
    local_features = local_features.to(device)

    # Sort the dictionary by its keys (labels)
    sorted_keys = sorted(edge_centroids.keys())
    sorted_centroids = [edge_centroids[key] for key in sorted_keys]

    # Stack sorted centroids into a tensor
    centroids_tensor = torch.stack(sorted_centroids)

    # Compute cosine similarity using PyTorch functions
    similarities = torch.mm(local_features, centroids_tensor.t())
    closest_centroids_cos_idx = torch.argmax(similarities, dim=1)
    max_similarities = torch.max(similarities, dim=1).values

    # Compute Euclidean distances using PyTorch
    distances = torch.norm(local_features.unsqueeze(1) - centroids_tensor, dim=2)
    closest_centroids_euc_idx = torch.argmin(distances, dim=1)

    # Indices where cosine and euclidean agree
    agreement_idx = (closest_centroids_cos_idx == closest_centroids_euc_idx).nonzero(as_tuple=True)[0]

    # Create a mask with all values set to False
    mask = torch.zeros_like(closest_centroids_cos_idx, dtype=torch.bool)

    # For the indices where cosine and euclidean agree, check if their similarity exceeds the threshold
    for idx in agreement_idx:
        if max_similarities[idx] > confidence_threshold:
            mask[idx] = True

    return closest_centroids_cos_idx, mask


def train_chessfl(model, dev_idx, model_path, cuda_name, optimizer, local_epochs, data_iid, dataset_name, seed, dev_path,
                  verbose=False, labeled=4000, rotate=True, edge_server_idx=0):
    device = torch.device(cuda_name)
    data_loader = load_data(data_iid=data_iid, dev_idx=dev_idx, dataset_name=dataset_name, seed=seed, is_unlabeled=True, labeled=labeled, rotate=rotate)
    with open(path.join(dev_path, f"edge_centroids_{edge_server_idx}.pkl"), "rb") as f:
        edge_centroids = cpickle.load(f)

    for epoch in range(local_epochs):
        losses = AverageMeter()
        accuracies = AverageMeter()

        for batch_idx, (img_w, img_s, target, img_rot, tf_type) in enumerate(data_loader):
            img_w, img_s, target = img_w.to(device), img_s.to(device), target.to(device)
            img_rot, tf_type = img_rot.to(device), tf_type.to(device)

            output_w, output_s, rot_output, x_out_w, x_out_s, x_out_rot = model(img_w, img_rot, img_s)

            normalized_features = F.normalize(x_out_w, p=2, dim=1)
            pseudo_labels, mask = distance_pseudo_labeling(edge_centroids, normalized_features)

            if mask.sum() > 0:
                hem_loss = centroid_kl_loss(x_out_s, pseudo_labels, edge_centroids, temperature=4, device=device)

                loss_pseudo = F.cross_entropy(output_s, pseudo_labels, reduction='none')
                loss_rot = F.cross_entropy(rot_output, tf_type, reduction='none')
                loss = ((loss_pseudo + loss_rot) * mask).mean() + hem_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses.update(loss.item(), img_w.size(0))
                pred = output_s.argmax(dim=1, keepdim=True)
                correct = pred.eq(pseudo_labels.view_as(pred)).sum().item()
                accuracies.update(correct / img_s.size(0), img_s.size(0))

        if verbose:
            print(f"[+] Device {dev_idx} Epoch {epoch + 1}: Loss - {losses.avg}, Accuracy - {np.round(100. * accuracies.avg, 2)}%")
        if (int(dev_idx)+1) % 10 == 0:
            print(f"[+] Device {dev_idx} Epoch {epoch + 1}: Loss - {losses.avg}, Accuracy - {np.round(100. * accuracies.avg, 2)}%")
    torch.save(model.state_dict(), model_path)
    return losses.avg, (100. * accuracies.avg)


def edge_clustering(model, dataset_name, cuda_name, edge_server_idx, dev_path, seed, labeled, rot_pred):
    if dataset_name == "cifar10" or dataset_name == "svhn":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes=100

    features_list = []
    labels_list = []
    data_loader = load_data(dev_idx=edge_server_idx, dataset_name=dataset_name, seed=seed, is_edge_server=True,
                            labeled=labeled, rotate=rot_pred)
    device = torch.device(cuda_name)
    # Collect features and true labels from the dataloader
    with torch.no_grad():
        for data, img_s, labels, img_rot, _ in data_loader:
            _, _, _, features, _, _ = model(data.to(device), img_rot.to(device), img_s.to(device))
            features_list.append(features)
            labels_list.append(labels)

    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    normalized_features = F.normalize(all_features, p=2, dim=1)

    all_labels_np = all_labels.cpu().numpy()  # Convert tensor to numpy array
    unique_labels = np.unique(all_labels_np)  # Find unique labels
    K = len(unique_labels)  # Count the number of unique labels

    # Grouping by labels and calculating initial centroids from labeled data
    edge_centroids = []
    for label in range(K):
        label_indices = (all_labels == label).nonzero(as_tuple=True)[0]
        label_features = torch.index_select(normalized_features, 0, label_indices.to(device))
        centroid = label_features.mean(dim=0)
        edge_centroids.append(centroid)

        # Mapping labels to centroids
    label_to_centroid = {label: edge_centroids[label] for label in range(K)}

    # Save to .pkl file
    with open(path.join(dev_path, f"edge_centroids_{edge_server_idx}.pkl"), "wb") as f:
        cpickle.dump(label_to_centroid, f)


def edge_train_chessfl(model, edge_server_idx, cuda_name, dataset_name, seed, edge_server_epochs, learning_rate,
                       comm_round, edge_opt_path, save_opt, labeled=4000, rot_pred=False):
    device = torch.device(cuda_name)
    data_loader = load_data(dev_idx=edge_server_idx, dataset_name=dataset_name, seed=seed, is_edge_server=True, labeled=labeled, rotate=rot_pred)
    model = model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if save_opt:
        if comm_round == 0:
            torch.save(optimizer.state_dict(), edge_opt_path)
        else:
            optimizer.load_state_dict(torch.load(edge_opt_path))

    losses = AverageMeter()
    accuracies = AverageMeter()
    for local_epoch in range(edge_server_epochs):
        for img_w, img_s, label, img_rot, tf_type in data_loader:
            img_w, img_s, label = img_w.to(device), img_s.to(device), label.to(device)
            img_rot, tf_type = img_rot.to(device), tf_type.to(device)
            optimizer.zero_grad()
            output_w, output_s, rot_output, x_out_w, x_out_s, x_out_rot = model(img_w, img_rot, img_s)

            loss_weak = criterion(output_w, label)
            spec_loss = spectral_loss(x_out_w, x_out_s)
            loss_rot = criterion(rot_output, tf_type)
            loss = loss_weak + loss_rot + spec_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.update(loss.item(), img_w.size(0))
            correct = torch.argmax(output_w, dim=1).eq(label).sum().item()
            accuracies.update(correct / img_w.size(0), img_w.size(0))

    return model, losses.avg, (100. * accuracies.avg)


def edge_train(model, edge_server_idx, cuda_name, dataset_name, seed, edge_server_epochs, learning_rate, comm_round,
               edge_opt_path, save_opt, labeled=4000):
    device = torch.device(cuda_name)
    data_loader = load_data(dev_idx=edge_server_idx, dataset_name=dataset_name, seed=seed, is_edge_server=True, labeled=labeled)
    model = model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if save_opt:
        if comm_round == 0:
            torch.save(optimizer.state_dict(), edge_opt_path)
        else:
            optimizer.load_state_dict(torch.load(edge_opt_path))

    losses = AverageMeter()
    accuracies = AverageMeter()
    for local_epoch in range(edge_server_epochs):
        for img, label in data_loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.update(loss.item(), img.size(0))
            correct = torch.argmax(outputs, dim=1).eq(label).sum().item()
            accuracies.update(correct / img.size(0), img.size(0))
    return model, losses.avg, (100. * accuracies.avg)


def test(model, loss_func, cuda_name, dataset_name, seed, verbose=False, labeled=4000, rot_pred=False):
    device = torch.device(cuda_name)
    model = model.to(device)
    if rot_pred:
        data_loader = load_data(dataset_name=dataset_name, seed=seed, test_global=True, labeled=labeled, rotate=rot_pred)
    else:
        data_loader = load_data(dataset_name=dataset_name, seed=seed, test_global=True, labeled=labeled)
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        if not rot_pred:
            for img, label in data_loader:
                img, label = img.to(device), label.to(device)
                outputs = model(img)
                loss = loss_func(outputs, label)
                losses.update(loss.item(), img.size(0))
                correct = torch.argmax(outputs, dim=1).eq(label).sum().item()
                accuracies.update(correct / img.size(0), img.size(0))
        else:
            for img, label in data_loader:
                img, label = img.to(device), label.to(device)
                outputs, _, _, _, _, _ = model(img, img, img)
                loss = loss_func(outputs, label)
                losses.update(loss.item(), img.size(0))
                correct = torch.argmax(outputs, dim=1).eq(label).sum().item()
                accuracies.update(correct / img.size(0), img.size(0))

    if verbose:
        print(f'[+] Test Loss: {losses.avg}, Test Accuracy: {100. * accuracies.avg}%')
        
    return losses.avg, 100. * accuracies.avg


def local_training(model_name, train_type, model_path, opt_path, cuda_name, learning_rate, local_epochs, dev_idx, seed, dev_path,
                   verbose=False, data_iid=True, comm_round=-1, save_opt=False, use_sbn=False, labeled=4000, edge_server_idx=0):
    device = torch.device(cuda_name)

    if train_type == "semifl" or train_type == "fixmatch":
        model = get_model(model_name=f"{model_name}", use_sbn=use_sbn).to(device)
    else:
        model = get_model(model_name=f"{model_name}", use_sbn=use_sbn, rot_pred=True).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if save_opt:
        if comm_round == 0:
            torch.save(optimizer.state_dict(), opt_path)
        else:
            optimizer.load_state_dict(torch.load(opt_path))

    if verbose:
        print(f"[+] Device {dev_idx} training...")

    start_time = time.time()
    if train_type == "fixmatch":
        train_loss, train_acc = train_fixmatch(
            model=model, model_path=model_path, cuda_name=cuda_name, optimizer=optimizer, local_epochs=local_epochs,
            verbose=verbose, data_iid=data_iid, dev_idx=dev_idx, dataset_name=model_name.split('_')[0], seed=seed, labeled=labeled)
    elif train_type == "semifl":
        train_loss, train_acc = train_semifl(
            model=model, model_path=model_path, cuda_name=cuda_name, optimizer=optimizer, local_epochs=local_epochs,
            verbose=verbose, data_iid=data_iid, dev_idx=dev_idx, dataset_name=model_name.split('_')[0], seed=seed, labeled=labeled)
    elif train_type == "chessfl":
        train_loss, train_acc = train_chessfl(
            model=model, model_path=model_path, cuda_name=cuda_name, optimizer=optimizer, local_epochs=local_epochs,
            verbose=verbose, data_iid=data_iid, dev_idx=dev_idx, dataset_name=model_name.split('_')[0], seed=seed, labeled=labeled, dev_path=dev_path, edge_server_idx=edge_server_idx)
    train_time = time.time() - start_time

    if verbose:
        print(f"[+] Train loss: {train_loss}, Train accuracy: {train_acc}%")

    return train_loss, train_acc, train_time
