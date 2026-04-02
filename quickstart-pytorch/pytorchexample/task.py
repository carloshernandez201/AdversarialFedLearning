"""pytorchexample: A Flower / PyTorch app."""
# citation flower readme and https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # Block 1 - same
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Block 2 - same
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Block 3 - added layer6
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Block 4 - added layer11
        self.layer9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # Block 5 - added layer15
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer14 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

   
       
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=0.9)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def _clone_state_dict(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _load_state_dict_into(model, state_dict):
    model.load_state_dict({k: v.detach().clone() for k, v in state_dict.items()})


def _poison_batch(images, labels, target_label=0, trigger_size=3, poison_fraction=0.3):
    """
    Add a white square trigger in the bottom-right corner to a subset of images
    and flip their labels to the attacker target label.
    """
    poisoned_images = images.clone()
    poisoned_labels = labels.clone()

    batch_size = images.size(0)
    num_poison = max(1, int(batch_size * poison_fraction))

    poisoned_images[:num_poison, :, -trigger_size:, -trigger_size:] = 1.0
    poisoned_labels[:num_poison] = target_label

    return poisoned_images, poisoned_labels


def _train_backdoor_local(
    model,
    trainloader,
    epochs,
    lr,
    device,
    target_label=0,
    poison_fraction=0.3,
    trigger_size=3,
    clean_weight=0.2,
    backdoor_weight=0.8,
    distance_reg=0.0,
    reference_state=None,
):
    """
    Generic malicious local training loop.

    distance_reg > 0 enables constrain-and-scale style behavior by penalizing
    distance from the received global model.
    """
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            poisoned_images, poisoned_labels = _poison_batch(
                images,
                labels,
                target_label=target_label,
                trigger_size=trigger_size,
                poison_fraction=poison_fraction,
            )

            optimizer.zero_grad()

            clean_logits = model(images)
            poison_logits = model(poisoned_images)

            clean_loss = criterion(clean_logits, labels)
            backdoor_loss = criterion(poison_logits, poisoned_labels)

            loss = clean_weight * clean_loss + backdoor_weight * backdoor_loss

            if distance_reg > 0.0 and reference_state is not None:
                dist = 0.0
                for name, param in model.state_dict().items():
                    ref = reference_state[name].to(device)
                    dist = dist + torch.sum((param - ref) ** 2)
                loss = loss + distance_reg * dist

            loss.backward()
            optimizer.step()


def _apply_scaled_update(model, global_state, scale_factor):
    """
    Replace model weights with:
        w_attack = w_global + scale_factor * (w_local - w_global)
    """
    attacked_state = {}
    local_state = model.state_dict()

    for name in local_state:
        delta = local_state[name] - global_state[name].to(local_state[name].device)
        attacked_state[name] = global_state[name].to(local_state[name].device) + scale_factor * delta

    model.load_state_dict(attacked_state)


def model_replacement_attack(
    global_state,
    local_model,
    lr,
    num_epochs,
    trainloader,
    device,
    target_label=0,
    poison_fraction=0.3,
    trigger_size=3,
    scale_factor=10.0,
):
    """
    Standard backdoor + model replacement:
    1. Train maliciously on poisoned batches
    2. Scale the update relative to the original global weights
    """
    _train_backdoor_local(
        model=local_model,
        trainloader=trainloader,
        epochs=num_epochs,
        lr=lr,
        device=device,
        target_label=target_label,
        poison_fraction=poison_fraction,
        trigger_size=trigger_size,
        clean_weight=0.2,
        backdoor_weight=0.8,
        distance_reg=0.0,
        reference_state=None,
    )

    _apply_scaled_update(local_model, global_state, scale_factor)


def rotating_malicious_attack(
    global_state,
    local_model,
    lr,
    num_epochs,
    trainloader,
    device,
    target_label=0,
    poison_fraction=0.25,
    trigger_size=3,
    scale_factor=8.0,
):
    """
    Client-side logic for the rotating malicious strategy.

    The actual *rotation* is selected by the server each round.
    This function is just the malicious update executed by whichever
    clients are active attackers in the current round.
    """
    _train_backdoor_local(
        model=local_model,
        trainloader=trainloader,
        epochs=num_epochs,
        lr=lr,
        device=device,
        target_label=target_label,
        poison_fraction=poison_fraction,
        trigger_size=trigger_size,
        clean_weight=0.3,
        backdoor_weight=0.7,
        distance_reg=0.0,
        reference_state=None,
    )

    _apply_scaled_update(local_model, global_state, scale_factor)


def constrain_and_scale_attack(
    global_state,
    local_model,
    lr,
    num_epochs,
    trainloader,
    device,
    target_label=0,
    poison_fraction=0.3,
    trigger_size=3,
    scale_factor=5.0,
    distance_reg=1e-4,
):
    """
    Backdoor objective + regularization to remain close to the global model,
    then a moderate scaling step.
    """
    _train_backdoor_local(
        model=local_model,
        trainloader=trainloader,
        epochs=num_epochs,
        lr=lr,
        device=device,
        target_label=target_label,
        poison_fraction=poison_fraction,
        trigger_size=trigger_size,
        clean_weight=0.4,
        backdoor_weight=0.6,
        distance_reg=distance_reg,
        reference_state=global_state,
    )

    _apply_scaled_update(local_model, global_state, scale_factor)