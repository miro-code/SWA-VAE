from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torchvision.datasets import MNIST

def get_cifar10_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_cifar10_debug_loaders(batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(train_dataset, [400, 100, 49500])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def get_mnist_loaders(batch_size=256):
    mnist_train = MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
    )
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_train, [55000, 5000])
    test_dataset = MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_mnist_debug_loaders(batch_size=256):
    mnist_train = MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
    )
    train_dataset, val_dataset, _ = torch.utils.data.random_split(mnist_train, [400, 100, 59500])
    test_dataset = MNIST(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32))]),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
