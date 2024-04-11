from models import ConvAutoencoder
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import plot_training, run_training, run_evaluation, forward_pass
import logging
import os


def weight_averaging(model_class, models):
    model = model_class()
    model_dict = model.state_dict()
    for key in model_dict.keys():
        #TODO this includes batchnorm running averages. Only include weights and biases
        model_dict[key] = sum([m[key] for m in models])/len(models)
    model.load_state_dict(model_dict)
    return model        

def failure_demo():
    results_dir = os.path.join("results", 'failure_demo')
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(message)s'
                    )

    model = ConvAutoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    train_loss, val_loss = run_training(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
    
    test_loss = run_evaluation(model, test_loader, criterion, device)
    logging.info(f'Test loss: {test_loss}')

    logging.info("Snapshot phase")
    snapshot_dicts = [model.state_dict()]

    n_snapshots = 10
    test_losses = [test_loss]
    for i in range(n_snapshots):
        model = ConvAutoencoder()
        model.load_state_dict(snapshot_dicts[-1])
        model.to(device)
        train_loss, val_loss = run_training(model, train_loader, val_loader, 1, criterion, optimizer, device)
        test_loss = run_evaluation(model, test_loader, criterion, device)
        snapshot_dicts.append(model.state_dict())
        test_losses.append(test_loss)
        train_loss += train_loss
        val_loss += val_loss
        
    plot_training(train_loss, val_loss, os.path.join(results_dir, 'training_plot.png'))
    average_test_loss = sum(test_losses)/len(test_losses)
    logging.info(f"Mean test loss: {average_test_loss}")
    max_test_loss = max(test_losses)
    logging.info(f"Max test loss: {max_test_loss}")

    averaged_model = weight_averaging(ConvAutoencoder, snapshot_dicts)
    forward_pass(averaged_model, train_loader, device)
    test_loss = run_evaluation(averaged_model, test_loader, criterion, device)
    logging.info(f'Averaged Model Test loss: {test_loss}')

def decoder_demo():
    results_dir = os.path.join("results", 'decoder_demo')
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(message)s'
                    )

    model = ConvAutoencoder()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    #TODO debug
    train_dataset, _ = torch.utils.data.random_split(train_dataset, [1000, 49000])
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [800, 200])
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    train_loss, val_loss = run_training(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)
    
    test_loss = run_evaluation(model, test_loader, criterion, device)
    logging.info(f'Test loss: {test_loss}')

    logging.info("Snapshot phase")
    snapshot_dicts = [model.state_dict()]

    n_snapshots = 10
    test_losses = [test_loss]
    for i in range(n_snapshots):
        model = ConvAutoencoder()
        model.load_state_dict(snapshot_dicts[-1])
        model.to(device)
        train_loss, val_loss = run_training(model, train_loader, val_loader, 1, criterion, optimizer, device)
        test_loss = run_evaluation(model, test_loader, criterion, device)
        snapshot_dicts.append(model.state_dict())
        test_losses.append(test_loss)
        train_loss += train_loss
        val_loss += val_loss
        
    plot_training(train_loss, val_loss, os.path.join(results_dir, 'training_plot.png'))
    average_test_loss = sum(test_losses)/len(test_losses)
    logging.info(f"Mean test loss: {average_test_loss}")
    max_test_loss = max(test_losses)
    logging.info(f"Max test loss: {max_test_loss}")

    averaged_model = weight_averaging(ConvAutoencoder, snapshot_dicts)
    forward_pass(averaged_model, train_loader, device)
    test_loss = run_evaluation(averaged_model, test_loader, criterion, device)
    logging.info(f'Averaged Model Test loss: {test_loss}')


if __name__ == '__main__':
    failure_demo()