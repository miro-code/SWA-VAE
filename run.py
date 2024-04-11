from models import ConvAutoencoder
import torch
from utils import plot_training, run_training, run_evaluation, forward_pass, plot_graph
import logging
import os
from datasets import get_cifar10_loaders, get_cifar10_debug_loaders
import numpy as np

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    num_epochs = 100
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
    train_loader, val_loader, test_loader = get_cifar10_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        model.encoder.requires_grad_(False)
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

def encoder_demo():
    results_dir = os.path.join("results", 'encoder_demo')
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
    train_loader, val_loader, test_loader = get_cifar10_debug_loaders()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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
        model.decoder.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
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

    #visualise loss for average latent representation of one model pair
    
    model1 = ConvAutoencoder()
    model1.load_state_dict(snapshot_dicts[0])
    model1.to(device)
    model2 = ConvAutoencoder()
    model2.load_state_dict(snapshot_dicts[-1])
    model2.to(device)
    model1.eval()
    model2.eval()
    n_steps = 10
    #create n_steps empty lists to store losses
    losses = [[] for _ in range(n_steps)]
    weight_pairs = [(1-i/n_steps-1, i/n_steps-1) for i in range(n_steps)]
    for img, _ in test_loader:
        img = img.to(device)
        latent1 = model1.encoder(img)
        latent2 = model2.encoder(img)
        for i, (w1, w2) in enumerate(weight_pairs):
            latent = w1*latent1 + w2*latent2
            output = model1.decoder(latent)
            loss = criterion(output, img)
            losses[i].append(loss.item())
    np.mean(losses, axis=1)
    plot_graph(np.arange(0, 1, 1/n_steps), np.mean(losses, axis=1), 'Weight for model 1', 'Loss', 'Loss for average latent representation', os.path.join(results_dir, 'averaged_latent_loss.png'))


if __name__ == '__main__':
    failure_demo()
    decoder_demo()
    encoder_demo()
    
    