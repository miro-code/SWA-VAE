
from matplotlib import pyplot as plt
import numpy as np
from pytorch_lightning import LightningModule


def plot_training(train_loss, val_loss, filename):
    plt.plot(train_loss, label='train loss', color='b')
    plt.plot(val_loss, label='val loss', color='r')
    plt.xticks(np.arange(1, len(train_loss)) + 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_graph(x, y, x_label, y_label, title, filename):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def forward_pass(model, data_loader, device):
    model.train()
    model.to(device)
    for img, y in data_loader:
        img = img.to(device)
        #inherets from lightning module
        if(isinstance(model, LightningModule)):
            model.step((img, y), 0)
        else:
            model(img)
    

def run_training(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
        train_loss = []
        val_loss = []
        model.train()
        for epoch in range(num_epochs):
            epoch_train_loss = 0
            for img, _ in train_loader:
                img = img.to(device)
                optimizer.zero_grad()
                output = model(img)
                loss = criterion(output, img)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            mean_epoch_train_loss = epoch_train_loss/len(train_loader)
            train_loss.append(mean_epoch_train_loss)
        
            epoch_val_loss = 0
            for img, _ in val_loader:
                img = img.to(device)
                output = model(img)
                loss = criterion(output, img)
                epoch_val_loss += loss.item()
            mean_epoch_val_loss = epoch_val_loss/len(val_loader)
            val_loss.append(mean_epoch_val_loss)
    
        return train_loss, val_loss

def run_evaluation(model, test_loader, criterion, device):
    test_loss = 0
    model.eval()
    for img, _ in test_loader:
        img = img.to(device)
        output = model(img)
        loss = criterion(output, img)
        test_loss += loss.item()
    mean_test_loss = test_loss/len(test_loader)
    return mean_test_loss


def weight_averaging(model_class, snapshot_list, model_arguments = {}):
    model = model_class(**model_arguments)
    model_dict = model.state_dict()
    for key in model_dict.keys():
        #TODO this includes batchnorm running averages. Only include weights and biases
        model_dict[key] = sum([m[key] for m in snapshot_list])/len(snapshot_list)
    model.load_state_dict(model_dict)
    return model  
