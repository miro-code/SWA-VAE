#vae training params and architecture from https://github.com/probml/pyprobml/blob/a662f44e891fa6f30ed1184558fd84efc42c8a56/deprecated/vae/standalone/vae_conv_mnist.py#L128
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from models import ConvVAE
import logging
from datasets import get_mnist_debug_loaders as get_mnist_loaders #TODO CHANGE
from utils import weight_averaging, forward_pass
import os

if __name__ == "__main__":
    parser = ArgumentParser(description="Hyperparameters for our experiments")
    parser.add_argument("--bs", type=int, default=512, help="batch size")
    parser.add_argument("--epochs", type=int, default=50, help="num epochs")
    parser.add_argument("--latent-dim", type=int, default=8, help="size of latent dim for our vae")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--kl-coeff", type=int, default=5, help="kl coeff aka beta term in the elbo loss function")
    parser.add_argument("--n_snapshots", type=int, default=5, help="How many snapshots to take for our weight averaging")
    hparams = parser.parse_args()

    results_dir = os.path.join("results", 'failure_demo')
    os.makedirs(results_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, 
                        handlers=[
                            logging.FileHandler(os.path.join(results_dir, 'log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(message)s'
                    )
    
    model_params = {
        "input_shape" : (1, 28, 28),
        "encoder_conv_filters" : [28, 64, 64],
        "decoder_conv_t_filters" : [64, 28, 1],
        "latent_dim" : hparams.latent_dim,
        "kl_coeff" : hparams.kl_coeff,
        "lr" : hparams.lr,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = ConvVAE(**model_params)

    train_loader, val_loader, test_loader = get_mnist_loaders(hparams.bs)
    trainer = Trainer(max_epochs=hparams.epochs)
    trainer.fit(model, train_loader)
    test_loss = trainer.validate(model, val_loader)[0]["val_recon_loss"]
    logging.info(f'Test loss: {test_loss}')
    logging.info("Snapshot phase")
    snapshot_dicts = [model.cpu().state_dict()]
    test_losses = [test_loss]
    for i in range(hparams.n_snapshots):
        model = ConvVAE(**model_params)
        model.load_state_dict(snapshot_dicts[-1])
        trainer = Trainer(max_epochs=1)
        trainer.fit(model, train_loader)
        test_loss = trainer.validate(model, val_loader)[0]["val_recon_loss"]
        snapshot_dicts.append(model.cpu().state_dict())
        test_losses.append(test_loss)

    average_test_loss = sum(test_losses)/len(test_losses)
    logging.info(f"Mean test loss: {average_test_loss}")
    min_test_loss = min(test_losses)
    logging.info(f"Min test loss: {min_test_loss}")
    
    averaged_model = weight_averaging(ConvVAE, snapshot_list=snapshot_dicts, model_arguments=model_params)
    forward_pass(averaged_model, train_loader, device)

    test_loss = trainer.validate(averaged_model, val_loader)[0]["val_recon_loss"]
    logging.info(f'Averaged Model Test loss: {test_loss}')