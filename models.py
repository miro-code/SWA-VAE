import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from pytorch_lightning import LightningModule

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, out_channels=32, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.Conv2d(64, out_channels=128, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, out_channels=256, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, out_channels=512, kernel_size= 3, stride= 2, padding  = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride = 2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, out_channels= 3, kernel_size= 3, padding= 1), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvVAEModule(nn.Module):
    def __init__(self, input_shape, encoder_conv_filters, decoder_conv_t_filters, latent_dim, deterministic=False):
        super(ConvVAEModule, self).__init__()
        self.input_shape = input_shape

        self.latent_dim = latent_dim
        self.deterministic = deterministic

        all_channels = [self.input_shape[0]] + encoder_conv_filters

        self.enc_convs = nn.ModuleList([])

        # encoder_conv_layers
        for i in range(len(encoder_conv_filters)):
            self.enc_convs.append(nn.Conv2d(all_channels[i], all_channels[i + 1], kernel_size=3, stride=2, padding=1))
            if not self.latent_dim == 2:
                self.enc_convs.append(nn.BatchNorm2d(all_channels[i + 1]))
            self.enc_convs.append(nn.LeakyReLU())

        self.flatten_out_size = self.flatten_enc_out_shape(input_shape)

        if self.latent_dim == 2:
            self.mu_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.mu_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim), nn.LeakyReLU(), nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.log_var_linear = nn.Linear(self.flatten_out_size, self.latent_dim)
        else:
            self.log_var_linear = nn.Sequential(
                nn.Linear(self.flatten_out_size, self.latent_dim), nn.LeakyReLU(), nn.Dropout(0.2)
            )

        if self.latent_dim == 2:
            self.decoder_linear = nn.Linear(self.latent_dim, self.flatten_out_size)
        else:
            self.decoder_linear = nn.Sequential(
                nn.Linear(self.latent_dim, self.flatten_out_size), nn.LeakyReLU(), nn.Dropout(0.2)
            )

        all_t_channels = [encoder_conv_filters[-1]] + decoder_conv_t_filters

        self.dec_t_convs = nn.ModuleList([])

        num = len(decoder_conv_t_filters)

        # decoder_trans_conv_layers
        for i in range(num - 1):
            self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
            self.dec_t_convs.append(
                nn.ConvTranspose2d(all_t_channels[i], all_t_channels[i + 1], 3, stride=1, padding=1)
            )
            if not self.latent_dim == 2:
                self.dec_t_convs.append(nn.BatchNorm2d(all_t_channels[i + 1]))
            self.dec_t_convs.append(nn.LeakyReLU())

        self.dec_t_convs.append(nn.UpsamplingNearest2d(scale_factor=2))
        self.dec_t_convs.append(
            nn.ConvTranspose2d(all_t_channels[num - 1], all_t_channels[num], 3, stride=1, padding=1)
        )
        self.dec_t_convs.append(nn.Sigmoid())

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def _run_step(self, x):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return z, recon, p, q

    def flatten_enc_out_shape(self, input_shape):
        x = torch.zeros(1, *input_shape)
        for l in self.enc_convs:
            x = l(x)
        self.shape_before_flattening = x.shape
        return int(np.prod(self.shape_before_flattening))

    def encode(self, x):
        for l in self.enc_convs:
            x = l(x)
        x = x.view(x.size()[0], -1)  # flatten
        mu = self.mu_linear(x)
        log_var = self.log_var_linear(x)
        return mu, log_var

    def decode(self, z):
        z = self.decoder_linear(z)
        recon = z.view(z.size()[0], *self.shape_before_flattening[1:])
        for l in self.dec_t_convs:
            recon = l(recon)
        return recon

    def forward(self, x):
        mu, log_var = self.encode(x)
        if self.deterministic:
            return self.decode(mu), mu, None
        else:
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var


class ConvVAE(LightningModule):
    def __init__(self, input_shape, encoder_conv_filters, decoder_conv_t_filters, latent_dim, kl_coeff=0.1, lr=0.001):
        super(ConvVAE, self).__init__()
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.vae = ConvVAEModule(input_shape, encoder_conv_filters, decoder_conv_t_filters, latent_dim)

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self.vae._run_step(x)

        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.sum()  # I tried sum, here
        kl *= self.kl_coeff

        loss = kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
