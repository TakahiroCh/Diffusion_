import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from typing import Optional


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class VAE(pl.LightningModule):
    def __init__(self,
                 picture_size: [int, int],
                 channels: int,
                 hidden_size: int,
                 alpha: int,
                 lr: float,
                 save_images: Optional[bool] = None,
                 save_path: Optional[str] = None):
        """Init function for the VAE

        Args:

        hidden_size (int): Latent Hidden Size
        alpha (int): Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        lr (float): Learning Rate, will not be used if auto_lr_find is used
        save_images (Optional[bool]): Boolean to decide whether to save images
        save_path (Optional[str]): Path to save images
        """

        super().__init__()
        self.picture_size = picture_size
        self.channels = channels
        self.input_size = channels * picture_size[0] * picture_size[1]
        self.hidden_size = hidden_size
        if save_images:
            self.save_path = f'{save_path}/VAE/'
        self.save_hyperparameters()
        self.save_images = save_images
        self.lr = lr
        # print("\n" + str(self.input_size))
        self.encoder = nn.Sequential(
            Flatten(),
            nn.Linear(self.input_size, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, self.hidden_size)
        )
        self.hidden2mu = nn.Linear(self.hidden_size, self.hidden_size)
        self.hidden2log_var = nn.Linear(self.hidden_size, self.hidden_size)
        self.alpha = alpha
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1),
            nn.Linear(128, 196), nn.BatchNorm1d(196), nn.LeakyReLU(0.1),
            nn.Linear(196, 392), nn.BatchNorm1d(392), nn.LeakyReLU(0.1),
            nn.Linear(392, self.input_size),
            Stack(self.channels, self.picture_size[0], self.picture_size[1]),
            nn.Tanh()
        )
        
        self.validation_step_outputs = []

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var

    def decode(self, x):
        x = self.decoder(x)
        return x

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick to allow gradients to backpropagate from the
        # stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.randn_like(sigma)
        return mu + sigma*z

    def training_step(self, batch, batch_idx):
        # x, _ = batch
        x = batch
        mu, log_var, x_out = self.forward(x)
        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(),recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss

        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x, _ = batch
        x = batch
        # print(x.shape)
        mu, log_var, x_out = self.forward(x)

        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        # print(kl_loss.item(), recon_loss.item())
        loss = recon_loss*self.alpha + kl_loss
        self.log('val_kl_loss', kl_loss, on_step=False, on_epoch=True)
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        # print(x.mean(),x_out.mean())
        self.validation_step_outputs.append(x_out)
        return x_out, loss

    # def validation_epoch_end(self, outputs):
    #     if not self.save_images:
    #         return
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
    #     choice = random.choice(outputs)
    #     output_sample = choice[0]
    #     output_sample = output_sample.reshape(-1, 1, 28, 28)
    #     # output_sample = self.scale_image(output_sample)
    #     save_image(
    #         output_sample,
    #         f"{self.save_path}/epoch_{self.current_epoch+1}.png",
    #         # value_range=(-1, 1)
    #     )

    def on_validation_epoch_end(self):
      if not self.save_images:
          return
      if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
      output_sample = random.choice(self.validation_step_outputs)[0]
      output_sample = output_sample.reshape(-1, self.channels, self.picture_size[0], self.picture_size[1])
      # output_sample = self.scale_image(output_sample)
      save_image(
          output_sample,
          f"{self.save_path}/epoch_{self.current_epoch+1}.png",
          # value_range=(-1, 1)
      )
      self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = ReduceLROnPlateau(optimizer,)
        return {
            "optimizer": optimizer, "lr_scheduler": lr_scheduler,
            "monitor": "val_loss"
        }

    def forward(self, x):
        mu, log_var = self.encode(x)
        hidden = self.reparametrize(mu, log_var)
        output = self.decoder(hidden)
        return mu, log_var, output