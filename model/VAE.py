import torch
from torch import nn
from torch.nn import functional as F
from typing import *
from einops import rearrange, repeat, reduce
import argparse

Tensor = TypeVar('torch.tensor')


class VAE(nn.Module):

    def __init__(self,
                 args: argparse.ArgumentParser,
                 **kwargs: dict) -> None:
        super().__init__()

        input_dim   = kwargs.get("input_dim", None)
        hidden_dims = kwargs.get("hidden_dims", None)    

        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 128, 128]
        self.latent_dim = hidden_dims[-1]//2

        modules = nn.ModuleList()
        # Build Encoder
        in_dim = input_dim
        for h_dim in hidden_dims:
            if h_dim != hidden_dims[-1]:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim),
                        #nn.BatchNorm2d(h_dim),
                        nn.SiLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Linear(in_dim, h_dim))
                )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # Build Decoder
        modules = nn.ModuleList()
        hidden_dims.reverse()
 
        in_dim = self.latent_dim
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.SiLU())
            )
            in_dim = h_dim

        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], input_dim))
        )

        self.decoder = nn.Sequential(*modules)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input  = rearrange(input, 'b n c -> b (n c)')
        input  = self.encoder(input)
        mu, log_var = input.chunk(2, dim = -1)
        return [mu, log_var]

    def decode(self, z: Tensor, input: Tensor = None) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        if input is not None:
            result = rearrange(result, 'b (n c) -> b n c', b = input.shape[0],  n = input.shape[1], c = input.shape[2])


        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z, input)
        return  [out, input, mu, log_var]
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons, input, mu, log_var = args
        kld_weight = kwargs.get("M_N", 0.1)

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
            
        return {'loss': recons_loss}

    def sample(self,
               B:int,
               N:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(B,
                        self.latent_dim)

        z = z.to(current_device)
        samples = self.decode(z)
        samples = rearrange(samples, 'b (n c) -> b n c', b = B, n = N)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()

    B = 16
    N = 1000
    C = 3
    x = torch.randn(B, N, C) # (B x C x H x W) 
    model_config = {'input_dim': N*C}
    model = VAE(args, **model_config) # C_out
    y = model(x)
    loss = model.loss_function(*y)
    sample = model.sample(B = 16, N= N, current_device=device)
