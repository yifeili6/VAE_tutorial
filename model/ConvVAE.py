import torch
from torch import nn
from torch.nn import functional as F
from typing import *
from einops import rearrange, repeat, reduce
import argparse

Tensor = TypeVar('torch.tensor')

class ConvVAE(nn.Module):

    def __init__(self,
                 args: argparse.ArgumentParser,
                 **kwargs: dict) -> None:
        super().__init__()

        input_dim   = kwargs.get("input_dim", None)
        in_channels = kwargs.get("in_channels", None)
        latent_dim  = kwargs.get("latent_dim", None)
        hidden_dims = kwargs.get("hidden_dims", None)

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.prebegin_adaptive = torch.nn.AdaptiveAvgPool2d(64) ##REDUCE to 64 on PURPOSE! so that output before ADAPTIVE pooling becomes 64 too!

        modules = nn.ModuleList()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim) # 4 is the h*w of the last conv layer
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = nn.ModuleList()

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4) 

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

        self.final_adaptive = torch.nn.AdaptiveAvgPool2d(input_dim)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # input [B x C_in x H_in x W_in]: [16, 3, 64, 64]
        input = self.prebegin_adaptive(input) # input: [B x C_in x H x W]: [16, 3, 64, 64]
        final_conv = self.encoder(input) # final_conv: [B x C_out x H_out x W_out]: [16, 512, 2, 2]
        result = rearrange(final_conv, 'b c h w -> b (c h w)') # result: [B x (C_out * H_out * W_out)]: [16, 512 * 2 * 2]

        # mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        b, c, h, w = final_conv.shape
        return [mu, log_var, (b, c, h, w)]

    def decode(self, z: Tensor, final_conv_shape: Tuple) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        b, c, h, w = final_conv_shape

        result = self.decoder_input(z)
        result = rearrange(result, 'b (c h w) -> b c h w', c = c, h = h, w = w)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.final_adaptive(result)
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
        mu, log_var, final_conv_shape = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z, final_conv_shape)
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

        recons, input, mu, log_var  = args
        kld_weight = kwargs.get("M_N", 0.1)
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()

    H_in, W_in = 128, 128
    C_in = 1
    B = 16
    x = torch.randn(B, C_in, H_in, W_in) # (B x C x H x W) 

    model_config = {'input_dim': H_in, 
                    'in_channels': C_in, 
                    'latent_dim': 10}
    
    model = ConvVAE(args, **model_config) # C_out
    y = model(x)
    loss_config = {'M_N': 0.005}
    loss = model.loss_function(*y, **loss_config)