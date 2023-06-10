import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from typing import *
from einops import rearrange, repeat, reduce
import argparse
Tensor = TypeVar('torch.tensor')


class PCA_torch(torch.nn.Module):
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 **kwargs:dict):
        super().__init__()
        
        full_matrices = kwargs.get("full_matrices", False)
        n_components_ = kwargs.get("n_components", 16)
        return_logp = kwargs.get("return_logp", False)

        n_components = n_components_ 
        setattr(self, "full_matrices", full_matrices)
        setattr(self, "n_components_", n_components)
        setattr(self, "return_logp", return_logp)

        self.register_buffer("U", torch.tensor(1.))
        self.register_buffer("S", torch.tensor(1.))
        self.register_buffer("Vt", torch.tensor(1.))
        self.register_buffer("explained_variance_ratio_", torch.tensor(1.))
        self.register_buffer("singular_values_", torch.tensor(1.))
        self.register_buffer("components_", torch.tensor(1.))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("mean_", torch.tensor(1.))
        self.register_buffer("pca_mean", torch.tensor(1.))
        self.register_buffer("pca_std", torch.tensor(1.))

    @staticmethod
    def svd_flip(u, v):
        # flip eigenvectors' sign to enforce deterministic output
        #https://github.com/scikit-learn/scikit-learn/blob/2e481f114169396660f0051eee1bcf6bcddfd556/sklearn/utils/extmath.py#L760:~:text=%23%20columns%20of%20u,.newaxis%5D
        # columns of u, rows of v
        max_abs_cols = torch.argmax(u.abs(), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v

    def fit(self, X: torch.Tensor):
        assert X.is_leaf and X.ndim == 3, "Specialized for VAE relevant PCA"
        X = rearrange(X, 'b n c -> b (n c)')
        self.n_components_ = min(self.n_components_, X.size(0), X.size(1)) #(custom, m, n)
        self.mean_ = torch.mean(X, dim=0, keepdim=True) 
        X_center = X - self.mean_
        
        U, S, Vt = torch.linalg.svd(X_center, full_matrices=False) #https://pytorch.org/docs/stable/generated/torch.linalg.svd.html#torch.linalg.svd:~:text=%5Cmathbb%7BC%7D-,C,%2C%20the%20full%20SVD%20of%20a%20matrix,-A%20%5Cin%20%5Cmathbb
        U, Vt = self.svd_flip(U, Vt)

        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("Vt", Vt)

        explained_variance_ = (S**2) / (X.size(0) - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.clone()  # Store the singular values.
        if 0 < self.n_components_ < 1:
            self.n_components_: int = torch.searchsorted(explained_variance_ratio_, self.n_components_, side="right") + 1 #ratio to integer

        self.register_buffer("explained_variance_ratio_", explained_variance_ratio_)
        self.register_buffer("singular_values_", singular_values_)
        self.register_buffer("components_", Vt.clone())
        
        self.register_buffer("fitted", torch.tensor(True))
        self.register_buffer("mean_", self.mean_)
        
        X_transformed, pca_mean, pca_std = self.transform(X, for_fit=True)
        self.register_buffer("pca_mean", pca_mean)
        self.register_buffer("pca_std", pca_std)

        return self



    # Below:
    # https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef5ee2a8aea80498388690e2213118efd/sklearn/decomposition/_base.py#L19:~:text=def%20transform(,return%20X_transformed
    def transform(self, X: torch.Tensor, for_fit: bool=False):
        if not self.return_logp:
            X = X.view(X.size(0), -1)
        

            X_center = X - self.mean_
            X_transformed = X_center @ self.components_[:self.n_components_].t()
            pca_mean = X_transformed.mean(0) ##
            pca_std  = X_transformed.std(0) ##             
            # return self.U[..., :self.n_components_] * S[:.self.n_components_]
            if not for_fit:
                return X_transformed #BD
            else:
                return X_transformed, pca_mean, pca_std
        else:
            return self._forward(X) #tuple(Tensor, Tensor)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input to transformed pca cooridinates
        :param input: 
        :return: 
        """
        self.fit(input)
        z  = self.transform(input)
        return z
    
    def decode(self, z: Tensor, input: Tensor) -> List[Tensor]:
        """
        Decodes the input from transformed pca cooridinates
        :param input: 
        :return: 
        """
        result  = self.inverse_transform(z)
        return result
        

    def forward(self, input: Tensor):
        z = self.encode(input)
        out = self.decode(z, input)
        return [out, input, z]

    def loss_function(self, 
                      *args,
                      **kwargs) -> dict:
        recon, input, z = args
        recons_loss = F.mse_loss(recon, input)
        return {'loss': recons_loss}


    def inverse_transform(self, X: torch.Tensor):
        if not self.return_logp:
            Xhat = X @ self.Vt[:self.n_components_] + self.mean_
            return Xhat.view(X.size(0), -1, 3) #BLC
        else:
            return self._inverse(X) #tuple(Tensor, Tensor)

    #BELOWs: https://github.com/noegroup/bgflow/blob/main/bgflow/nn/flow/crd_transform/pca.py#:~:text=.std))-,def%20_whiten(self%2C%20x)%3A,return%20y%2C%20dlogp,-Footer
    def _whiten(self, X):
        # Whiten
        X = X.view(X.size(0), -1)
        X_center = X - self.mean_
        output_z = X_center @ self.components_[:self.n_components_].t()
        #Jacobian
        dlogp = self.jacobian_xz * torch.ones((X.shape[0], 1)).to(X)

        return output_z, dlogp

    def _blacken(self, X):
        # Blacken
        Xhat = X @ self.Vt[:self.n_components_] + self.mean_
        output_x = Xhat.view(X.size(0), -1, 3)
        # Jacobian
        dlogp = -self.jacobian_xz * torch.ones((X.shape[0], 1)).to(X)

        return output_x, dlogp

    def _forward(self, X, *args, **kwargs):
        y, dlogp = self._whiten(X)
        return y, dlogp

    def _inverse(self, X, *args, **kwargs):
        y, dlogp = self._blacken(X)
        return y, dlogp

    @property
    def jacobian_xz(self, ):
        std = self.singular_values_.sqrt()[:self.n_components_]
        return -std.log().sum()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
   


    B = 5000
    N = 1000
    C = 3
    x = torch.randn(B, N, C) # (B x C x H x W) 

    model_config = {'full_matrices':False, 'n_components':64}


    model = PCA_torch(args, **model_config) # C_out
    y = model(x)
    loss = model.loss_function(*y)
    print(loss)
    print(torch.allclose(x,  y[0], atol=1e-3), )
