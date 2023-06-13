from model import ConvVAE, VAE, AE, PCA_torch
import torch
from torch import nn
from torch.nn import functional as F
from typing import *
from einops import rearrange, repeat, reduce
import wandb
import os 
from dataloader import *
import argparse


class Model(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.args = args
        model_configs = kwargs.get("model_configs", None)
        if args.which_model == 'vae':
            self.model_block = VAE(args, **model_configs) 
        if args.which_model == "conv_vae":
            self.model_block = ConvVAE(args, **model_configs)                 
        elif args.which_model == "ae":
            self.model_block = AE(args, **model_configs)       
        elif args.which_model == "pca":
            self.model_block = PCA_torch(**model_configs)   
    
        # self.beta = args.beta
        # self.data_mean = None
        # self.data_std = None
        # self.loader_length = None

        if self.args.nolog:
            pass
        else:
            self.wandb_run = wandb.init(project="VAEGAN_MD", entity="yifei6", name=args.name, group="DDP_runs")
            wandb.watch(self.model_block, log="all")
            os.environ["WANDB_CACHE_DIR"] = os.getcwd()

    def forward(self, coords, args, **kwargs):
        #WIP: add atom/res types!
        coords = coords.contiguous()
        y = self.model_block(coords)
        return y
    

    def loss_function(self, coords,  args, **kwargs):
        y = self(coords, args)
        loss = self.model_block.loss_function(*y, **kwargs)
        return loss


def train(args, 
          model, 
          traindata,
          valdata,):
    if args.which_model == "pca":
        train_pca(args, model, traindata, valdata)
    if args.which_model in ['vae', "conv_vae", "ae"]:
        train_nn(args, model, traindata, valdata)

    

def train_nn(args, 
          model,
          traindata, 
          valdata, 
          save_model = True,):
     # train loop
    model.train()
    for epoch in range(epochs):
        # train loop
        for batch_idx, data in enumerate(traindata):      
            if torch.cuda.is_available():
                data = data.cuda()
 
            optimizer.zero_grad()
            loss_ = model.loss_function(data, args, **loss_config)
            loss = loss_['loss']
            loss.backward()
            optimizer.step()
        
            print(loss)


        # validation loop
        for batch_idx, data in enumerate(valdata):
            if torch.cuda.is_available():
                data = data.cuda()
            loss_ = model.loss_function(data, args, **loss_config)
            loss = loss_['loss']
            print(f"this is validation loss: {loss}")

def train_pca(args, 
              model, 
              traindata,
              valdata,
              savee_model = True):
    pass 
    
def load_data_nn(args):
    # dataloader  
    data = DataModule(args)
    data.setup()
    traindata, valdata, testdata = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    return traindata, valdata, testdata

def laod_data_pca(args):
    pass
    data = DataModule(args)
    data.setup()
    traindata, valdata, testdata = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
    # traindata, valdata, testdata = train_dataloaders.dataset.dataset, val_dataloaders.dataset.dataset


if __name__ == '__main__':
    # arguemnts
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
    args.batch_size = 16
    args.split_portion = [60, 80]
    args.split_method = 'middle'
    args.num_workers = 1
    args.which_model = "vae"   
    args.nolog = True
    epochs = 40

    # dataloader  
    data = DataModule(args)
    data.setup()
    traindata, valdata, testdata = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()


    # initialize model (VAE)
    B, N, C = next(iter(traindata)).shape
    model_config = {'input_dim': N*C}
    model_configs = {'model_configs': model_config}  
    loss_config   = {'M_N': 0.005}
    args.model_configs = model_configs  
    args.loss_config = loss_config


    model = Model(args, **model_configs) # C_out
    if torch.cuda.is_available():
        model = model.cuda()


    optimizer = torch.optim.Adam(model.parameters())

    # train loop
    model.train()
    for epoch in range(epochs):
        # train loop
        for batch_idx, data in enumerate(traindata):      
            if torch.cuda.is_available():
                data = data.cuda()
 
            optimizer.zero_grad()
            loss_ = model.loss_function(data, args, **loss_config)
            loss = loss_['loss']
            loss.backward()
            optimizer.step()
        
            print(loss)


        # validation loop
        for batch_idx, data in enumerate(valdata):
            if torch.cuda.is_available():
                data = data.cuda()
            loss_ = model.loss_function(data, args, **loss_config)
            loss = loss_['loss']
            print(f"this is validation loss: {loss}")



