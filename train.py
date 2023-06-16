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
        
        # model selection
        self.args = args
        model_configs = kwargs.get("model_configs", None)
        if args.which_model == 'vae':
            self.model_block = VAE(args, **model_configs) 
        if args.which_model == "conv_vae":
            self.model_block = ConvVAE(args, **model_configs)                 
        elif args.which_model == "ae":
            self.model_block = AE(args, **model_configs)       
        elif args.which_model == "pca":
            self.model_block = PCA_torch(args, **model_configs)   
    
        # self.beta = args.beta
        # self.data_mean = None
        # self.data_std = None
        # self.loader_length = None

        # data loading ...
        self.traindata = None
        self.valdata = None
        self.testdata = None

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


    def train_(self):
        args = self.args
        if args.which_model == "pca":
            self.load_data_pca()
            loss_and_others = self.train_pca()
        if args.which_model in ['vae', "conv_vae", "ae"]:
            self.load_data_nn()
            loss_and_others = self.train_nn()
        return loss_and_others


    def load_data_pca(self):
        args = self.args
        data = DataModule(args)
        data.setup()
        self.traindata, self.valdata, self.testdata = data.train_dataloader().dataset.dataset, data.val_dataloader().dataset.dataset, data.test_dataloader().dataset
        return self.traindata, self.valdata, self.testdata

    def train_pca(self):
        #y = model(traindata, args)
        args = self.args
        traindata = self.traindata
        loss = self.loss_function(traindata, args) 
        if args.save_model == True: 
            pass
        if args.train_verbose:
            return [self(traindata, args), loss]
        else: 
            return loss
        
    def load_data_nn(self):
        # dataloader  
        args = self.args
        data = DataModule(args)
        data.setup()
        self.traindata, self.valdata, self.testdata = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()
        return self.traindata, self.valdata, self.testdata

    def train_nn(self):
        traindata = self.traindata
        valdata  = self.valdata
        loss_config = self.args.loss_config
        optimizer = torch.optim.Adam(self.model_block.parameters())
        # train loop
        self.model_block.train()

        for epoch in range(epochs):
            # train loop
            for batch_idx, data in enumerate(traindata):      
                if torch.cuda.is_available():
                    data = data.cuda()
    
                optimizer.zero_grad()
                loss_ = self.loss_function(data, args, **loss_config)
                loss = loss_['loss']
                loss.backward()
                optimizer.step()
            
                print(loss)

            # validation loop
            for batch_idx, data in enumerate(valdata):
                if torch.cuda.is_available():
                    data = data.cuda()
                loss_ = self.loss_function(data, args, **loss_config)
                loss = loss_['loss']
                print(f"this is validation loss: {loss}")

        if args.train_verbose:
            return [loss_ ]
        else: 
            return loss    





if __name__ == '__main__':
    # arguemnts
    parser = argparse.ArgumentParser(description='Training')
    args = parser.parse_args()
    args.batch_size = 16
    args.split_portion = [60, 80]
    args.split_method = 'middle'
    args.num_workers = 1
    args.nolog = True
    epochs = 40
    args.save_model = False
    args.train_verbose = False

    ## PCA model 
    # args.which_model = "pca" 
    # model_config = {'full_matrices':False, 'n_components':64}
    # model_configs = {'model_configs': model_config} 
    
    ## VAE model
    # args.which_model = "vae"   
    # N = 1200 # atom number
    # C = 3
    # model_config = {'input_dim': N*C}
    # model_configs = {'model_configs': model_config}  
    # loss_config   = {'M_N': 0.005}
    # args.loss_config = loss_config

    ## AE model
    # args.which_model = "vae"   
    # N = 1200 # atom number
    # C = 3
    # model_config  = {'input_dim': N*C}
    # model_configs = {'model_configs': model_config}    
    # loss_config   = {} 
    # args.loss_config = loss_config

    ## CONV_VAE model
    args.which_model = "conv_vae"
    N = 100 # atom number
    args.truncate = 100

    model_config = {'input_dim': N, 
                    'in_channels': 1, 
                    'latent_dim': 64} 
    model_configs = {'model_configs': model_config}    
    loss_config   = {'M_N': 0.005}
    args.loss_config = loss_config 

    
    model = Model(args, **model_configs) 
    loss = model.train_()








