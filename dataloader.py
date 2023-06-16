import torch 
import os 
import numpy as np
from typing import *
import argparse

class ProteinDataset(torch.utils.data.Dataset):
    """Normalized dataset and reverse-normalization happens here..."""
    def __init__(self, args: argparse.ArgumentParser,  dataset: List[Union[torch.Tensor, np.array]]):
        super().__init__()
        self.reference = dataset[0]
        self.trajectory = dataset[1]
        
        # if os.path.exists(args.standard_file):
        #     mean_and_std_of_trained = torch.from_numpy(np.load(args.standard_file))
        #     mean, std = mean_and_std_of_trained[0], mean_and_std_of_trained[1]
        #     self.trajectory, self.mean, self.std = self.normalize(self.trajectory, mean, std) #Raw (x,y,z) to Normalized (x,y,z)
        # else:
        self.trajectory, self.mean, self.std = self.normalize(self.trajectory) #Raw (x,y,z) to Normalized (x,y,z)
        #np.save(args.standard_file, torch.stack([self.mean, self.std], dim=0).detach().cpu().numpy() )
        if args.which_model == 'conv_vae':
            assert self.reference.ndim == 4 and self.trajectory.ndim == 4, "dimensions are incorrect..."
        else: 
            assert self.reference.ndim == 3 and self.trajectory.ndim == 3, "dimensions are incorrect..."

        self.min = None
        self.max = None
        
    def __len__(self):
        return len(self.trajectory) #train length...
    
    def __getitem__(self, idx):
        coords = self.trajectory[idx] #B, L, 3
        return coords
    
    def normalize(self, coords, mean=None, std=None):
        if mean == None or std == None:
            coords = coords.view(coords.size(0), -1)
            mean = coords.mean(dim=0) #(B,C)
            std = coords.std(dim=0) #(B,C)
            coords_ = (coords - mean) / std
            coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
            return coords_, mean, std #coords_ is SCALED BL3 shape dataset!
        else:
            coords = coords.view(coords.size(0), -1)
            coords_ = (coords - mean) / std
            coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
            return coords_, mean, std #coords_ is SCALED BL3 shape dataset!

    @staticmethod
    def unnormalize(coords, mean=None, std=None, min=None, max=None, index: torch.LongTensor=None):
        #min max index are placeholders, doing nothing...
        assert mean != None and std != None, "Wrong arguments..."
        coords = coords.view(coords.size(0), -1)
        coords_ = (coords * std) + mean
        coords_ = coords_.view(coords.size(0), coords.size(1)//3 ,3) #Back to original shape (B,L,3)
        return coords_ #Reconstructed unscaled (i.e. raw) dataset (BL3)


def extract_trajectory(args):
    N = 1200 # Number of atoms
    B = 128   # Batch size   
    # if args.which_model =='conv_vae':
    #     data = torch.randn(B, 1, N, N)
    #     # fake open state data        
    #     mean = 100
    #     std = 10    
    #     data_open = std * data + mean
    #     # fake close state data 
    #     mean = 200
    #     std = 5       
    #     data_close = std * data + mean   
    #     reference = data_open[0][None]
    #     trajectory = torch.cat([data_open, data_close], dim=0)            

    # else:
    data = torch.randn(B, N, 3)
    # fake open state data
    mean = 100
    std = 10    
    data_open = std * data + mean

    # fake close state data 
    mean = 200
    std = 5       
    data_close = std * data + mean

    reference = data_open[0][None]
    trajectory = torch.cat([data_open, data_close], dim=0)
    return reference, trajectory
    

class ProteinDatasetDistogram(torch.utils.data.Dataset):
    """Normalized dataset and reverse-normalization happens here..."""
    def __init__(self, args: argparse.ArgumentParser, dataset: List[Union[torch.Tensor, np.array]]):
        super().__init__()
        self.args = args
        self.reference = dataset[0]
        self.trajectory = dataset[1]
        # if os.path.exists(args.standard_file_distogram):
        #     mean_and_std_of_trained = torch.from_numpy(np.load(args.standard_file_distogram))
        #     mean, std = mean_and_std_of_trained[0], mean_and_std_of_trained[1]
        #     self.trajectory, self.mean, self.std = self.normalize(self.trajectory, mean, std) #Raw (x,y,z) to Normalized (x,y,z)
        #     self.min, self.max = self.trajectory.min(), self.trajectory.max()
        #     self.trajectory = (self.trajectory - self.min) / (self.max - self.min) #2nd normalization; range [0-1]
        # else:
        self.trajectory, self.mean, self.std = self.normalize(self.trajectory) #Raw (x,y,z) to Normalized (x,y,z)
        #np.save(args.standard_file_distogram, torch.stack([self.mean, self.std], dim=0).detach().cpu().numpy() )
        self.min, self.max = self.trajectory.min(), self.trajectory.max()
        self.trajectory = (self.trajectory - self.min) / (self.max - self.min) #2nd normalization; range [0-1]
        assert self.reference.ndim == 3 and self.trajectory.ndim == 4, "dimensions are incorrect..."
        # self.min = None
        # self.max = None
        
    def __len__(self):
        return len(self.trajectory) #train length...
    
    def __getitem__(self, idx):
        assert isinstance(self.args.truncate, int), "truncate argument for window selection for distogram must be an integer..."
        b, c, d0, d1 = self.trajectory.size()
        
        if self.args.truncate > 0:
            start = np.random.randint(d0 - self.args.truncate) #choose an integer btw [0, d-truncate]
            end = start + self.args.truncate
            dists = self.trajectory[idx, :, start:end, start:end] #B,1,trunc,trunc
            # print(start, end, dists.shape)
            dists = dists.contiguous()
            return dists #, torch.LongTensor([start, end])
        else:
            dists = self.trajectory[idx]
            dists = dists.contiguous()
            return dists #, None
        
    def normalize(self, coords, mean=None, std=None):
        batch_dists = torch.cdist(coords, coords) #BLL
        if mean == None or std == None:
            mean = batch_dists.mean(dim=0) #(LL)
            std = batch_dists.std(dim=0) #(LL)
            dists_ = (batch_dists - mean) / std #BLL
            dists_ = dists_[:,None,...] #->B1LL
            return dists_, mean, std #B1LL, LL, LL
        else:
            dists_ = (batch_dists - mean) / std
            dists_ = dists_[:,None,...] #->B1LL
            return dists_, mean, std  #B1LL, LL, LL

    @staticmethod
    def unnormalize(dists, mean=None, std=None, min=None, max=None, index: torch.LongTensor=None):
        # print(dists.shape, mean.shape, std.shape)
        b, _, l0, l1 = dists.size() #B1LL
        dists = dists.view(b, l0, l1) #-> BLL
        dists = dists * (max - min) + min
        assert mean != None and std != None and min != None and max != None, "Wrong arguments..."
        dists_ = (dists * std) + mean #BLL
        # dists_ = dists_[:,index[:,0]:index]
        #WIP for indexing of distance!
        # if index[0] is not None:
        #     dists_ = torch.stack([dists_[i, s:e, s:e] for i, (s,e) in enumerate(index.unbind(dim=0))], dim=0) #->BLL
        # else:
        #     pass
        return dists_ #Reconstructed unscaled BLL
    

class DataModule():
    def __init__(self, args=None, **kwargs):
        super(DataModule, self).__init__()
        datasets = extract_trajectory(args) #tuple of reference and traj
        self.dataset = ProteinDataset(args, datasets) if args.which_model != 'conv_vae' else ProteinDatasetDistogram(args, datasets)
        self.reference = self.dataset.reference #Reference data of (1,L,3); This is Unscaled (real XYZ)
        self.mean = self.dataset.mean
        self.std = self.dataset.std
        self.min = self.dataset.min
        self.max = self.dataset.max
     
        self.batch_size = args.batch_size
        split_portion   = args.split_portion
        split_method    = args.split_method

        self.seed = kwargs.get("seed", 42)          
        if split_method == 'end': 
            split_portion = split_portion[0]
            assert split_portion <= 100 and split_portion > 0, "this parameter must be a positive number equal or less than 100..."
            self.split_portion = (split_portion / 100) if split_portion > 1 else split_portion
            self.train_data_length = int(len(self.dataset) * self.split_portion)
            self.valid_data_length = int(len(self.dataset) * (1 - self.split_portion)/2 )
            
            self.train_val_dataset = self.dataset[ : (self.train_data_length+self.valid_data_length) ]
            self.test_dataset      = self.dataset[(self.train_data_length+self.valid_data_length) : ]
            self.split_portion = [self.split_portion]
            
        if split_method == 'middle': 
            assert len(split_portion) == 2, "Need two numbers to indicate the testing data in the middle"
            assert split_portion[1] - split_portion[0] >= 0, "second number need to be bigger than first number"
            assert max(split_portion) <= 100 and all(split_portion) > 0, "each parameter must be a positive number and the sum should equal or less than 100..."          
            self.split_portion = split_portion
            self.split_portion_begin = (split_portion[0] / 100) if split_portion[0] > 1 else split_portion[0]
            self.split_portion_end   = (split_portion[1] / 100) if split_portion[1] > 1 else split_portion[1]
            
            self.train_data_length_begin = int(len(self.dataset) * self.split_portion_begin)
            self.train_data_length_end   = int(len(self.dataset) * self.split_portion_end)       
            self.train_data_length       = self.train_data_length_begin + len(self.dataset) - self.train_data_length_end # int(len(self.dataset) * self.split_portion_begin + len(self.dataset)*(1 - self.split_portion_end))
            
            self.valid_data_length       = int(len(self.dataset) * (self.split_portion_end - self.split_portion_begin)/2 )

            self.train_val_dataset  = self.dataset[np.r_[: self.train_data_length_begin + self.valid_data_length, self.train_data_length_end: len(self.dataset)]]
            self.test_dataset       = self.dataset[self.train_data_length_begin + self.valid_data_length : self.train_data_length_end]
        
        self.num_workers = args.num_workers

    #@pl.utilities.distributed.rank_zero_only
    def setup(self, stage=None):
        self.trainset, self.validset= torch.utils.data.random_split(self.train_val_dataset, [self.train_data_length, self.valid_data_length], generator=torch.Generator().manual_seed(self.seed)) 
        self.testset = self.test_dataset #Pristine Last frames in correct times...
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.trainset, shuffle=True, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.validset, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)

    def test_dataloader(self):
        if not self.split_portion[0] in [100., 1.]:
            return torch.utils.data.DataLoader(self.testset, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)
        else:
            return torch.utils.data.DataLoader(self.dataset, shuffle=False, num_workers=self.num_workers, batch_size=self.batch_size, drop_last=False, pin_memory=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--split_portion', '-spl', type=int,  default=[60, 80], nargs="*", help='Torch dataloader and Pytorch lightning split of batches')
    parser.add_argument('--split_method', '-spl_m', type=str, default= 'middle', choices=['middle', 'end'], help=' end: testing with data in the end; middle: testing with data in the middle')   
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data prefetch') 
    parser.add_argument('--which_model', type=str, default='vae', help='model') 

    args = parser.parse_args()

    # args.split_method = 'end'
    # args.split_portion = [1]
    # args.split_method = 'middle'
    # args.split_portion = [1, 1]

    data = DataModule(args)
    data.setup()
    traindata, valdata, testdata = data.train_dataloader(), data.val_dataloader(), data.test_dataloader()

    
    print(len(traindata), len(valdata), len(testdata))
    print(next(iter(traindata)).shape, next(iter(testdata)).shape)
