import torch
import pandas as pd
from torch.utils.data import Dataset
from .transformer import add, multi, slop

class SpectralDataset(Dataset):
    def __init__(self, path, aug_p=0.5):
        
        self.df = pd.read_csv(path)
        self.X = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float32)
        self.shape = self.df.shape
        self.add_aug = add(p=aug_p)
        self.multi_aug = multi(p=aug_p)
        self.slop_aug = slop(p=aug_p)
        
    def __getitem__(self,idx):
        
        x1 = self.X[idx].unsqueeze(0)
        x2 = self.X[idx].unsqueeze(0)
        x1_np = x1.numpy()
        x2_np = x2.numpy()
        x1_np = self.add_aug(x1_np)
        x2_np = self.add_aug(x2_np)
        x1_np = self.multi_aug(x1_np)
        x2_np = self.multi_aug(x2_np)
        x1_np = self.slop_aug(x1_np)
        x2_np = self.slop_aug(x2_np)
        
        x1 = torch.from_numpy(x1_np).squeeze(0).float()
        x2 = torch.from_numpy(x2_np).squeeze(0).float()
        
        return x1, x2
    
    def __len__(self):
        
        return self.shape[0]
        