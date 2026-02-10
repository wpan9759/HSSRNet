import torch
import pandas as pd
from torch.utils.data import Dataset
from transforms import add, multi, slop

class MyDataset(Dataset):
    def __init__(self, path, test = False):
        self.test = test
        self.df = pd.read_csv(path)
        self.ids = self.df.iloc[:, 0].values  # 提取第一列ID
        self.X = torch.tensor(self.df.iloc[:, 2:].values, dtype=torch.float32)
        self.Y = torch.tensor( self.df.iloc[:, 1].values, dtype=torch.float32)
        self.shape = self.df.shape
        self.add_aug = add(p=0.5)
        self.multi_aug = multi(p=0.5)
        self.slop_aug = slop(p=0.5)
    def __getitem__(self,idx):
        sample_id = self.ids[idx]  # 获取样本ID
        if not self.test:
            x = self.X[idx].unsqueeze(0)
            x_np = x.numpy()
            x_np = self.add_aug(x_np)
            x_np = self.multi_aug(x_np)
            x_np = self.slop_aug(x_np)
            
            x = torch.from_numpy(x_np).squeeze(0).float()
            y = self.Y[idx]
        else:            
            x, y = self.X[idx], self.Y[idx]
        
        return (sample_id, x, y)
    def __len__(self):
        return self.shape[0]
    


        
        
        
        