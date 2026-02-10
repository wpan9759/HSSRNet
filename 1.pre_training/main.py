from config.args import parse_args
from models.resnet import ResNet1D
from models.head import BYOLHead
from trainer.byol_trainer import BYOLTrainer
from dataset.spectral_dataset import SpectralDataset
import random
import os
import torch
import numpy as np

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")
    
    seed_torch(args.seed)
        
    base_model = ResNet1D().to(device)
    
    predictor = BYOLHead().to(device)    
    target_network = ResNet1D().to(device)

    optimizer = torch.optim.Adam(
        list(base_model.parameters()) + list(predictor.parameters()), 
        lr=args.lr
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    trainer = BYOLTrainer(
        online_network=base_model,
        target_network=target_network,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        predictor=predictor,
        device=device,
        batch_size = args.batch_size,
        max_epochs = args.max_epochs,
        m = args.m,
        save_dir=args.save_dir,
        save_freq=args.save_freq
    )
    
    train_dataset = SpectralDataset(
        path = args.data_path,
        aug_p=args.aug_p
    )
    
    trainer.train(train_dataset)

if __name__ == '__main__':
    main()