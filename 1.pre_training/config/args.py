import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SSL")
    
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--data_path", type=str, 
                      default="dataset/all.csv", 
                      help="Path to training data")
    

    parser.add_argument("--aug_p", type=float, default=0.5, help="Augmentation probability")


    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.96, help="LR scheduler gamma")
    

    parser.add_argument("--m", type=float, default=0.996, help="EMA decay rate")
    

    parser.add_argument("--save_freq", type=int, default=10, help="Model saving frequency (epochs)")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Base directory for saving models")
    
    return parser.parse_args()