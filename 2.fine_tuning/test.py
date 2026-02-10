import argparse
import torch
import os
import logging
import warnings
import pandas as pd

import tqdm

from torch import nn
from datasets import MyDataset
from torch.utils.data import DataLoader


class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model
    def forward(self, input, label): 
        input = input.unsqueeze(1)
        output = self.model(input)       
        return output

def get_model(args, device, models):
    print(models)
    nclass = args.nclass
    input_length = args.input_length
    
    from models.resnet import ResNet1D
    model = ResNet1D(num_classes = nclass, input_length = input_length)

    model = FullModel(model)
    model = model.to(device)
    return model


def main():
    args2 = parse_args()
    
    if args2.device == 'cpu':
        device = torch.device('cpu')
    elif args2.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = MyDataset(path = args2.test_path, test = True)
    test_loader = DataLoader(test_dataset, batch_size=args2.test_batchsize, shuffle=False)
    
    model = get_model(args2, device, models=args2.models)
    model.eval()
    predictions = []
    with torch.no_grad():
        model_state_file = args2.weight_path
        if os.path.isfile(model_state_file):
            print('loading checkpoint successfully')
            logging.info("=> loading checkpoint '{}'".format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint)
        else:
            warnings.warn('weight is not existed !!!"')

        for sample_ids, data, label in tqdm.tqdm(test_loader):
            data, label = data.to(device), label.to(device)
            label = label.long()
            logit = model(data, label)
            
            logit = logit.argmax(dim=1)
            logit = logit.cpu().detach().numpy()
            

            for id, pred in zip(sample_ids, logit):
                predictions.append({"ID": id, "Prediction": pred})
  

    result_df = pd.DataFrame(predictions)
    result_df.to_csv("test_predictions.csv", index=False)
    print("预测结果已保存到 test_predictions.csv")
  
def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--test_batchsize", type=int, default=32)
    parser.add_argument("--input_length", type=int, default=2151)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select device (cpu/gpu)')
    parser.add_argument("--test_path", type=str, default='data/test.csv')
    parser.add_argument("--weight_path", type=str, default='best_weight.pkl')
    
    args2 = parser.parse_args()
    return args2    

if __name__ == '__main__':
    main()