import argparse
import torch
import random
import numpy as np
import os
import time
import tqdm
import logging

from datasets import MyDataset
from torch.utils.data import DataLoader
from torch import nn

from class_metric import ClassMetric

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


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, label):
        loss = self.ce_loss(prediction, label)
        return loss
    

def get_dataset_loaders(args):
       
    train_dataset = MyDataset(path = args.train_path)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True)
    
    val_dataset = MyDataset(path = args.val_path)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batchsize, shuffle=True)
    
    return train_loader, val_loader

class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model
        self.ce_loss = CrossEntropyLoss()
    def forward(self, input, label):
        input = input.unsqueeze(1)
        output = self.model(input)       
        losses = self.ce_loss(output, label)
        return losses, output

def get_model(args, device, models):
    print(models)
    nclass = args.nclass
    input_length = args.input_length
      
    from models.resnet import ResNet1D
    model = ResNet1D(num_classes = nclass, input_length = input_length)
    
    if args.use_pretrain:
        pretrained_dict = torch.load(args.pretrain_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if not k.startswith(('dense'))}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded pretrained weights from {args.pretrain_path}")

    model = FullModel(model)
    model = model.to(device)
    return model
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def train(dataloader_train, device, model, args2, optimizer, epoch):
    model.train()
    
    ACC = [0]
    F1 = [0]
    Precision = [0]
    Recall = [0]
    
    ave_loss = AverageMeter()
    nclass = args2.nclass
    metric = ClassMetric(nclass)
    for _, data, label in tqdm.tqdm(dataloader_train):
        data, label = data.to(device), label.to(device)
        label = label.long()
        losses, logit = model(data, label)
        loss = losses.mean()
        ave_loss.update(loss.item())
        
        logit = logit.argmax(dim=1)
        logit = logit.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        metric.addBatch(logit, label)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        
    reduced_loss = ave_loss.average()
    print_loss = torch.from_numpy(np.array(reduced_loss)).to(device).cpu().item()  
    
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()
    mprecision = np.nanmean(precision[0:args2.nclass])
    mrecall = np.nanmean(recall[0:args2.nclass])
    
    ACC = ACC + acc
    Recall = Recall + mrecall
    Precision = Precision + mprecision
    F1 = F1 + 2 * Precision * Recall / (Precision + Recall)
    
    ACC = torch.from_numpy(ACC).to(device).item()
    F1 = torch.from_numpy(F1).to(device).item()
    Recall = torch.from_numpy(Recall).to(device).item()
    Precision = torch.from_numpy(Precision).to(device).item()
    Lr = optimizer.param_groups[0]['lr']
    return {
        "acc_t":ACC,
        "f1_t":F1,
        "precision_t":Precision,
        "recall_t":Recall,
        "loss_t":print_loss,
        "lr_t":Lr
        }

def validate(dataloader_val, device, model, args2):
    model.eval()
    
    ACC = [0]
    F1 = [0]
    Precision = [0]
    Recall = [0]
    
    ave_loss = AverageMeter()
    nclass = args2.nclass
    metric = ClassMetric(nclass)
    with torch.no_grad():
        for _, data, label in tqdm.tqdm(dataloader_val):
            data, label = data.to(device), label.to(device)
            label = label.long()
            losses, logit = model(data, label)
            loss = losses.mean()
            ave_loss.update(loss.item())
            
            logit = logit.argmax(dim=1)
            logit = logit.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            metric.addBatch(logit, label)
           
    reduced_loss = ave_loss.average()
    print_loss = torch.from_numpy(np.array(reduced_loss)).to(device).cpu().item()  
    
    acc = metric.Accuracy()
    precision = metric.Precision()
    recall = metric.Recall()
    mprecision = np.nanmean(precision[0:args2.nclass])
    mrecall = np.nanmean(recall[0:args2.nclass])
    
    ACC = ACC + acc
    Recall = Recall + mrecall
    Precision = Precision + mprecision
    F1 = F1 + 2 * Precision * Recall / (Precision + Recall)
    
    ACC = torch.from_numpy(ACC).to(device).item()
    F1 = torch.from_numpy(F1).to(device).item()
    Recall = torch.from_numpy(Recall).to(device).item()
    Precision = torch.from_numpy(Precision).to(device).item()
    return {
        "acc_v":ACC,
        "f1_v":F1,
        "precision_v":Precision,
        "recall_v":Recall,
        "loss_v":print_loss,
        }      
        
def main():
    args2 = parse_args()
    
    seed_torch(args2.seed)
       
    save_name = "{}_lr{}_epoch{}_batchsize{}".format(args2.models, args2.lr, args2.end_epoch,
                                                        args2.train_batchsize)
    save_dir = args2.save_dir
    if not os.path.exists(os.path.join(save_dir, save_name)):
        os.makedirs(os.path.join(save_dir, save_name) + '/weights/')
    weight_save_dir = os.path.join(save_dir, save_name + '/weights')
    logging.basicConfig(filename=os.path.join(save_dir, save_name) + '/train.log', level=logging.INFO)
    
    train_loader, val_loader = get_dataset_loaders(args2)
    
    if args2.device == 'cpu':
        device = torch.device('cpu')
    elif args2.device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(args2, device, models=args2.models)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args2.lr)    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    
    best_acc = 0
    
    for epoch in range(args2.end_epoch):
        print("Epoch: {}/{}".format(epoch + 1, args2.end_epoch))
        start_train = time.time()
        train_hist = train(train_loader, device, model, args2, optimizer, epoch)
        end_train = time.time()
        train_time = end_train - start_train
        print("train_epoch:[{}/{}], acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, lr:{:.6f}, time:{:.4f}s".
              format(epoch + 1,args2.end_epoch, train_hist['acc_t'], train_hist['f1_t'], train_hist['precision_t'], train_hist['recall_t'], train_hist['loss_t'], train_hist['lr_t'], train_time))
        lr_scheduler.step()
        
        start_val = time.time() 
        val_hist = validate(val_loader, device, model, args2)#开始验证
        end_val = time.time()
        val_time = end_val - start_val
        print("valid_epoch:[{}/{}], acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, time:{:.4f}s".
              format(epoch + 1,args2.end_epoch, val_hist['acc_v'], val_hist['f1_v'], val_hist['precision_v'], val_hist['recall_v'], val_hist['loss_v'], val_time))
        logging.info("train_epoch:[{}/{}], acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f},lr:{:.6f}, time:{:.4f}s, valid_epoch:[{}/{}], acc:{:.4f}, f1:{:.4f},precision:{:.4f},recall:{:.4f},loss:{:.4f}, time:{:.4f}s".
                     format(epoch + 1,args2.end_epoch, train_hist['acc_t'], train_hist['f1_t'], train_hist['precision_t'], train_hist['recall_t'], train_hist['loss_t'], train_hist['lr_t'], train_time, 
                            epoch + 1,args2.end_epoch, val_hist['acc_v'], val_hist['f1_v'], val_hist['precision_v'], val_hist['recall_v'], val_hist['loss_v'], val_time)
                     )
        torch.save(model.state_dict(),
                       weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_epoch_{}.pkl'
                       .format(args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch))
        if val_hist['acc_v'] >= best_acc and val_hist['acc_v'] != 0:
            best_acc = val_hist['acc_v']
            best_weight_name = weight_save_dir + '/{}_lr{}_epoch{}_batchsize{}_best_epoch_{}.pkl'.format(
                    args2.models, args2.lr, args2.end_epoch, args2.train_batchsize, epoch)
            torch.save(model.state_dict(), best_weight_name)
            torch.save(model.state_dict(), weight_save_dir + '/best_weight.pkl')
        
def parse_args():
    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--end_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--val_batchsize", type=int, default=32)
    parser.add_argument("--input_length", type=int, default=2151)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default='./work_dir')
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='Select device (cpu/gpu)')
    parser.add_argument("--train_path", type=str, default='data/train.csv')
    parser.add_argument("--val_path", type=str, default='data/val.csv')
    parser.add_argument("--use_pretrain", action='store_true', default = True, help="Use pretrained weights")
    parser.add_argument("--pretrain_path", type=str, 
                      default="best_model.pth",
                      help="Path to pretrained weights")
    
    
    args2 = parser.parse_args()
    return args2    

if __name__ == '__main__':
    main()