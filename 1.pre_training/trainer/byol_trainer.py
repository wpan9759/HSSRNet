import torch
from torch.utils.data.dataloader import DataLoader
from loss.byol_loss import BYOLLoss
import os
import matplotlib.pyplot as plt 

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, batch_size, max_epochs, m, lr_scheduler, save_dir=None, save_freq=10):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = max_epochs
        self.m = m
        self.batch_size = batch_size
        self.lr_scheduler= lr_scheduler
        self.loss_fn = BYOLLoss()
        self.loss_history = []
        self.best_loss = float('inf')
        self.save_dir = save_dir
        self.save_freq = save_freq
        
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            

    
    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def train(self, train_dataset):

        if self.save_dir:
            self.loss_file = open(os.path.join(self.save_dir, "loss_log.txt"), "w")
            self.loss_file.write("epoch,loss\n")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.initializes_target_network()
        for epoch_counter in range(self.max_epochs):
            epoch_loss = 0.0
            for (batch_view_1, batch_view_2) in train_loader:
                batch_view_1 = batch_view_1.to(self.device).unsqueeze(1)
                batch_view_2 = batch_view_2.to(self.device).unsqueeze(1)
                
                loss = self.update(batch_view_1, batch_view_2)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()               
                self._update_target_network_parameters()
            self.lr_scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            self.loss_history.append(avg_loss)
            print(f"Epoch {epoch_counter} Loss: {avg_loss:.4f}")
            if self.save_dir:
                self.loss_file.write(f"{epoch_counter},{avg_loss:.4f}\n")

            if self.save_dir:
                self._save_checkpoint(epoch_counter, avg_loss)

        if self.save_dir:
            self.loss_file.close()
            self._plot_loss_curve()
    
    def _save_checkpoint(self, epoch, loss):
        if (epoch + 1) % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save({
                'online': self.online_network.state_dict(),
                'target': self.target_network.state_dict(),
                'predictor': self.predictor.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'loss': loss
            }, path)

        if loss < self.best_loss:
            self.best_loss = loss
            path = os.path.join(self.save_dir, "best_model.pth")
            torch.save(self.online_network.state_dict(), path)
            
    def update(self, batch_view_1, batch_view_2):

        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))


        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.loss_fn.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.loss_fn.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()
    
    def _plot_loss_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, 'b-o', linewidth=2)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        

        save_path = os.path.join(self.save_dir, "loss_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    