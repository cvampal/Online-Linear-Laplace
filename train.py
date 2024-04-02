import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from model import MLP, estimate_fisher
from dataset import *

class OnlineLearner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.train_datasets, self.test_datasets = get_permuted_MNIST(num_task=cfg['num_task'], seed=cfg['seed'])
        self.model = MLP(output_size=cfg['num_class']).to(self.device)
        self.current_parameter = {}
        self.current_fisher = {}
        self.validation_acc = []
        
    def evaluate(self, idx):
        acc = []
        for i, ds_val in enumerate(self.test_datasets[:idx]):
            dl = torch.utils.data.DataLoader(ds_val, batch_size=1000)
            self.model.eval()
            correct, counts = 0, 0
            for x,y in dl:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x).argmax(-1).view(-1)
                correct += (pred == y).sum().item()
                counts += x.shape[0]
            acc.append(correct/counts)
        #print(f"Test accuracy: {acc}")
        return np.array(acc).mean()
    
    def train(self, dataset, iters, idx):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999))
        self.model.train()
        iters_left = 1
        progress_bar = tqdm.tqdm(range(1, iters+1))
        for batch_index in range(1, iters+1):
            iters_left -= 1
            if iters_left==0:
                data_loader = iter(torch.utils.data.DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=True))
                iters_left = len(data_loader)
            x, y = next(data_loader)
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            y_hat = self.model(x)
            classifier_loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')
            if idx>1 and (self.cfg["train_mode"] == "online_laplace_diagonal"):
                laplace_loss = []
                for n, p in self.model.named_parameters():
                    map_parameter = self.current_parameter[n]
                    fisher = self.current_fisher[n]
                    laplace_loss.append((fisher*(p-map_parameter)**2).sum() )
                laplace_loss = 0.5 * sum(laplace_loss)
            else:
                laplace_loss = 0
            loss = classifier_loss + laplace_loss
            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
            'Task {no} | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(no=idx, loss=loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
        progress_bar.close()
    
    
    def update_laplace_estimate(self, dataset, weight=1.0):
        fisher, self.current_parameter = estimate_fisher(self.model, dataset, n_samples=self.cfg['n_samples_fisher'], device=self.device)
        if self.current_fisher :
            for n, p in fisher.items():
                self.current_fisher[n] += fisher[n] * weight
        else:
            self.current_fisher = fisher
            
    def train_all(self):
        for i in range(self.cfg['num_task']):
            self.train(self.train_datasets[i], self.cfg['epoch'], idx=i+1)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
            if  (self.cfg["train_mode"] == "online_laplace_diagonal"):
                self.update_laplace_estimate(self.train_datasets[i])
                
        torch.save(torch.tensor(self.validation_acc), f"./plots/{self.cfg['train_mode']}.pt")

    def train_cumulative(self):
        for i in range(self.cfg['num_task']):
            
            #Train on all the previous datasets
            # for j in range(i+1):
            #     self.train(self.train_datasets[j], self.cfg['epoch'], idx=j+1)
            temp_dataset = torch.utils.data.ConcatDataset(self.train_datasets[:i+1])
            self.train(temp_dataset, self.cfg['epoch'], idx=i+1)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
                
        torch.save(torch.tensor(self.validation_acc), f"./plots/{self.cfg['train_mode']}.pt")

    def print_data_shape(self):
        for i in range(self.cfg['num_task']):
            self.train_datasets[i] = torch.utils.data.ConcatDataset(self.train_datasets[:i+1])
            print(f"Task {i+1}: Train: {len(self.train_datasets[i])}, Test: {len(self.test_datasets[i])}")
        

if __name__ == '__main__':      
    
    # Can update new configurations here  
    # cfg = {"device": 'cuda',
    #     "num_task": 50,
    #     "num_class": 10,
    #     "seed": 42,
    #     "batch_size": 128,
    #     "lr": 0.01,
    #     "epoch": 200,
    #     "train_mode": 'online_laplace_diagonal',
    #     "n_samples_fisher": 200,
        
    #     }

    cfg = {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 128,
        "lr": 0.001,
        "epoch": 200,
        "train_mode": 'cumulative',        
        }
    l = OnlineLearner(cfg)
    # l.train_all()
    l.train_cumulative()
    # l.print_data_shape()