import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from model import MLP, estimate_fisher
from dataset import *
from ivon import IVON
from torch.nn.utils import parameters_to_vector
import copy

class OnlineLearner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.train_datasets, self.test_datasets = get_permuted_MNIST(num_task=cfg['num_task'], seed=cfg['seed'])
        self.model = MLP(output_size=cfg['num_class']).to(self.device)
        self.current_parameters = copy.deepcopy(parameters_to_vector(self.model.parameters()).detach().clone())
        self.current_hessian = torch.zeros_like(self.current_parameters)
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
        print(f"Test accuracy: {acc}")
        return np.array(acc).mean()
    
    
    def train(self, dataset, iters, idx):
        optimizer = IVON(self.model.parameters(), lr=self.cfg['lr'], ess=len(dataset), hess_init=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=1e-5)
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
            for _ in range(1):
                with optimizer.sampled_params(train=True):
                    y_hat = self.model(x)
                    optimizer.zero_grad()
                    classifier_loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')
                    laplace_loss = (self.current_hessian * (parameters_to_vector(self.model.parameters()) - self.current_parameters)**2).sum()
                    loss = classifier_loss + self.cfg['lambda'] * laplace_loss
                    loss.backward()
                    
            optimizer.step()
            scheduler.step()
            
            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)
            progress_bar.set_description(
            'Task {no} | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(no=idx, loss=loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
            
        progress_bar.close()
        
        # update currect map and hessian
        self.current_hessian += optimizer.state_dict()['param_groups'][0]['hess'].clone()
        self.current_parameters = copy.deepcopy(parameters_to_vector(self.model.parameters()).detach().clone())
            
    def train_all(self):
        for i in range(self.cfg['num_task']):
            self.train(self.train_datasets[i], self.cfg['epoch'], idx=i+1)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
                
        torch.save(torch.tensor(self.validation_acc), f"./plots/{self.cfg['train_mode']}.pt")

        

if __name__ == '__main__':      
    
    # Can update new configurations here  
    cfg = {"device": 'cuda',
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 256,
        "lr": 0.1,
        "epoch": 1000,
        "train_mode": 'ivon',
        "lambda": 1.0,
        }
    l = OnlineLearner(cfg)
    l.train_all()
    
