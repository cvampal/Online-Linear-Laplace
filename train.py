import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from model import MLP
from dataset import *

class OnlineLearner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.train_datasets, self.test_datasets = get_permuted_MNIST(num_task=cfg['num_task'], seed=cfg['seed'])
        self.model = MLP(output_size=cfg['num_class']).to(self.device)
        
    def evaluate(self, idx):
        acc = []
        for i, ds_val in tqdm.tqdm(enumerate(self.test_datasets[:idx])):
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
        
        return np.array(acc).mean()
    
    def train(self, dataset, iters):
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
            loss = torch.nn.functional.cross_entropy(input=y_hat, target=y, reduction='mean')
            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
            '<CLASSIFIER> | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(loss=loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
        progress_bar.close()
            
    def train_all(self):
        for i in range(self.cfg['num_task']):
            self.train(self.train_datasets[i], self.cfg['epoch'])
            print(self.evaluate(i))
        
        
cfg = {"device": 'cuda',
       "num_task": 5,
       "num_class": 10,
       "seed": 42,
       "batch_size": 256,
       "lr": 0.0001,
       "epoch": 200
    }

l = OnlineLearner(cfg)
l.train_all()