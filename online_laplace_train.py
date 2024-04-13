import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from model import MLP
from dataset import *
from laplace import Laplace
from laplace.curvature import AsdlGGN, AsdlEF
from torch.nn.utils import parameters_to_vector


class OnlineLearner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda:3' if torch.cuda.is_available() and cfg['cuda'] else 'cpu')
        self.train_datasets, self.test_datasets = get_permuted_MNIST(num_task=cfg['num_task'], seed=cfg['seed'])
        self.model = MLP(output_size=cfg['num_class']).to(self.device)
        self.la =  Laplace(self.model, 'classification',
                            subset_of_weights = 'all',
                            hessian_structure = cfg["hessian_structure"],
                            prior_mean = torch.zeros_like(parameters_to_vector(self.model.parameters())),
                            prior_precision = cfg["prior_prec_init"],
                            backend = AsdlGGN if cfg["approx_type"] == 'ggn' else AsdlEF)
        self.validation_acc = []
        
    def evaluate(self, idx):
        acc = []
        
        for i, ds_val in enumerate(self.test_datasets[:idx]):
            dl = torch.utils.data.DataLoader(ds_val, batch_size=1000)
            correct, counts = 0, 0
            for x,y in dl:
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    #pred = self.la(x, pred_type=self.cfg['pred_type'], link_approx=self.cfg['link_approx'], n_samples=self.cfg['mc_samples'])
                    pred = self.model(x)
                    pred = pred.argmax(-1).view(-1)
                correct += (pred == y).sum().item()
                counts += x.shape[0]
            acc += [correct/counts]
        print(acc)
        return np.array(acc).mean()
    
    def train(self, dataset, iters, idx):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=1e-5)
        self.model.train()
        iters_left = 1
        N = len(dataset)
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
            laplace_loss = - self.la.log_prob(parameters_to_vector(self.model.parameters())) / N
            loss = classifier_loss + self.cfg['lambda'] * laplace_loss
            accuracy = (y == y_hat.max(1)[1]).sum().item()*100 / x.size(0)
            loss.backward()
            optimizer.step()
            progress_bar.set_description(
            'Task {no} | training loss: {loss:.3} | training accuracy: {prec:.3}% |'
                .format(no=idx, loss=loss.item(), prec=accuracy)
            )
            progress_bar.update(1)
            scheduler.step()
        progress_bar.close()
    

            
    def train_all(self):
        for i in range(self.cfg['num_task']):
            self.train(self.train_datasets[i], self.cfg['epoch'], idx=i+1)
            self.la.fit(torch.utils.data.DataLoader(self.train_datasets[i], batch_size=self.cfg['batch_size'], shuffle=True), override=False)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
        name = "_".join([self.cfg['dataset_name'], self.cfg['hessian_structure'], self.cfg['approx_type']])
        torch.save(torch.tensor(self.validation_acc), f"./plots/{name}.pt")

        

if __name__ == '__main__':      
    import sys
    #Can update new configurations here  
    cfg = {
        "dataset_name" : "permuted-mnist",
        "num_task": 50,
        "num_class": 10,
        "seed": 42,
        "batch_size": 256,
        "lr": 0.001,
        "epoch": 1000,
        "hessian_structure": 'diag', # ['diag', 'kron', 'full']
        "approx_type": "ggn", # ['ggn', 'ef'] 
        "prior_structure": "scaler", # ['all', 'layerwise', 'scalar']
        "prior_prec_init": 1e-3,
        "lambda": 1,
        "pred_type": "glm", # ['glm', 'nn']
        "link_approx" : "probit",    #['mc', 'probit', 'bridge', 'bridge_norm']
        "mc_samples": 100        
        }
    cfg["hessian_structure"] = sys.argv[1]
    cfg["approx_type"] = sys.argv[2]
    cfg['cuda'] = True # bool(int(sys.argv[3]))
    print(cfg)
    trainer = OnlineLearner(cfg)
    trainer.train_all()