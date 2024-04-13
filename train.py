import numpy as np
import tqdm
import torch
from torchvision import datasets, transforms
from model import MLP, estimate_fisher, update_omega
from dataset import *
from configs import cnfgs
class OnlineLearner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg['device'])
        self.train_datasets, self.test_datasets = get_permuted_MNIST(num_task=cfg['num_task'], seed=cfg['seed'])
        self.model = MLP(output_size=cfg['num_class']).to(self.device)
        
        # Store the current parameter
        self.current_parameter = {}
        
        # List to store the MAP parameters for all the tasks
        self.map_parameters = []
        
        # Store the current fisher information matrix
        self.current_fisher = {}
        
        # Store the current synaptic strength
        self.W = {}
        self.current_omega = {}
        
        self.validation_acc = []
    
    # Initialize the W matrix
    def init_W(self):
        W = {}
        p_old = {}
        omega_init = {}
        for n, p in self.model.named_parameters():
            W[n] = p.detach().clone().zero_()
            p_old[n] = p.data.clone()  
            omega_init[n] = p.detach().clone().zero_()  
        return W, p_old, omega_init
    
    # Update the W matrix
    def update_W(self, W):
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                W[n].add_(-p.grad*(p.detach()-p_old[n]))
            # p_old[n] = p.detach().clone()
        return W
        
    def get_laplace_loss(self, model, current_fisher, parameter):
        laplace_loss = []
        for n, p in model.named_parameters():
            map_parameter = parameter[n]
            fisher = current_fisher[n]
            laplace_loss.append((fisher*(p-map_parameter)**2).sum() )
        return 0.5 * sum(laplace_loss)
    
    def get_si_loss(self, model, current_omega, parameter):
        si_loss = []
        for n, p in model.named_parameters():
            map_parameter = parameter[n]
            omega = current_omega[n]
            si_loss.append((omega*(p-map_parameter)**2).sum() )
        return self.cfg["c"] * sum(si_loss)
    
    def get_ewc_loss(self, model, current_fisher, map_parameters):
        ewc_loss = []
        # EWC Loss, L = 0.5 * sum( fisher * (theta - theta*)^2 )
        for n, p in model.named_parameters():
            # map_parameter = parameter[n]
            fisher = current_fisher[n]
            ewc_penalty = []
            for prev_map_parameter in map_parameters:
                ewc_penalty.append((p-prev_map_parameter[n])**2)
            
            ewc_loss.append((fisher*sum(ewc_penalty)).sum() )
        return 0.5 * sum(ewc_loss)
    
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
        
        # Initialize the W matrix
        self.W, self.current_parameter, self.current_omega = self.init_W()
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
            # if idx == 1 or self.cfg["train_mode"] == "normal":
            #     loss = classifier_loss
            if idx>1 and (self.cfg["train_mode"] == "online_laplace_diagonal"):
                laplace_loss = self.get_laplace_loss(self.model, self.current_fisher, self.current_parameter)
                loss = classifier_loss + laplace_loss
            elif idx>1 and (self.cfg["train_mode"] == "ewc_diagonal"):
                laplace_loss = self.get_ewc_loss(self.model, self.current_fisher, self.map_parameters)
                loss = classifier_loss + laplace_loss
            elif idx==1 and self.cfg["train_mode"] == "si_diagonal":
                self.W = self.update_W(self.W)
                loss = classifier_loss
            elif idx>1 and self.cfg["train_mode"] == "si_diagonal":
                self.W = self.update_W(self.W)
                si_loss = self.get_si_loss(self.model, self.current_omega, self.current_parameter)
                loss = classifier_loss + si_loss
            else:
                loss = classifier_loss
            # loss = classifier_loss + laplace_loss
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
        # Store all the MAP parameters
        self.map_parameters.append(self.current_parameter)
        if self.current_fisher :
            for n, p in fisher.items():
                self.current_fisher[n] += fisher[n] * weight
        else:
            self.current_fisher = fisher
    
    def update_si_estimate(self, weight=1.0):
        self.current_omega, self.current_parameter = update_omega(model=self.model, 
                                                                cfg=self.cfg, 
                                                                prev_omega=self.current_omega, 
                                                                W=self.W, 
                                                                p_old=self.current_parameter)
        # if self.current_omega:
        #     for n, p in omega.items():
        #         self.current_omega[n] += omega[n] * weight
        # else:  
        #     self.current_omega = omega
            
    def train_all(self):
        for i in range(self.cfg['num_task']):
            self.train(self.train_datasets[i], self.cfg['epoch'], idx=i+1)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
            if  (self.cfg["train_mode"] == "online_laplace_diagonal"):
                self.update_laplace_estimate(self.train_datasets[i])
            elif (self.cfg["train_mode"] == "ewc_diagonal"):
                self.update_laplace_estimate(self.train_datasets[i])
            elif (self.cfg["train_mode"] == "si_diagonal"):
                self.update_si_estimate()
                
        torch.save(torch.tensor(self.validation_acc), f"./plots/{self.cfg['train_mode']}.pt")

    def train_cumulative(self):
        for i in range(self.cfg['num_task']):
            
            #Train on all the previous datasets
            temp_dataset = torch.utils.data.ConcatDataset(self.train_datasets[:i+1])
            self.train(temp_dataset, self.cfg['epoch'], idx=i+1)
            acc = self.evaluate(i+1)
            self.validation_acc.append(acc)
            print(f"Avg Test Accuracy: {acc: .3f}")
                
        torch.save(torch.tensor(self.validation_acc), f"./plots/{self.cfg['train_mode']}.pt")

if __name__ == '__main__':      
    
    cfg_ol = cnfgs[0]
    cfg_ewc = cnfgs[1]
    cfg_si = cnfgs[2]
    cfg_cum = cnfgs[3]
    cfg_norm = cnfgs[4]
    
    # Change this to select a specific configuration
    l = OnlineLearner(cfg_si)
    
    # Comment this when training on cumulative datasets, otherwise uncomment
    l.train_all()
    
    # Uncomment this to train on cumulative datasets
    # l.train_cumulative()