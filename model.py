import torch
from torch.nn import functional as F


class MLP(torch.nn.Module):

    def __init__(self, input_size=28*28, output_size=10):
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 10)
        
    def forward(self, x):
        out = self.l1(x.view(-1, 28*28))
        out = self.l2(out.relu())
        out = self.l3(out.relu())
        return out
   
# create a LeNet-like architecture with two convolutional layers with 5×5 convolutions with 
# 20 and 50 channels respectively and a fully connected hidden layer with 500 units. Use ReLU 
# nonlinearities and perform a 2×2 max-pooling operation after each convolutional layer with stride 2 
class CNN(torch.nn.Module):
    
        def __init__(self, input_size=32*32, output_size=10):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 20, kernel_size=5)
            self.conv2 = torch.nn.Conv2d(20, 50, kernel_size=5)
            self.fc1 = torch.nn.Linear(50*4*4, 500)
            self.fc2 = torch.nn.Linear(500, 10)
            
        def forward(self, x):
            out = self.conv1(x)
            out = F.max_pool2d(out, 2)
            out = F.relu(out)
            out = self.conv2(out)
            out = F.max_pool2d(out, 2)
            out = F.relu(out)
            out = out.view(-1, 50*4*4)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)
            return out
        
        # define a function to change the final layer of the model
def change_final_layer(model, output_size):
    model.fc = torch.nn.Linear(model.fc.in_features, output_size)
    return model

# define a function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def estimate_fisher(model, dataset, n_samples, device):
    est_fisher_info = {}
    parameters = {}
    for n, p in model.named_parameters():
        est_fisher_info[n] = p.detach().clone().zero_()

    model.eval()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for index,(x,y) in enumerate(data_loader):
        if n_samples is not None:
            if index > n_samples:
                break
        output = model(x.to(device))
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(retain_graph=True if (label_index+1)<output.shape[1] else False)
            for n, p in model.named_parameters():
                if p.grad is not None:
                    est_fisher_info[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}
    
    for n, p in model.named_parameters():
        parameters[n] =  p.detach().clone()
    return est_fisher_info, parameters

# Update omega for Synaptic Intelligence
def update_omega(model, cfg, prev_omega, W, p_old):
        '''After completing training on a task, update the per-parameter regularization strength.

        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        
        for n, p in model.named_parameters():

            # Find/calculate new values for quadratic penalty on parameters
            # p_prev = getattr(model, '{}_SI_prev_task'.format(n))
            p_prev = p_old[n]
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W[n]/(p_change**2 + cfg["epsilon"])
            
            omega_new = prev_omega[n] + omega_add

            # If requested, clamp the value of omega
            if cfg["omega_max"] is not None:
                omega_new = torch.clamp(omega_new, min=0, max=cfg["omega_max"])

            # Store these new values in the model
            # model.register_buffer('{}_SI_prev_task'.format(n), p_current)
            # model.register_buffer('{}_SI_omega'.format(n), omega_new)
            
        return omega_new, p_current





# def update_fisher(old_fisher, new_fisher, weight=1.0):
#     updated_fisher = {}
#     for n, p in old_fisher.items():
#         updated_fisher[n] = old_fisher[n] * weight + new_fisher[n]
#     return updated_fisher