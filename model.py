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

# def update_fisher(old_fisher, new_fisher, weight=1.0):
#     updated_fisher = {}
#     for n, p in old_fisher.items():
#         updated_fisher[n] = old_fisher[n] * weight + new_fisher[n]
#     return updated_fisher