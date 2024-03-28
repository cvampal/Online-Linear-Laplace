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