import matplotlib.pyplot as plt
import torch

x1 = torch.load("plots/normal.pt")
x2 = torch.load("plots/online_laplace_diagonal.pt")

plt.plot(x1, label="Normal Training")
plt.plot(x2, label="Online Diagonal Laplace")
plt.xlabel(" No of tasks")
plt.ylabel("Avg test accuracy")
plt.legend()
plt.savefig("plots/results.png")
