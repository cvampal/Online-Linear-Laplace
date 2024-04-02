import matplotlib.pyplot as plt
import torch

x1 = torch.load("plots/normal.pt")
x2 = torch.load("plots/ewc_diagonal.pt")
x3 = torch.load("plots/cumulative.pt")
x4 = torch.load("plots/online_laplace_diagonal.pt")
# x5 = torch.load("plots/si_diagonal.pt")


plt.plot(x1, label="Normal Training", linestyle='dashed')
plt.plot(x2, label="EWC Diagonal", linestyle='dashed')
plt.plot(x3, label="Cumulative Training", linestyle='dotted', color='black')
plt.plot(x4, label="Online Laplace Diagonal", linestyle='dashed')
# plt.plot(x5, label="SI Diagonal", linestyle='dashed')
plt.xlabel(" No of tasks")
plt.ylabel("Avg test accuracy")
plt.legend()
plt.savefig("plots/results.png")
