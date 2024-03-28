import torch, time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []
        self.layers.append(nn.Conv2d(1, 10, kernel_size=5))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers.append(nn.Conv2d(10, 20, kernel_size=5))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(320, 50))
        self.layers.append(nn.Linear(50, 10))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def calc_acc(outputs,
             labels):
    return torch.mean((torch.argmax(outputs, dim=-1) == labels).float())

def train(model: nn.Module,
          optimizer,
          criterion,
          num_epochs,
          device,
          train_loader,
          val_loader=None):
    # model setting
    model.train()
    model.to(device=device)
    loss_list = []
    # train loop
    for epoch_idx in range(num_epochs):
        model.train()
        running_acc = running_loss = total = 0
        for inputs, labels in tqdm(train_loader):

            # prepare mini-batch data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward path
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward path 
            optimizer.zero_grad() # clear old gradients
            loss.backward() # calculate new gradients
            optimizer.step() # update weights

            running_loss += loss.item()
            running_acc += calc_acc(outputs, labels).item()
            total += 1

            loss_list.append(loss.item())

            # exit()

        # #############################
        # TODO: calculate mean and variance of loss here!!!!
        # #############################
        loss_mean = torch.mean(torch.tensor(loss_list))
        loss_variance = torch.var(torch.tensor(loss_list))
        print("Mean Loss: {:.6f}, Variance Loss: {:.6f}".format(loss_mean, loss_variance))
        running_loss /= total
        running_acc /= total
        testing_acc = test(model,device,val_loader).item()
        print("Epoch {0:d}: TrainLoss {1:.6f}, TrainAcc {2:.4f}, TestAcc {3:.4f}".format(
            epoch_idx, running_loss, running_acc, testing_acc))
    return loss_list


def test(model:nn.Module,
         device,
         test_loader):
    model.eval()
    model.to(device=device)
    testing_acc = total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            testing_acc += calc_acc(outputs, labels)
            total += 1
    return testing_acc / total

if __name__ == '__main__':
    # 0. hyper-parameters.
    batch_size_list = [8, 16, 32, 64, 128]
    # num_epochs_list = [1, 2, 4, 6, 8]
    num_epochs_list = [1]
    learning_rate = 1e-3
    device = torch.device("cuda:0")
    data_root = './data/mnist'

    # 1. define dataset + dataloader.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)

    # 2. traverse through batchsize and epoch number.
    sgd_loss_list = []
    for batch_size in batch_size_list:
        for num_epochs in num_epochs_list:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            iteration_number = len(train_loader) * num_epochs / batch_size
            print("Batch Size: {}, Epoch Number: {}, Iteration Number: {}".format(batch_size, num_epochs, iteration_number))
            model = Network().to(device)
            # print(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            sgd_loss_list.append(train(model,
                optimizer,
                criterion,
                num_epochs,
                device,
                train_loader,
                test_loader))

    # 4. compare the data.
    import matplotlib.pyplot as plt

    os.makedirs("./pic", exist_ok=True)
    fig, ax = plt.subplots()
    for i, loss_list in enumerate(sgd_loss_list):
        x_ax = list(range(len(loss_list)))
        ax.plot(x_ax, loss_list, label="B: {}, E: {}".format(batch_size_list[i//len(num_epochs_list)], num_epochs_list[i%len(num_epochs_list)]))
    ax.set_ylabel("train loss")
    ax.set_xlabel("iteration")
    ax.legend(loc='upper right')
    plt.savefig("./pic/hyper-parameters.png")
    plt.close(fig)
