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
          verify_loader,
          val_loader=None):
    # model setting
    model.train()
    model.to(device=device)
    loss_list = []
    variance_list = []
    mean_list = []
    # train loop
    for epoch_idx in range(num_epochs):
        model.train()
        running_acc = running_loss = total = 0
        #for inputs, labels in tqdm(train_loader):
        for inputs, labels in train_loader:

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
            
            # calculate variance of loss every 1/16 of the training process
            if total % (len(train_loader) // 16) == 10 and len(variance_list) < 20:
                its = 0
                varify_list = []
                #for inputs, labels in tqdm(verify_loader):
                for inputs, labels in verify_loader:    
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    varify_list.append(loss.item())
                    its += 1
                    if (its >= 50):
                        loss_var = torch.var(torch.tensor(varify_list))
                        loss_mean = torch.mean(torch.tensor(varify_list))
                        variance_list.append(loss_var)
                        mean_list.append(loss_mean)
                        print("LossVar {0:.6f}, LostMean {1:.6f}".format(loss_var, loss_mean))
                        break

            # exit()

        # #############################
        # TODO: calculate mean and variance of loss here!!!!
        # #############################
        running_loss /= total
        running_acc /= total
        testing_acc = test(model,device,val_loader)
        print("Epoch {0:d}: TrainLoss {1:.6f}, TrainAcc {2:.4f}, TestAcc {3:.4f}".format(
            epoch_idx, running_loss, running_acc, testing_acc))
    return loss_list, variance_list, mean_list


def test(model:nn.Module,
         device,
         test_loader):
    model.eval()
    model.to(device=device)
    testing_acc = total = 0
    loss_list = []
    with torch.no_grad():
        #for inputs, labels in tqdm(test_loader):
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            testing_acc += calc_acc(outputs, labels)
            total += 1
        # calculate mean and variance of loss
        lost_var = torch.var(torch.tensor(loss_list))
        lost_mean = torch.mean(torch.tensor(loss_list))
        print("LossVar {0:.6f}, LostMean {1:.6f}".format(lost_var, lost_mean))
    return (testing_acc / total).item()

if __name__ == '__main__':
    # 0. hyper-parameters.
    batch_size_list = [8, 16, 32, 64, 128]
    # num_epochs_list = [1, 2, 4, 6, 8]
    num_epochs_list = [1, 4]
    batch_epoch_pairs = []
    learning_rate = 1e-3
    device = torch.device("cuda:0")
    data_root = './data/mnist'

    # 1. define dataset + dataloader.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    verify_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)

    # 2. traverse through batchsize and epoch number.
    sgd_loss_list = []
    sgd_variance_list = []
    sgd_mean_list = []
    for batch_size in batch_size_list:
        for num_epochs in num_epochs_list:
            if batch_size == 128 and num_epochs == 4:
                continue
            batch_epoch_pairs.append((batch_size, num_epochs))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            verify_loader = DataLoader(verify_dataset, batch_size=batch_size, shuffle=False)
            iteration_number = len(train_loader) * num_epochs / batch_size
            print("Batch Size: {}, Epoch Number: {}, Iteration Number: {}".format(batch_size, num_epochs, iteration_number))
            model = Network().to(device)
            # print(model)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            loss_list, variance_list, mean_list = train(
                model, 
                optimizer, 
                criterion, 
                num_epochs, 
                device, 
                train_loader, 
                verify_loader, 
                test_loader)
            # sgd_loss_list.append(loss_list)
            sgd_variance_list.append(variance_list)
            sgd_mean_list.append(mean_list)

    # 4. compare the data.
    import matplotlib.pyplot as plt

    os.makedirs("./pic", exist_ok=True)
    fig, ax = plt.subplots()
    for i, mean_list in enumerate(sgd_mean_list):
        x_ax = list(range(len(mean_list)))
        batch_size, num_epochs = batch_epoch_pairs[i]
        ax.plot(x_ax, mean_list, label="B: {}, E: {}".format(batch_size, num_epochs))
    ax.set_ylabel("mean of loss")
    ax.set_xlabel("iteration")
    ax.legend(loc='upper right')
    plt.savefig("./pic/hyper-parameters_mean.png")
    plt.close(fig)


    fig_var, ax_var = plt.subplots()
    for i, variance_list in enumerate(sgd_variance_list):
        x_ax_var = list(range(len(variance_list)))
        batch_size, num_epochs = batch_epoch_pairs[i]
        ax_var.plot(x_ax_var, variance_list, label="B: {}, E: {}".format(batch_size, num_epochs))
    ax_var.set_ylabel("variance of loss")
    ax_var.set_xlabel("iteration")
    ax_var.legend(loc='upper right')
    plt.savefig("./pic/hyper-parameters_variance.png")
    plt.close(fig_var)



