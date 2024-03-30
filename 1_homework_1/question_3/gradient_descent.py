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

class Nesterov(object):
    def __init__(self, parameters, lr=1e-3) -> None:
        self.counter=2 # k
        self.lr = lr
        self.parameters=list(parameters) # W
        self.buffer1=[] # V_{k-1}
        self.buffer2=[] # V_{k-2}
        for param in self.parameters:
            tensor1 = torch.zeros_like(param.data)
            tensor2 = torch.zeros_like(param.data)
            tensor1[:] = param.data
            tensor2[:] = param.data
            self.buffer1.append(tensor1)
            self.buffer2.append(tensor2)
            self.buffer1[-1]._require_grad=False
            self.buffer2[-1]._require_grad=False
    
    def zero_grad(self):
        for param in self.parameters:
            param._grad = None
        return
    
    def step(self):
        for param, param_m1, param_m2 in zip(self.parameters, self.buffer1, self.buffer2):
            # Example: SGD updates
            # param.data[:] = param.data - self.lr * param.grad
            # ##############################
            # TODO: Your code here!
            
            # Nesterov updates
            param_m1.data[:] = param.data - self.lr * param.grad
            param.data[:] = param_m1.data + (self.counter-1)/(self.counter+2)*(param_m1.data-param_m2.data)
            param_m2.data[:] = param_m1.data

            # ###############################
            None
        self.counter += 1
        return

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
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 5
    device = torch.device("cuda:0")
    data_root = './data/mnist'

    # 1. define dataset + dataloader.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. define network structure.
    model = Network().to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    sgd_loss_list=train(model,
          optimizer,
          criterion,
          1,
          device,
          train_loader,
          test_loader)

    # 3. nesterov
    model = Network().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Nesterov(model.parameters(), lr=learning_rate)
    nesterov_loss_list = train(model,
          optimizer,
          criterion,
          1,
          device,
          train_loader,
          test_loader)

    # 4. compare the data.
    os.makedirs("./pic", exist_ok=True)
    x_ax = list(range(len(sgd_loss_list)))
    fig, ax = plt.subplots()
    ax.plot(x_ax, sgd_loss_list, label="sgd")
    ax.plot(x_ax, nesterov_loss_list, label="nesterov")
    ax.set_ylabel("train loss")
    ax.set_xlabel("iteration")
    ax.legend(loc='upper right')
    plt.savefig("./pic/optimizer.png")