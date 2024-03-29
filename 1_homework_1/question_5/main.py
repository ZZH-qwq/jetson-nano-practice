import os
import pickle as pkl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from zzhmodel import MODELRESH, MODELRESW, YourModel
import numpy as np
from dataset import Cifar100Dataset, COLOR, DATARESH, DATARESW, N_CLASS
from tqdm import tqdm

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
    best_acc = 0.
    # train loop
    for epoch_idx in range(num_epochs):
        model.train()
        running_acc = running_loss = total = 0
        tqdm_train_loader = tqdm(train_loader)
        for inputs, labels in tqdm_train_loader:

            # prepare mini-batch data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward path
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # print(outputs)
            # print(labels)

            # backward path 
            optimizer.zero_grad() # clear old gradients
            loss.backward() # calculate new gradients
            optimizer.step() # update weights

            running_loss += loss.item()
            running_acc += calc_acc(outputs, labels).item()
            total += 1

            loss_list.append(loss.item())

            tqdm_train_loader.set_postfix_str("Epoch: {e:d}, loss: {l:.4f}".format(e=epoch_idx,l=loss.item()))

            # exit()

        running_loss /= total
        running_acc /= total
        testing_acc = test(model,device,val_loader).item()
        if testing_acc > best_acc:
            torch.save(net.state_dict(),"./best.pth")
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
    batch_size = 16
    epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = torch.device("cuda:0")

    # dataset
    root="./data/cifar-100-python"
    train_set = Cifar100Dataset(root=root, is_train=True)
    val_set = Cifar100Dataset(root=root, is_train=False)
    train_loader = DataLoader(train_set,
                                shuffle=True,
                                batch_size=batch_size)
    val_loader = DataLoader(val_set,shuffle=False,batch_size=batch_size)

    # load pretrain weights
    net = YourModel(num_classes=N_CLASS)
    # net.load_state_dict(torch.load("./latest.pth"))
    # pre_trained_path = "./mobilenet_v2.pth"
    # pre_weights = torch.load(pre_trained_path, map_location=device)
    # delete classifier weights, preserve the initialized ones.
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(net,
            optimizer,
            criterion,
            epochs,
            device,
            train_loader,
            val_loader)