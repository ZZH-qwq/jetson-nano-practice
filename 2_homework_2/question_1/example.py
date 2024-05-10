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
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub, QConfig, HistogramObserver, PerChannelMinMaxObserver
import argparse


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
            optimizer.zero_grad()  # clear old gradients
            loss.backward()  # calculate new gradients
            optimizer.step()  # update weights

            running_loss += loss.item()
            running_acc += calc_acc(outputs, labels).item()
            total += 1

            loss_list.append(loss.item())

            tqdm_train_loader.set_postfix_str(
                "Epoch: {e:d}, loss: {l:.4f}".format(e=epoch_idx, l=loss.item()))

            # exit()

        running_loss /= total
        running_acc /= total
        testing_acc = test(model, device, val_loader).item()
        if testing_acc > best_acc:
            torch.save(net.state_dict(), "./best.pth")
        print("Epoch {0:d}: TrainLoss {1:.6f}, TrainAcc {2:.4f}, TestAcc {3:.4f}".format(
            epoch_idx, running_loss, running_acc, testing_acc))
    return loss_list


def test(model: nn.Module,
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


def DynamicQuantize(model):
    print("Size of model before quantization")
    print_size_of_model(model)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    print(quantized_model)
    print("Size of model before quantization")
    print_size_of_model(quantized_model)
    return quantized_model


class PTQModel(nn.Module):
    def __init__(self, fp_model) -> None:
        super().__init__()
        self.fp_model = fp_model
        if isinstance(self.fp_model.layers[-1], nn.Softmax):
            # pop the layers that can't be quantized!!
            self.fp_model.layers.pop(-1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fp_model(x)
        x = self.dequant(x)
        return x


def PTQ(model,
        train_loader,
        test_loader=None):
    # test the ptq model again!
    acc = test(model, "cpu:0", test_loader)
    print("Float accurracy: {acc:.4f}".format(acc=acc.item()))

    print("Size of model before quantization")
    print_size_of_model(model)

    # fuse layers and get quantize config.
    print(model)
    model.fuse_model()
    print(model)
    ptq_model = PTQModel(model)
    qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.quint8, reduce_range=True),
                      weight=PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8, qscheme=torch.per_channel_symmetric
    ))
    # qconfig = torch.ao.quantization.get_default_qconfig('x86')
    ptq_model.qconfig = qconfig
    torch.quantization.prepare(ptq_model, inplace=True)
    print(ptq_model.qconfig)

    # Calibration
    testing_acc = total = 0
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(train_loader)):
            ptq_model(inputs)
            if idx == 500:
                break

    # Convert to quantized model
    torch.quantization.convert(ptq_model, inplace=True)
    print(ptq_model)

    # calculate size
    print("Size of model after quantization")
    print_size_of_model(ptq_model)

    # test the ptq model again!
    acc = test(ptq_model, "cpu:0", test_loader)
    print("PTQ accurracy: {acc:.4f}".format(acc=acc.item()))

    for input_tensor, _ in test_loader:
        break

    torch.onnx.export(
        ptq_model,
        {"x": input_tensor},
        "quant.onnx",
        input_names=["img"],
        output_names=["class"],
    )


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    model_size = os.path.getsize("temp.p")/1e6
    print('Size (MB):', model_size)
    os.remove('temp.p')
    return model_size


if __name__ == '__main__':
    batch_size = 64
    epochs = 5
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = torch.device("cuda:0")

    parser = argparse.ArgumentParser()
    parser.add_argument("--quantizeType", type=str, default="ptq")
    args = parser.parse_args()
    quantizeType = args.quantizeType

    # dataset
    root = "./data/cifar-100-python"
    train_set = Cifar100Dataset(root=root, is_train=True)
    val_set = Cifar100Dataset(root=root, is_train=False)
    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=batch_size)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size)

    # load pretrain weights
    net = YourModel(num_classes=N_CLASS)

    if quantizeType != "":
        net.load_state_dict(torch.load("best.pth", map_location="cuda:0"))
        torch.backends.quantized.engine = 'x86'
    net.to(device)

    if quantizeType == "":
        # train
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train(net,
              optimizer,
              criterion,
              epochs,
              device,
              train_loader,
              val_loader)

    # dynamic quantize
    if quantizeType == "dynamic":
        DynamicQuantize(net)
    elif quantizeType == "ptq":
        PTQ(net, train_loader, val_loader)
