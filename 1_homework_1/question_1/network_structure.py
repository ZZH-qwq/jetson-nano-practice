import torch, time
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
batch_size = 32
learning_rate = 0.001
num_epochs = 5
device = torch.device("cuda:0")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data/mnist', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        print(x.shape, "\n")
        for layer in self.layers:
            start = time.time()
            x = layer(x)
            end = time.time()
            print(x.shape, "{0:d} mu s\n".format(int((end-start)*1e6)), sep=" ")
        return x


model = Network().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def calc_acc(outputs, labels):
    return torch.mean((torch.argmax(outputs, dim=-1) == labels).float())

def train():
    for epoch_idx in range(num_epochs):
        model.train()
        running_acc = running_loss = total = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += calc_acc(outputs, labels).item()
            total += 1

            if total > 1:
                exit()


if __name__ == '__main__':
    beg_t = time.time()
    train()
    end_t = time.time()

    print("Elapsed Time Per Epoch: %.4f s" % ((end_t - beg_t) / num_epochs))

