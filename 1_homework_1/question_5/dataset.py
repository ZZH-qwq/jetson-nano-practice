import os
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from zzhmodel import MODELRESH, MODELRESW
import numpy as np

COLOR=3
DATARESH=32
DATARESW=32
N_CLASS=100

# transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def preprocess(data, labels):
    # NCHW
    data = data.reshape(-1,COLOR,DATARESH,DATARESW)
    labels = torch.tensor(labels, dtype=int)
    return data, labels

def build_data(root="../data/cifar-100-python",is_train=True):
    train_data, train_labels, test_data, test_labels = None, None, None, None
    train_file = os.path.join(root, "train")
    test_file = os.path.join(root, "test")
    with open(train_file, 'rb') as fo:
        train_dict = pkl.load(fo, encoding='bytes')
    with open(test_file, 'rb') as fo:
        test_dict = pkl.load(fo, encoding='bytes')
    print(train_dict.keys())
    print(train_dict[b"fine_labels"].__len__())
    print(train_dict[b"data"].shape)
    print(test_dict[b"fine_labels"].__len__())
    print(test_dict[b"data"].shape)
    if is_train:
        train_data, train_labels = preprocess(train_dict[b"data"], train_dict[b"fine_labels"])
    else:
        test_data, test_labels = preprocess(test_dict[b"data"], test_dict[b"fine_labels"])
    return train_data, train_labels, test_data, test_labels

class Cifar100Dataset(Dataset):
    def __init__(self, root="../data/cifar-100-python", is_train=True):
        super().__init__()
        if is_train:
            self.data,self.labels,_,_ = build_data(root,is_train)
        else:
            _,_,self.data,self.labels = build_data(root,is_train)

    def __getitem__(self, idx):
        # return transform(torch.tensor(self.data[idx]).float()/255.), self.labels[idx]
        return torch.tensor(self.data[idx]).float()/255., self.labels[idx]


    def __len__(self):
        return len(self.labels)

if __name__=="__main__":
    root="../data/cifar-100-python"
    train_set = Cifar100Dataset(root=root, is_train=True)
    val_set = Cifar100Dataset(root=root, is_train=False)
    train_loader = DataLoader(train_set,shuffle=True,batch_size=32,pin_memory=True,num_workers=4)
    val_loader = DataLoader(val_set,shuffle=False,batch_size=32)