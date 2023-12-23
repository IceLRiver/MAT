import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def load_single(path_x_train, path_y_train, path_x_train_pos):
    # 加载数据集
    x_train = np.load(path_x_train)
    y_train = np.load(path_y_train)
    x_train_pos = np.load(path_x_train_pos)
    x_train = np.concatenate((x_train, x_train_pos), axis=1)
    y_train = np.squeeze(y_train)
    print(x_train.shape, y_train.shape)
    x_train = x_train.transpose(2, 0, 1)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_train, y_train = shuffle(x_train, y_train, random_state=2)
    print(x_train.shape, y_train.shape)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # 查看数据集参数

    train_dataset = CustomDataset(x_train, y_train)
    test_dataset = CustomDataset(x_test, y_test)

    return train_dataset, test_dataset