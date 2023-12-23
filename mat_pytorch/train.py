import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import os

from model.mat import MAT 
from data_loader import load_single

    
def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def train(model, train_loader, test_loader, save_path, learning_rate=0.001, weight_decay=0.0001, num_epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    checkpoint_filepath = os.path.join(save_path, 'checkpoint', 'model_bak.pth')
    
    best_accuracy = 0.0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        scheduler.step()
        losses_batch = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses_batch.append(loss.item())
            loss.backward()
            optimizer.step()
        losses.append([sum(losses_batch)])

        accuracy = test(model, test_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {sum(losses_batch):.4f}, Test Accuracy: {accuracy * 100:.3f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_filepath)

    print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")
    
    # 保存模型结构和参数
    torch.save(model, os.path.join(save_path, 'model/model.pth'))
    losses_np = np.array(losses)
    np.save('{0}/loss/losses.npy'.format(save_path), losses_np)

    return losses_np

def plot_training(loss):
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label='train_loss')
    plt.title('train_loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print(torch.__version__)
    dataset = 'CWRU'  # CWRU or motor

    if dataset == 'CWRU':
        snr = None
        work_condition = 'allHPallF'
        end = 'FE'  # FE(fan end), or DE(drive end)
        data_folder = r''  # 根据实际情况
        path_x_train = os.path.join(data_folder, snr + r'x_train ' + work_condition + r' ' + end + 'get.npy')
        path_y_train = os.path.join(data_folder, snr + r'y_train ' + work_condition + r' ' + end + 'get.npy')
        path_x_train_pos = os.path.join(data_folder, snr + r'x_train ' + work_condition + ' CF ' + end + 'get.npy')

        exper_save_folder = os.path.join('saved', dataset + '_' + work_condition + '_' + end + '_' + snr)
        if not os.path.exists(exper_save_folder):
            os.makedirs(exper_save_folder)
            os.makedirs(os.path.join(exper_save_folder, 'checkpoint'))
            os.makedirs(os.path.join(exper_save_folder, 'model'))
            os.makedirs(os.path.join(exper_save_folder, 'loss'))
            print(f"文件夹 '{exper_save_folder}' 不存在，已创建。")

    input_shape = (11, 1201, 1)
    num_classes = 27
    max_fre = 12000
    num_epochs = 300

    mat = MAT(input_shape, num_classes, max_fre)
    
    train_dataset, test_dataset = load_single(path_x_train, path_y_train, path_x_train_pos)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 训练
    loss = train(model=mat, train_loader=train_loader, test_loader=test_loader, save_path=exper_save_folder, num_epochs=num_epochs)
    plot_training(loss)

if __name__ == "__main__":
    main()
