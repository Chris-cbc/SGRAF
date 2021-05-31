import torch
import torch.nn as nn
import torch.nn.init as init
from data_loader import train_loader, valid_loader, test_loader


# 模型定义及参数初始化
# 这里我们使用torch.nn中自带的实现.由于后续要定义的损失函数nn.nn.CrossEntropyLoss中包含了softmax的操作, 所以这里不再需要定义relu和softmax．

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super(Net, self).__init__()
        self.l1 = nn.Linear(num_inputs, num_hiddens)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(num_hiddens, num_outputs)

    def forward(self, X):
        X = X.view(X.shape[0], -1)
        o1 = self.relu1(self.l1(X))
        o2 = self.l2(o1)

        return o2

    def init_params(self):
        for param in self.parameters():
            # print(param.shape)
            init.normal_(param, mean=0, std=0.01)


num_inputs, num_outputs, num_hiddens = 28 * 28, 10, 256
net = Net(num_inputs, num_outputs, num_hiddens)
net.init_params()
# 定义损失函数
loss = nn.CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)


# 训练模型


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


num_epochs = 5


def train():
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_loader:
            y_hat = net(X)  # 前向传播
            l = loss(y_hat, y).sum()  # 计算loss
            l.backward()  # 反向传播

            optimizer.step()  # 参数更新
            optimizer.zero_grad()  # 清空梯度

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_loader, net)
        print('epoch %d, loss %.4f, train_acc %.3f，test_acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == "__main__":
    train()
