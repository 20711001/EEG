import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

"""
    生成DCNN模型的代码,时间很长,需6~7小时,谨慎运行;需运行将主函数取消注释,并准备好数据文件
"""

class DCNN(nn.Module):
    def __init__(self, num_classes, use_lstm=True):
        super(DCNN, self).__init__()

        # 卷积层部分
        self.conv1 = nn.Conv1d(16, 1024, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(1024)

        self.conv2 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(512)

        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)

        self.conv5 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.3)

        # 循环层部分
        self.rnn_type = 'LSTM' if use_lstm else 'GRU'
        if use_lstm:
            self.rnn1 = nn.LSTM(64, 64, batch_first=True)
            self.rnn2 = nn.LSTM(64, 64, batch_first=True)
        else:
            self.rnn1 = nn.GRU(64, 64, batch_first=True)
            self.rnn2 = nn.GRU(64, 64, batch_first=True)

        self.dropout2 = nn.Dropout(0.3)

        # 全连接层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 输入形状: (batch_size, 16, 80)

        # 卷积层
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.dropout1(x)

        # 调整维度以适应RNN (batch_size, seq_len, features)
        x = x.transpose(1, 2)

        # 循环层
        if self.rnn_type == 'LSTM':
            x, (h_n, c_n) = self.rnn1(x)
            x, (h_n, c_n) = self.rnn2(x)
        else:
            x, h_n = self.rnn1(x)
            x, h_n = self.rnn2(x)

        x = self.dropout2(x)

        # 只取最后一个时间步的输出
        x = x[:, -1, :]

        # 全连接层
        x = self.fc(x)

        return F.softmax(x, dim=1)



    def extract_features(self, x):
        """
        提取中间层特征
        :param x: 输入数据 (batch_size, 16, 80)
        :return: RNN的特征
        """
        # 卷积层特征
        conv_features = []
        x = F.relu(self.bn1(self.conv1(x)))
        conv_features.append(x)  # conv1 的输出

        x = F.relu(self.bn2(self.conv2(x)))
        conv_features.append(x)  # conv2 的输出

        x = F.relu(self.bn3(self.conv3(x)))
        conv_features.append(x)  # conv3 的输出

        x = F.relu(self.bn4(self.conv4(x)))
        conv_features.append(x)  # conv4 的输出

        x = F.relu(self.conv5(x))
        conv_features.append(x)  # conv5 的输出

        x = self.dropout1(x)

        # 调整维度以适应RNN
        x = x.transpose(1, 2)

        # 循环层
        if self.rnn_type == 'LSTM':
            x, (h_n, c_n) = self.rnn1(x)
            x, (h_n, c_n) = self.rnn2(x)
        else:
            x, h_n = self.rnn1(x)
            x, h_n = self.rnn2(x)

        # 只取最后一个时间步的输出
        x = x[:, -1, :]  # (batch_size, 64)

        return x



class SignalDataset(Dataset):
    def __init__(self, data, labels):
        """初始化数据集
        Args:
            data: numpy数组, 形状为(n_samples, height, width, channels)
            labels: numpy数组, 包含样本标签
        """
        self.data = data  # 存储信号数据
        self.labels = labels  # 存储对应标签

    def __len__(self):
        return len(self.data)  # 返回数据集样本总数

    def __getitem__(self, idx):

        x = torch.from_numpy(self.data[idx].astype(np.float32))
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y  # 返回单个样本和标签

def train_model():
    # 1. 数据准备
    data_dir = "16_channels_seg"
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    # 加载所有NPZ文件数据
    X, y = [], []
    for file in npz_files:
        data = np.load(os.path.join(data_dir, file))
        for key in data.files:
            X.append(data[key])  # 添加通道维度
            y.append(file.split('S')[1].split('R')[0])  # 从文件名提取标签
        data.close()

    X = np.array(X)  # 转换为numpy数组
    y = LabelEncoder().fit_transform(y)  # 标签编码为数字

    # 2. 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 创建PyTorch数据集和数据加载器
    train_loader = DataLoader(
        SignalDataset(X_train, y_train),
        batch_size=32,
        shuffle=True # 随机打乱数据,训练集数据打乱防止模型学习到不相干的时间序列关系
    )
    test_loader = DataLoader(
        SignalDataset(X_test, y_test),
        batch_size=32,
        shuffle=False
    )
    # 3. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNN(num_classes=len(np.unique(y))).to(device)

    # 4. 定义优化目标
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 5. 训练循环
    for epoch in range(200):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        # 训练批次
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        # 验证批次
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                test_acc += (outputs.argmax(1) == labels).sum().item()

        # 打印统计信息
        print(f"Epoch {epoch + 1}/200")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {100 * train_acc / len(train_loader.dataset):.2f}%")
        print(f"Test Loss: {test_loss / len(test_loader):.4f} | "
              f"Test Acc: {100 * test_acc / len(test_loader.dataset):.2f}%")
        print("-" * 50)

    # 6. 保存模型
    torch.save(model.state_dict(), "models/DCNN_16x80_sec.pth")
    print("训练完成，模型已保存")



if __name__ == "__main__":
    # train_model()