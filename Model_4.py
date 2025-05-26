import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

"""
    生成DCNN模型的代码,时间很长,需6~7小时,谨慎运行;需运行将主函数取消注释,并准备好数据文件
    
    对DCNN模型进行三元损失函数优化,并增加参数显示,如召回率等构建混淆矩阵
"""


class DCNN(nn.Module):
    def __init__(self, num_classes, use_lstm=True, embedding_size=64):
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
        self.fc = nn.Linear(64, embedding_size)

    def forward(self, x, return_embedding=False):
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

        # 生成嵌入向量
        embedding = F.normalize(self.fc(x), p=2, dim=1)

        if return_embedding:
            return embedding

        if self.classifier is not None:
            return F.softmax(self.classifier(embedding), dim=1)

        return embedding

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


class TripletEEGDataset(Dataset):
    def __init__(self, data, labels):
        """初始化数据集
        Args:
            data: numpy数组, 形状为(n_samples, height, width, channels)
            labels: numpy数组, 包含样本标签
        """
        self.data = data  # 存储信号数据
        self.labels = labels  # 存储对应标签
        self.label_to_indices = defaultdict(list)

        # 用于后续高效生成训练三元组
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    def __len__(self):
        return len(self.data)  # 返回数据集样本总数

    def __getitem__(self, index):
        # 根据索引找到某样本的正,负样本.返回的是本身样本,正样本和负样本
        anchor_data = self.data[index]
        anchor_label = self.labels[index]

        # 随机选取同label的正样本
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive_data = self.data[positive_index]

        # 随机选取不同label的负样本
        negative_label = np.random.choice(list(set(self.labels) - {anchor_label}))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative_data = self.data[negative_index]

        return (
            torch.from_numpy(anchor_data.astype(np.float32)),
            torch.from_numpy(positive_data.astype(np.float32)),
            torch.from_numpy(negative_data.astype(np.float32))
        )


# 计算三元损失模型的等错误率EER(EER是错误接受率和错误拒绝率相等时的错误率)
# 错误接受率:非目标用户被误认为目标用户的概率
# 错误拒绝率:目标用户被系统错误拒绝的概率
def compute_eer(model, dataloader, device):
    distances, labels = [], []

    with torch.no_grad():
        for anchor, pos, neg in dataloader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            # 获取嵌入向量
            anchor_emb = model(anchor, return_embedding=True)
            pos_emb = model(pos, return_embedding=True)
            neg_emb = model(neg, return_embedding=True)

            # 计算距离
            pos_dist = F.pairwise_distance(anchor_emb, pos_emb)
            neg_dist = F.pairwise_distance(anchor_emb, neg_emb)

            distances.extend(pos_dist.cpu().tolist() + neg_dist.cpu().tolist())
            labels.extend([1] * len(pos_dist) + [0] * len(neg_dist))

    # 计算EER
    fpr, tpr, _ = roc_curve(labels, distances)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


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

    # 创建三元组数据集
    train_dataset = TripletEEGDataset(X_train, y_train)
    test_dataset = TripletEEGDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCNN(num_classes=0, embedding_size=64).to(device)  # num_classes=0表示只输出嵌入

    # 4. 定义优化目标
    criterion = TripletMarginLoss(margin=1.0, p=2)  # margin=1.0, 欧氏距离
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 5. 训练循环
    for epoch in range(200):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        # 训练批次
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()

            # 获取嵌入向量
            anchor_emb = model(anchor, return_embedding=True)
            pos_emb = model(pos, return_embedding=True)
            neg_emb = model(neg, return_embedding=True)

            # 计算三重损失
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证批次
        model.eval()
        eer = compute_eer(model, test_loader, device)

        # 打印统计信息
        print(f"Epoch {epoch + 1}/200")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Test EER: {eer:.2%}")
        print("-" * 50)

    # 6. 保存模型
    torch.save(model.state_dict(), "models/DCNN_16x80_sec.pth")
    print("训练完成，模型已保存")


if __name__ == "__main__":
    train_model()
