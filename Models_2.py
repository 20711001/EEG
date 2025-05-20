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
    原本用于ve_cnn_64*80模型的构建,但时间过长,没有运行,对其能否运行存疑
"""


class VeCNNNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=109):
        super(VeCNNNet, self).__init__()

        # 第一卷积层：输入通道1，输出32，核(3,5)，步长(1,2)，填充(1,2)
        self.conv1 = nn.Conv2d(input_channels, 32, (3, 5), (1, 2), (1, 2))
        self.pool1 = nn.MaxPool2d((1, 2), (1, 2))  # 最大池化

        # 第二卷积层：保持32通道
        self.conv2 = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1))
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2))

        # 第三卷积层：扩展至64通道
        self.conv3 = nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1))
        self.pool3 = nn.MaxPool2d((2, 2), (2, 2))

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            # 修改此处输入尺寸为 64x80
            dummy_input = torch.randn(1, input_channels, 64, 80)
            dummy_output = self._forward_features(dummy_input)
            fc_input_features = dummy_output.numel() // dummy_output.size(0)

        # 全连接层结构
        self.fc1 = nn.Linear(fc_input_features, 512)
        self.dropout = nn.Dropout(0.5)  # 50%的dropout防止过拟合
        self.fc2 = nn.Linear(512, num_classes)  # 输出分类结果

    def _forward_features(self, x):
        """特征提取部分"""
        x = F.elu(self.conv1(x))  # ELU激活函数
        x = self.pool1(x)
        x = F.elu(self.conv2(x))
        x = self.pool2(x)
        x = F.elu(self.conv3(x))
        x = self.pool3(x)
        return x

    def forward(self, x):
        # 完整前向传播流程
        x = self._forward_features(x)  # 特征提取
        x = torch.flatten(x, 1)  # 展平特征图
        x = F.elu(self.fc1(x))  # 全连接层1
        x = self.dropout(x)  # 随机失活
        x = self.fc2(x)  # 输出层
        return x

    def extract_features(self, x):
        """提取512维特征向量"""
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))  # 输出shape: [batch, 512]
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
        # 转换数据为PyTorch张量并调整维度顺序 (H,W,C) -> (C,H,W)
        x = torch.from_numpy(self.data[idx].astype(np.float32)).permute(2, 0, 1)
        # 将标签转换为长整型张量
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y  # 返回单个样本和标签


def train_model():
    # 1. 数据准备
    data_dir = "64_channels_seg"
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    # 加载所有NPZ文件数据
    X, y = [], []
    for file in npz_files:
        data = np.load(os.path.join(data_dir, file))
        for key in data.files:
            X.append(data[key][..., np.newaxis])  # 添加通道维度
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
    model = VeCNNNet(input_channels=1, num_classes=len(np.unique(y))).to(device)

    # 4. 定义优化目标
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    # 5. 训练循环
    for epoch in range(20):
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
        print(f"Epoch {epoch + 1}/20")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | "
              f"Train Acc: {100 * train_acc / len(train_loader.dataset):.2f}%")
        print(f"Test Loss: {test_loss / len(test_loader):.4f} | "
              f"Test Acc: {100 * test_acc / len(test_loader.dataset):.2f}%")
        print("-" * 50)

    # 6. 保存模型
    torch.save(model.state_dict(), "models/ve_cnn_64x80.pth")
    print("训练完成，模型已保存")




if __name__ == "__main__":
    train_model()