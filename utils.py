from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)  # 确保标签为整数
    return X, y


def preprocess_data(X, y):
    # 先划分数据集，再归一化
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 保持类别分布
    )

    # 归一化（基于训练集参数）
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # One-hot编码
    if y_train.min() < 0 or y_train.max() > 9:
        raise ValueError("标签值超出范围 [0, 9]")
    y_train_onehot = np.eye(10)[y_train]
    y_val_onehot = np.eye(10)[y_val]

    return X_train_scaled, X_val_scaled, y_train_onehot, y_val_onehot


def compute_loss(predictions, targets, loss_type='cross_entropy', regularization=None, lambda_=0.01):
    """
    计算损失值（已适配你的模型输出）

    参数:
        predictions: 模型输出（未经 softmax 的原始值）
        targets: One-hot 编码的真实标签
        loss_type: 损失类型，支持 'cross_entropy' 或 'mse'
    """
    if loss_type == 'cross_entropy':
        # 交叉熵损失（已适配你的模型输出为 logits）
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(targets * np.log(predictions))

    elif loss_type == 'mse':
        # 均方误差
        loss = np.mean((predictions - targets) ** 2)

    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

    # 正则化（此处假设模型权重在反向传播中处理，故省略）
    return loss