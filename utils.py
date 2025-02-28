from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)  # 确保标签为整数
    return X, y

def preprocess_data(X, y):
    # 先划分数据集（使用原始标签进行分层）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 关键修复：使用原始标签 y，而非 One-hot 编码后的结果
    )

    # 归一化（基于训练集参数）
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # One-hot编码（划分后再转换）
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
        # 添加 Softmax 数值稳定性处理
        max_vals = np.max(predictions, axis=1, keepdims=True)
        stable_preds = predictions - max_vals
        softmax_probs = np.exp(stable_preds) / np.sum(np.exp(stable_preds), axis=1, keepdims=True)
        loss = -np.mean(targets * np.log(softmax_probs + 1e-15))
        return loss
    else:
        raise ValueError("Unsupported loss type")