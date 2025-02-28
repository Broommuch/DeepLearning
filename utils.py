from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)  # 确保标签为整数
    return X, y

def preprocess_data(X, y):
    # 划分数据集（先划分再归一化）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 确保分布一致
    )

    # 归一化：用训练集参数归一化所有数据
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # One-hot编码（确保标签在 0-9 范围内）
    if y_train.min() < 0 or y_train.max() > 9:
        raise ValueError("标签值超出预期范围 [0, 9]")
    y_train_onehot = np.eye(10)[y_train]
    y_val_onehot = np.eye(10)[y_val]

    return X_train_scaled, X_val_scaled, y_train_onehot, y_val_onehot