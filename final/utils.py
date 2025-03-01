from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 数据预处理
def load_data():
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.astype(int).to_numpy()

    # 归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # One-hot编码
    onehot = OneHotEncoder(sparse_output=False)
    y = onehot.fit_transform(y.reshape(-1, 1))

    # 划分训练验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val