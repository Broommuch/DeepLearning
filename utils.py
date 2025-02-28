from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    return X, y


def preprocess_data(X, y):
    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot编码
    y_onehot = np.eye(10)[y.astype(int)]

    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_onehot, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val
