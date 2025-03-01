import numpy as np
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


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size,
                 activation='relu', l2_lambda=1e-4):
        # 初始化参数（双层结构）
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.activation = activation
        self.l2_lambda = l2_lambda

    # 激活函数全家桶
    def _activation(self, Z, type):
        if type == 'relu':
            return np.maximum(0, Z)
        elif type == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif type == 'tanh':
            return np.tanh(Z)
        return Z

    def _activation_deriv(self, Z, type):
        if type == 'relu':
            return (Z > 0).astype(float)
        elif type == 'sigmoid':
            s = self._activation(Z, 'sigmoid')
            return s * (1 - s)
        elif type == 'tanh':
            return 1 - np.tanh(Z) ** 2
        return np.ones_like(Z)

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        # 第一层
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._activation(self.Z1, self.activation)

        # 输出层
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.softmax(self.Z2)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        corect_logprobs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss

    def backward(self, X, y, lr):
        m = X.shape[0]

        # 输出层梯度
        dZ2 = self.softmax(self.Z2) - y
        dW2 = np.dot(self.A1.T, dZ2) / m + self.l2_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._activation_deriv(self.Z1, self.activation)
        dW1 = np.dot(X.T, dZ1) / m + self.l2_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 参数更新
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


# 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.01, batch_size=64):
    for epoch in range(epochs):
        # Mini-batch训练
        permutation = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # 前向传播
            output = model.forward(X_batch)

            # 反向传播
            model.backward(X_batch, y_batch, lr)

        # 验证集评估
        val_output = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
        loss = model.compute_loss(val_output, y_val)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}")


# 主程序
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    # 创建双层网络（784 → 256 → 10），可自由选择激活函数
    model = NeuralNetwork(
        input_size=784,
        hidden_size=256,
        output_size=10,
        activation='relu'  # 可替换为sigmoid或tanh
    )

    # 训练参数
    train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        lr=0.01,
        batch_size=64
    )

    # 最终验证集准确率
    final_output = model.forward(X_val)
    final_acc = np.mean(np.argmax(final_output, axis=1) == np.argmax(y_val, axis=1))
    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")