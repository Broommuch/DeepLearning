import numpy as np


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