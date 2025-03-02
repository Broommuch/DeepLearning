import numpy as np


# 神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation='relu', alpha=0.01,
                 l2_lambda=1e-4, learning_rate=0.01, batch_size=64):
        # 网络参数
        self.layer_dims = [input_size] + hidden_sizes + [output_size]
        self.activation = activation.lower()
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 初始化参数
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            if self.activation in ['relu', 'leaky_relu']:
                scale = np.sqrt(2.0 / self.layer_dims[i])
            else:
                scale = np.sqrt(1.0 / self.layer_dims[i])

            self.weights.append(np.random.randn(self.layer_dims[i], self.layer_dims[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_dims[i + 1])))

        self.caches = []

    # 激活函数
    def _activate(self, Z):
        if self.activation == 'relu':
            return np.maximum(0, Z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation == 'tanh':
            return np.tanh(Z)
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, Z, self.alpha * Z)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    # 导数函数
    def _activate_deriv(self, Z):
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(Z) ** 2
        elif self.activation == 'leaky_relu':
            return np.where(Z > 0, 1, self.alpha)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.caches = []
        A = X

        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.caches.append((A.copy(), Z.copy()))
            A = self._activate(Z)

        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.caches.append((A.copy(), Z.copy()))
        return self.softmax(Z)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        corect_logprobs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m

        reg_loss = 0.5 * self.l2_lambda * sum(np.sum(W ** 2) for W in self.weights)
        return data_loss + reg_loss

    def backward(self, X, y):
        m = X.shape[0]
        grads = []

        dZ = self.forward(X) - y

        for i in reversed(range(len(self.weights))):
            A_prev, _ = self.caches[i]

            dW = np.dot(A_prev.T, dZ) / m + self.l2_lambda * self.weights[i]
            db = np.sum(dZ, axis=0, keepdims=True) / m
            grads.insert(0, (dW, db))

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                _, Z_prev = self.caches[i - 1]
                dZ = dA_prev * self._activate_deriv(Z_prev)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[i][0]
            self.biases[i] -= self.learning_rate * grads[i][1]