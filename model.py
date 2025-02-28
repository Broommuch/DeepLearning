import numpy as np

class Layer:
    """基础层抽象类"""
    def forward(self, x):
        raise NotImplementedError

    def backward(self, x, grad):
        raise NotImplementedError

class FullyConnectedLayer(Layer):
    def __init__(self, n_in, n_out, activation='relu', l2_lambda=0):
        self.n_in = n_in
        self.n_out = n_out
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)  # He初始化
        self.B = np.zeros((1, n_out))
        self.activation = self.get_activation(activation)
        self.activation_deriv = self.get_activation_deriv(activation)
        self.l2_lambda = l2_lambda

    def get_activation(self, name):
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'tanh':
            return np.tanh
        else:
            raise ValueError("Invalid activation function")

    def get_activation_deriv(self, name):
        if name == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'tanh':
            return lambda x: 1 - np.square(x)
        else:
            raise ValueError("Invalid activation derivative")

    def forward(self, x):
        self.z = np.dot(x, self.W) + self.B
        self.a = self.activation(self.z)
        return self.a

    def backward(self, x, grad):
        d_z = grad * self.activation_deriv(self.a)  # 使用 self.a 而非 self.z
        dW = np.dot(x.T, d_z) + self.l2_lambda * self.W  # 修正正则化系数
        dB = np.sum(d_z, axis=0, keepdims=True)
        dx = np.dot(d_z, self.W.T)  # 修正维度
        return dx, dW, dB

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y_true):
        # 计算初始梯度（假设使用交叉熵损失）
        predictions = self.forward(x)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        grad = (predictions - y_true) / y_true.shape[0]

        grads = []
        for layer in reversed(self.layers):
            grad, dw, db = layer.backward(grad)
            grads.append((dw, db))
        return grads