import numpy as np


class Layer:
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
        elif name == 'softmax':
            def softmax(x):
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)

            return softmax
        else:
            raise ValueError("Invalid activation function")

    def get_activation_deriv(self, name):
        if name == 'relu':
            return lambda x: np.where(x > 0, 1, 0)
        elif name == 'sigmoid':
            return lambda x: x * (1 - x)
        elif name == 'tanh':
            return lambda x: 1 - np.square(x)
        elif name == 'softmax':
            return lambda x: 1  # 占位符
        else:
            raise ValueError("Invalid activation derivative")

    def forward(self, x):
        self.z = np.dot(x, self.W) + self.B
        self.a = self.activation(self.z)
        return self.a

    def backward(self, x, grad):
        # 计算激活函数导数（基于激活后的值 a）
        d_z = grad * self.activation_deriv(self.a)

        # 参数梯度（含L2正则化）
        dW = np.dot(x.T, d_z) + self.l2_lambda * self.W
        dB = np.sum(d_z, axis=0, keepdims=True)

        # 输入梯度
        dx = np.dot(d_z, self.W.T)
        return dx, dW, dB


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        self.layer_outputs = [x]  # 保存各层输入
        for layer in self.layers:
            x = layer.forward(x)
            self.layer_outputs.append(x)
        return x

    def backward(self, x, y_true):
        # 前向传播以记录各层输入
        predictions = self.forward(x)

        # 计算初始梯度（交叉熵损失 + softmax）
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        grad = (predictions - y_true) / y_true.shape[0]

        grads = []
        # 反向遍历各层及其输入
        for layer, layer_input in zip(reversed(self.layers), reversed(self.layer_outputs[:-1])):
            grad, dw, db = layer.backward(layer_input, grad)
            grads.append((dw, db))

        return grads