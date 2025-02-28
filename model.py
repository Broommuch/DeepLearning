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
        # 权重形状 (n_in, n_out)
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        self.B = np.zeros((1, n_out))  # 偏置形状 (1, n_out)
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
            # Softmax 的导数在交叉熵损失中已简化为 (predictions - targets)
            # 此处返回 1，表示梯度直接传递（实际计算中已通过损失函数处理）
            return lambda x: 1
        else:
            raise ValueError("Invalid activation derivative")

    def forward(self, x):
        # 矩阵乘法 x.dot(W) 形状应为 (batch_size, n_out)
        self.z = np.dot(x, self.W) + self.B
        self.a = self.activation(self.z)
        return self.a

    def backward(self, x, grad):
        # 根据激活函数类型计算导数
        if self.activation == 'softmax':
            # Softmax + 交叉熵的梯度已简化为 (predictions - targets)
            d_z = grad  # 直接使用传入的梯度
        else:
            # 其他激活函数基于激活后的值 a 计算导数
            d_z = grad * self.activation_deriv(self.a)

        # 计算参数梯度（含L2正则化）
        dW = np.dot(x.T, d_z) + self.l2_lambda * self.W
        dB = np.sum(d_z, axis=0, keepdims=True)

        # 计算输入梯度
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
        # 前向传播获取预测值
        predictions = self.forward(x)

        # 计算交叉熵损失梯度（针对 softmax 输出）
        grad = (predictions - y_true) / y_true.shape[0]  # 除以 batch_size

        # 反向传播各层
        grads = []
        layer_inputs = [x] + [layer.a for layer in self.layers[:-1]]
        for layer, layer_input in zip(reversed(self.layers), reversed(layer_inputs)):
            grad, dw, db = layer.backward(layer_input, grad)
            grads.append((dw, db))
        return grads