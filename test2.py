import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 超参数配置 (可根据实验需求修改)
config = {
    # 网络结构参数
    'hidden_layers': [256, 128],  # 隐藏层结构
    'activations': ['relu', 'relu'],  # 各层激活函数
    'use_dropout': True,  # 是否使用Dropout
    'dropout_rate': 0.5,  # Dropout概率
    'initialization': 'he',  # 参数初始化方法(he/xavier)

    # 训练参数
    'epochs': 20,  # 训练轮数
    'batch_size': 64,  # 批大小
    'learning_rate': 0.01,  # 学习率
    'l2_lambda': 1e-4,  # L2正则化系数

    # 优化器参数
    'optimizer': 'adam',  # 优化器(sgd/momentum/adam)
    'momentum': 0.9,  # 动量系数
    'beta1': 0.9,  # Adam参数
    'beta2': 0.999,  # Adam参数
    'epsilon': 1e-8,  # 数值稳定项
}

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0  # 归一化
y = mnist.target.astype(np.int32)

# One-hot编码
one_hot = OneHotEncoder(sparse_output=False)
y_onehot = one_hot.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42)

# 划分验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


# 定义激活函数及其导数
class ReLU:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(x):
        return (x > 0).astype(float)


class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(x):
        sig = Sigmoid.forward(x)
        return sig * (1 - sig)


class Tanh:
    @staticmethod
    def forward(x):
        return np.tanh(x)

    @staticmethod
    def backward(x):
        return 1 - np.tanh(x) ** 2


# 定义全连接层
class DenseLayer:
    def __init__(self, input_dim, output_dim, activation, initialization):
        if initialization == 'he':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif initialization == 'xavier':
            self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        self.b = np.zeros((1, output_dim))
        self.activation = activation
        self.cache = None

    def forward(self, x, training=True):
        self.cache = x
        z = np.dot(x, self.W) + self.b
        return z

    def backward(self, dout):
        x = self.cache
        dW = np.dot(x.T, dout) + config['l2_lambda'] * self.W
        db = np.sum(dout, axis=0, keepdims=True)
        dx = np.dot(dout, self.W.T)
        return dx, dW, db


# 定义Dropout层
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1 - self.rate)
            return x * self.mask
        return x

    def backward(self, dout):
        return dout * self.mask


# 定义神经网络模型
class NeuralNetwork:
    def __init__(self, config):
        self.layers = []
        self.dropouts = []
        input_dim = 784

        # 构建网络结构
        for i, (units, act) in enumerate(zip(config['hidden_layers'], config['activations'])):
            self.layers.append(DenseLayer(input_dim, units, act, config['initialization']))
            if config['use_dropout'] and i != len(config['hidden_layers']) - 1:  # 最后一层不加Dropout
                self.dropouts.append(Dropout(config['dropout_rate']))
            input_dim = units

        # 输出层
        self.layers.append(DenseLayer(input_dim, 10, None, config['initialization']))

        # 初始化优化器参数
        if config['optimizer'] in ['momentum', 'adam']:
            self.v = [{'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                      for layer in self.layers]
        if config['optimizer'] == 'adam':
            self.m = [{'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                      for layer in self.layers]
            self.t = 0

    def forward(self, x, training=True):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer.forward(x)
            if layer.activation == 'relu':
                x = ReLU.forward(x)
            elif layer.activation == 'sigmoid':
                x = Sigmoid.forward(x)
            elif layer.activation == 'tanh':
                x = Tanh.forward(x)
            if training and i < len(self.dropouts):
                x = self.dropouts[i].forward(x, training)

        # 输出层
        x = self.layers[-1].forward(x)
        return x

    def backward(self, dout):
        grads = []
        dout, dW, db = self.layers[-1].backward(dout)
        grads.append({'W': dW, 'b': db})

        for i in reversed(range(len(self.layers) - 1)):
            if self.layers[i].activation == 'relu':
                dout *= ReLU.backward(self.layers[i].cache)
            elif self.layers[i].activation == 'sigmoid':
                dout *= Sigmoid.backward(self.layers[i].cache)
            elif self.layers[i].activation == 'tanh':
                dout *= Tanh.backward(self.layers[i].cache)

            if i < len(self.dropouts):
                dout = self.dropouts[i].backward(dout)

            dout, dW, db = self.layers[i].backward(dout)
            grads.insert(0, {'W': dW, 'b': db})

        return grads

    def update_params(self, grads):
        if config['optimizer'] == 'sgd':
            self._sgd_update(grads)
        elif config['optimizer'] == 'momentum':
            self._momentum_update(grads)
        elif config['optimizer'] == 'adam':
            self._adam_update(grads)

    def _sgd_update(self, grads):
        for layer, grad in zip(self.layers, grads):
            layer.W -= config['learning_rate'] * grad['W']
            layer.b -= config['learning_rate'] * grad['b']

    def _momentum_update(self, grads):
        for i, (layer, grad) in enumerate(zip(self.layers, grads)):
            self.v[i]['W'] = config['momentum'] * self.v[i]['W'] + config['learning_rate'] * grad['W']
            self.v[i]['b'] = config['momentum'] * self.v[i]['b'] + config['learning_rate'] * grad['b']
            layer.W -= self.v[i]['W']
            layer.b -= self.v[i]['b']

    def _adam_update(self, grads):
        self.t += 1
        for i, (layer, grad) in enumerate(zip(self.layers, grads)):
            # 更新一阶动量
            self.m[i]['W'] = config['beta1'] * self.m[i]['W'] + (1 - config['beta1']) * grad['W']
            self.m[i]['b'] = config['beta1'] * self.m[i]['b'] + (1 - config['beta1']) * grad['b']
            # 更新二阶动量
            self.v[i]['W'] = config['beta2'] * self.v[i]['W'] + (1 - config['beta2']) * (grad['W'] ** 2)
            self.v[i]['b'] = config['beta2'] * self.v[i]['b'] + (1 - config['beta2']) * (grad['b'] ** 2)
            # 偏差校正
            m_hat_w = self.m[i]['W'] / (1 - config['beta1'] ** self.t)
            m_hat_b = self.m[i]['b'] / (1 - config['beta1'] ** self.t)
            v_hat_w = self.v[i]['W'] / (1 - config['beta2'] ** self.t)
            v_hat_b = self.v[i]['b'] / (1 - config['beta2'] ** self.t)
            # 参数更新
            layer.W -= config['learning_rate'] * m_hat_w / (np.sqrt(v_hat_w) + config['epsilon'])
            layer.b -= config['learning_rate'] * m_hat_b / (np.sqrt(v_hat_b) + config['epsilon'])


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[np.arange(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss


# 初始化模型
model = NeuralNetwork(config)

# 训练循环
train_loss_history = []
val_acc_history = []

for epoch in range(config['epochs']):
    # 训练阶段
    epoch_loss = 0
    for i in range(0, X_train.shape[0], config['batch_size']):
        # 获取mini-batch
        X_batch = X_train[i:i + config['batch_size']]
        y_batch = y_train[i:i + config['batch_size']]

        # 前向传播
        logits = model.forward(X_batch, training=True)
        probs = softmax(logits)
        loss = cross_entropy_loss(probs, y_batch)

        # 添加L2正则化
        l2_loss = 0
        for layer in model.layers:
            l2_loss += 0.5 * config['l2_lambda'] * np.sum(layer.W ** 2)
        total_loss = loss + l2_loss
        epoch_loss += total_loss

        # 反向传播
        dout = (probs - y_batch) / config['batch_size']  # Softmax梯度
        grads = model.backward(dout)
        model.update_params(grads)

    # 验证阶段
    val_logits = model.forward(X_val, training=False)
    val_probs = softmax(val_logits)
    val_pred = np.argmax(val_probs, axis=1)
    val_acc = accuracy_score(np.argmax(y_val, axis=1), val_pred)

    # 记录指标
    train_loss_history.append(epoch_loss / (X_train.shape[0] // config['batch_size']))
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch + 1}/{config['epochs']} | Train Loss: {train_loss_history[-1]:.4f} | Val Acc: {val_acc:.4f}")

# 测试集评估
test_logits = model.forward(X_test, training=False)
test_probs = softmax(test_logits)
test_pred = np.argmax(test_probs, axis=1)
test_acc = accuracy_score(np.argmax(y_test, axis=1), test_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")