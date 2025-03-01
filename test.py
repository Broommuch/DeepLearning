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


# 神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, l2_lambda=1e-4):
        self.layers = []
        prev_size = input_size

        # 初始化各层参数
        for size in hidden_sizes:
            self.layers.append({
                'W': np.random.randn(prev_size, size) * np.sqrt(2. / prev_size),
                'b': np.zeros((1, size))
            })
            prev_size = size

        # 输出层
        self.layers.append({
            'W': np.random.randn(prev_size, output_size) * np.sqrt(2. / prev_size),
            'b': np.zeros((1, output_size))
        })
        self.l2_lambda = l2_lambda

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.cache = [{'A': X}]
        for i, layer in enumerate(self.layers):
            Z = np.dot(self.cache[-1]['A'], layer['W']) + layer['b']
            if i == len(self.layers) - 1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
            self.cache.append({'Z': Z, 'A': A})
        return A

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        corect_logprobs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m
        reg_loss = 0.5 * self.l2_lambda * sum(np.sum(layer['W'] ** 2) for layer in self.layers)
        return data_loss + reg_loss

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        grads = []

        # 输出层梯度
        dZ = self.cache[-1]['A'] - y
        for l in reversed(range(len(self.layers))):
            dW = np.dot(self.cache[l]['A'].T, dZ) / m + self.l2_lambda * self.layers[l]['W']
            db = np.sum(dZ, axis=0, keepdims=True) / m
            if l > 0:
                dA = np.dot(dZ, self.layers[l]['W'].T)
                dZ = dA * self.relu_deriv(self.cache[l]['Z'])
            grads.insert(0, {'dW': dW, 'db': db})

        # 更新参数
        for l in range(len(self.layers)):
            self.layers[l]['W'] -= learning_rate * grads[l]['dW']
            self.layers[l]['b'] -= learning_rate * grads[l]['db']


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

    # 创建模型 (输入784, 隐藏层[256,128], 输出10)
    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        l2_lambda=1e-4
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