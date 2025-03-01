import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation='relu', l2_lambda=1e-4,
                 learning_rate=0.01, batch_size=64):
        # 网络参数
        self.layer_dims = [input_size] + hidden_sizes + [output_size]
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # 初始化参数
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_dims) - 1):
            # He初始化
            scale = np.sqrt(2.0 / self.layer_dims[i])
            self.weights.append(np.random.randn(self.layer_dims[i], self.layer_dims[i + 1]) * scale)
            self.biases.append(np.zeros((1, self.layer_dims[i + 1])))

        # 缓存中间结果
        self.caches = []

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        self.caches = []
        A = X

        # 前向传播所有隐藏层
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.caches.append((A.copy(), Z.copy()))
            A = self.relu(Z)

        # 输出层前保存最后的激活输出
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        self.caches.append((A.copy(), Z.copy()))
        return self.softmax(Z)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        corect_logprobs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m

        # L2正则化项
        reg_loss = 0
        for W in self.weights:
            reg_loss += 0.5 * self.l2_lambda * np.sum(W ** 2)

        return data_loss + reg_loss

    def backward(self, X, y):
        m = X.shape[0]
        grads = []

        # 输出层梯度
        dZ = self.forward(X) - y  # 确保使用最新缓存

        # 反向传播所有层
        for i in reversed(range(len(self.weights))):
            # 获取正确的A_prev
            A_prev, _ = self.caches[i]

            # 计算梯度
            dW = np.dot(A_prev.T, dZ) / m + self.l2_lambda * self.weights[i]
            db = np.sum(dZ, axis=0, keepdims=True) / m
            grads.insert(0, (dW, db))

            # 计算前一层梯度（除输入层外）
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                _, Z_prev = self.caches[i - 1]
                dZ = dA_prev * self.relu_deriv(Z_prev)

        # 更新参数
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads[i][0]
            self.biases[i] -= self.learning_rate * grads[i][1]


# 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs=20):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        epoch_loss = 0

        # Mini-batch训练
        permutation = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], model.batch_size):
            indices = permutation[i:i + model.batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # 前向传播
            output = model.forward(X_batch)
            batch_loss = model.compute_loss(output, y_batch)
            epoch_loss += batch_loss * X_batch.shape[0]

            # 反向传播
            model.backward(X_batch, y_batch)

        # 记录训练损失
        history['train_loss'].append(epoch_loss / X_train.shape[0])

        # 验证集评估
        val_output = model.forward(X_val)
        val_loss = model.compute_loss(val_output, y_val)
        val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    return history


# 可视化函数
def plot_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], 'g-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_misclassified_samples(X, y_true, y_pred, num_samples=25):
    misclassified = np.where(y_pred != y_true)[0]
    np.random.shuffle(misclassified)
    selected = misclassified[:num_samples]

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(selected):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('misclassified_samples.png')
    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载数据
    X_train, X_val, y_train, y_val = load_data()

    # 模型配置示例（可自由修改）
    config = {
        'input_size': 784,
        'hidden_sizes': [256, 128],  # 支持任意结构如[512, 256], [128]等
        'output_size': 10,
        'l2_lambda': 1e-4,
        'learning_rate': 0.01,
        'batch_size': 64
    }

    # 初始化模型
    model = NeuralNetwork(**config)

    # 训练模型
    history = train(model, X_train, y_train, X_val, y_val, epochs=20)

    # 生成预测结果
    val_probs = model.forward(X_val)
    y_pred = np.argmax(val_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # 可视化结果
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_misclassified_samples(X_val, y_true, y_pred)