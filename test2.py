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


# 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs=20):
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],  # 新增训练准确率记录
        'val_acc': []
    }

    for epoch in range(epochs):
        epoch_loss = 0

        # 训练阶段
        permutation = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], model.batch_size):
            batch_end = min(i + model.batch_size, X_train.shape[0])
            indices = permutation[i:batch_end]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            output = model.forward(X_batch)
            batch_loss = model.compute_loss(output, y_batch)
            epoch_loss += batch_loss * X_batch.shape[0]

            model.backward(X_batch, y_batch)

        # 计算训练集准确率
        train_output = model.forward(X_train)
        train_acc = np.mean(np.argmax(train_output, axis=1) == np.argmax(y_train, axis=1))

        # 验证集评估
        val_output = model.forward(X_val)
        val_acc = np.mean(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1))

        # 记录指标
        history['train_loss'].append(epoch_loss / X_train.shape[0])
        history['val_loss'].append(model.compute_loss(val_output, y_val))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    return history


# 可视化函数（更新版）
def visualize_results(history, y_true, y_pred, X_val):
    # 训练曲线（损失）
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig('loss_curves.png')
    plt.show()
    plt.close()

    # 准确率曲线（新增）
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Accuracy")
    plt.savefig('accuracy_curves.png')
    plt.show()
    plt.close()

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

    # 错误样本
    plt.figure(figsize=(12, 12))
    wrong = np.where(y_pred != y_true)[0][:25]
    for i, idx in enumerate(wrong):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_val[idx].reshape(28, 28), cmap='gray')
        plt.title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=8)
        plt.axis('off')
    plt.suptitle("Misclassified Samples")
    plt.tight_layout()
    plt.savefig('misclassified_samples.png')
    plt.show()
    plt.close()


# 主程序
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        activation='relu',
        alpha=0.01,
        learning_rate=0.01,
        batch_size=64,
        l2_lambda=1e-4
    )

    history = train(model, X_train, y_train, X_val, y_val, epochs=20)
    y_pred = np.argmax(model.forward(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    visualize_results(history, y_true, y_pred, X_val)