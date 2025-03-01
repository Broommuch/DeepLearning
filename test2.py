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
    def __init__(self, input_size, hidden_size, output_size,
                 activation='relu', l2_lambda=1e-4):
        # 初始化参数
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.l2_lambda = l2_lambda

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_deriv(self, Z):
        return Z > 0

    def softmax(self, Z):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X):
        # 第一层
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        # 输出层
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.softmax(self.Z2)

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        corect_logprobs = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        data_loss = np.sum(corect_logprobs) / m
        reg_loss = 0.5 * self.l2_lambda * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss

    def backward(self, X, y, learning_rate):
        m = X.shape[0]

        # 输出层梯度
        dZ2 = self.softmax(self.Z2) - y
        dW2 = np.dot(self.A1.T, dZ2) / m + self.l2_lambda * self.W2
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m + self.l2_lambda * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # 参数更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# 训练函数
def train(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.01, batch_size=64):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        epoch_loss = 0
        permutation = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            output = model.forward(X_batch)
            batch_loss = model.compute_loss(output, y_batch)
            epoch_loss += batch_loss * X_batch.shape[0]

            model.backward(X_batch, y_batch, lr)

        history['train_loss'].append(epoch_loss / X_train.shape[0])

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
    # 数据加载
    X_train, X_val, y_train, y_val = load_data()

    # 模型初始化
    model = NeuralNetwork(
        input_size=784,
        hidden_size=256,
        output_size=10,
        l2_lambda=1e-4
    )

    # 模型训练
    history = train(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.01)

    # 生成预测结果
    val_probs = model.forward(X_val)
    y_pred = np.argmax(val_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # 可视化结果
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_misclassified_samples(X_val, y_true, y_pred)