from model import NeuralNetwork, FullyConnectedLayer
from utils import compute_loss, load_data, preprocess_data
import numpy as np
import matplotlib.pyplot as plt

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=50, batch_size=64, learning_rate=0.01):
    loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            loss = compute_loss(y_pred, y_batch, loss_type='cross_entropy')
            epoch_loss += loss

            # 手动计算交叉熵梯度（softmax输出层专用）
            grad = (y_pred - y_batch) / y_batch.shape[0]

            # 反向传播
            grads = model.backward(X_batch, grad)

            # 参数更新
            for layer, (dw, db) in zip(model.layers, reversed(grads)):
                layer.W -= learning_rate * dw
                layer.B -= learning_rate * db

        # 记录损失和验证集准确率
        avg_loss = epoch_loss / (len(X_train) // batch_size)
        loss_history.append(avg_loss)

        # 验证集评估
        val_pred = model.forward(X_val)
        val_pred_probs = val_pred  # 输出层已用softmax，直接取概率
        val_acc = np.mean(np.argmax(y_val, axis=1) == np.argmax(val_pred_probs, axis=1))
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    return loss_history, val_acc_history

def build_model():
    model = NeuralNetwork()
    model.add_layer(FullyConnectedLayer(784, 256, activation='relu', l2_lambda=0.001))
    model.add_layer(FullyConnectedLayer(256, 10, activation='softmax'))  # 输出层用softmax
    return model