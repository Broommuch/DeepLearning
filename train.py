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
        # Mini-batch遍历
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失（交叉熵）
            loss = compute_loss(y_pred, y_batch, loss_type='cross_entropy')
            epoch_loss += loss

            # 反向传播
            grads = model.backward(X_batch, y_batch)

            # 参数更新（逐层更新）
            for layer, (dw, db) in zip(model.layers, reversed(grads)):
                layer.W -= learning_rate * dw
                layer.B -= learning_rate * db

        # 记录平均损失
        avg_loss = epoch_loss / (len(X_train) // batch_size)
        loss_history.append(avg_loss)

        # 验证集评估
        val_pred = model.forward(X_val)
        val_pred_probs = np.exp(val_pred) / np.sum(np.exp(val_pred), axis=1, keepdims=True)
        val_acc = np.mean(np.argmax(y_val, axis=1) == np.argmax(val_pred_probs, axis=1))
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

    return loss_history, val_acc_history

def build_model():
    model = NeuralNetwork()
    model.add_layer(FullyConnectedLayer(784, 256, activation='relu', l2_lambda=0.001))
    model.add_layer(FullyConnectedLayer(256, 10, activation='softmax'))  # 输出层用softmax
    return model