from utils import load_data, preprocess_data
import matplotlib.pyplot as plt
from model import NeuralNetwork
from utils import load_data, compute_loss

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # Mini-batch遍历
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 计算损失
            loss = cross_entropy_loss(y_batch, y_pred)

            # 反向传播
            grads = model.backward(X_batch, y_batch)

            # 参数更新
            for (dw, db) in grads:
                model.layers[0].W -= learning_rate * dw
                model.layers[0].B -= learning_rate * db

        # 验证集评估
        val_loss = cross_entropy_loss(y_val, model.forward(X_val))
        val_acc = np.mean(np.argmax(y_val, axis=1) == np.argmax(model.forward(X_val), axis=1))

        loss_history.append(loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Val Acc: {val_acc:.4f}")

    return loss_history, val_acc_history

def build_model():
    model = NeuralNetwork()
    model.add_layer(FullyConnectedLayer(784, 256))
    model.add_layer(ReLU())
    return model