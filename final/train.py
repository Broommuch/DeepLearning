import numpy as np

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