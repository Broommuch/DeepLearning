import numpy as np

# 修改后的训练函数（添加历史记录）
def train(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.01, batch_size=64):
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # 训练阶段
        epoch_train_loss = 0
        permutation = np.random.permutation(X_train.shape[0])
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # 前向传播并计算损失
            output = model.forward(X_batch)
            batch_loss = model.compute_loss(output, y_batch)
            epoch_train_loss += batch_loss * X_batch.shape[0]

            # 反向传播
            model.backward(X_batch, y_batch, lr)

        # 记录训练损失
        history['train_loss'].append(epoch_train_loss / X_train.shape[0])

        # 验证阶段
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