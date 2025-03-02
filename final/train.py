import numpy as np


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