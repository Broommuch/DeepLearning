from train import build_model, train_model
from utils import load_data, preprocess_data
from visualize import plot_loss_curves, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. 加载并预处理数据
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(X, y)

    # 检查归一化范围
    print("Pixel 值范围:", X_train.min(), X_train.max())  # 应为 [0.0, 1.0]

    # 检查 One-hot 编码
    print("y_train[0] 的 One-hot 编码:", y_train[0])  # 应为类似 [0,0,1,...,0]
    # 统计原始数据、训练集、验证集的类别分布
    print("原始数据分布:", np.bincount(y) / len(y))
    print("训练集分布:", np.bincount(np.argmax(y_train, axis=1)) / len(y_train))
    print("验证集分布:", np.bincount(np.argmax(y_val, axis=1)) / len(y_val))

    # 验证前5个样本的标签与图像是否匹配
    for i in range(5):
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        true_label = np.argmax(y_train[i])  # 从 One-hot 解码
        plt.title(f"True Label: {true_label}")
        plt.show()

    # 2. 构建模型
    model = build_model()

    # 3. 训练模型（传入验证集）
    loss_history, val_acc_history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,  # 添加验证集
        epochs=10,
        batch_size=128,
        learning_rate=0.05
    )

    # 4. 可视化训练曲线
    plot_loss_curves(loss_history, val_acc_history)

    # 5. 生成混淆矩阵
    val_pred = model.forward(X_val)
    plot_confusion_matrix(y_val, val_pred)