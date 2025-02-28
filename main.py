from train import build_model, train_model
from utils import load_data, preprocess_data
from visualize import plot_loss_curves, plot_confusion_matrix
import numpy as np

if __name__ == "__main__":
    # 1. 加载并预处理数据
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(X, y)

    # 2. 构建模型
    model = build_model()

    # 3. 训练模型（传入验证集）
    loss_history, val_acc_history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,  # 添加验证集
        epochs=10,
        batch_size=64,
        learning_rate=0.01
    )

    # 4. 可视化训练曲线
    plot_loss_curves(loss_history, val_acc_history)

    # 5. 生成混淆矩阵
    val_pred = model.forward(X_val)
    plot_confusion_matrix(y_val, val_pred)