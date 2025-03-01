from model import NeuralNetwork
from utils import load_data
from train import train
import numpy as np

# 主程序
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    # 创建双层网络（784 → 256 → 10），可自由选择激活函数
    model = NeuralNetwork(
        input_size=784,
        hidden_size=256,
        output_size=10,
        activation='relu'  # 可替换为sigmoid或tanh
    )

    # 训练参数
    train(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=20,
        lr=0.01,
        batch_size=64
    )

    # 最终验证集准确率
    final_output = model.forward(X_val)
    final_acc = np.mean(np.argmax(final_output, axis=1) == np.argmax(y_val, axis=1))
    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")