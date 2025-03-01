from model import NeuralNetwork
from utils import load_data
from train import train
import numpy as np
from visualize import visualize_history

# 修改后的主程序
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    model = NeuralNetwork(
        input_size=784,
        hidden_size=256,
        output_size=10,
        activation='relu'
    )

    # 训练并获取历史数据
    history = train(model, X_train, y_train, X_val, y_val)

    # 可视化训练过程
    visualize_history(history)