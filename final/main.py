from model import NeuralNetwork
from utils import load_data
from train import train
import numpy as np
from visualize import visualize_results

# 主程序
if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data()

    model = NeuralNetwork(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        activation='relu',  # relu/sigmoid/tanh/leaky_relu
        alpha=0.01,
        learning_rate=0.01,
        batch_size=64,
        l2_lambda=1e-4
    )

    history = train(model, X_train, y_train, X_val, y_val, epochs=20)
    y_pred = np.argmax(model.forward(X_val), axis=1)
    y_true = np.argmax(y_val, axis=1)

    visualize_results(history, y_true, y_pred, X_val)