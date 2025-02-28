from train import build_model, train_model
from utils import load_data

if __name__ == "__main__":
    # 加载数据
    X_train, y_train = load_data()

    # 构建模型
    model = build_model()

    # 训练模型
    train_model(model, X_train, y_train, epochs=10)