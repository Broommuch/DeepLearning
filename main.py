from train import build_model, train_model
from utils import load_data, preprocess_data,compute_loss
from visualize import plot_loss_curves, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. 加载并预处理数据
    X, y = load_data()
    X_train, X_val, y_train, y_val = preprocess_data(X, y)


    # 添加梯度检验代码
    def numerical_gradient(model, X, y):
        eps = 1e-7
        grads_numerical = []

        # 遍历所有层的参数
        for layer in model.layers:
            # 仅处理权重 W（忽略偏置 B）
            W_shape = layer.W.shape
            W_flat = layer.W.flatten()
            grad_W = np.zeros_like(W_flat)

            # 对每个权重计算数值梯度
            for i in range(len(W_flat)):
                original = W_flat[i]

                # 正向扰动
                W_flat[i] = original + eps
                layer.W = W_flat.reshape(W_shape)
                loss_plus = compute_loss(model.forward(X), y)

                # 负向扰动
                W_flat[i] = original - eps
                layer.W = W_flat.reshape(W_shape)
                loss_minus = compute_loss(model.forward(X), y)

                # 恢复原始值
                W_flat[i] = original
                grad_W[i] = (loss_plus - loss_minus) / (2 * eps)

            grads_numerical.append(grad_W.reshape(W_shape))

        return grads_numerical


    # 比较数值梯度与反向传播梯度
    X_sample, y_sample = X_train[:10], y_train[:10]
    model = build_model()

    # 反向传播梯度
    output = model.forward(X_sample)
    grads_backprop = model.backward(X_sample, y_sample)

    # 数值梯度
    grads_numerical = numerical_gradient(model, X_sample, y_sample)

    # 调整梯度顺序
    grads_backprop_reversed = list(reversed(grads_backprop))

    # 打印梯度形状并计算误差
    for i, (gn, gb) in enumerate(zip(grads_numerical, grads_backprop_reversed)):
        print(f"Layer {i + 1}:")
        print("  数值梯度形状:", gn.shape)
        print("  反向传播梯度形状:", gb[0].shape)
        error = np.linalg.norm(gn - gb[0]) / (np.linalg.norm(gn) + np.linalg.norm(gb[0]))
        print(f"  相对误差: {error:.6f}")

    pass
    #debug utils.py
    # # 检查归一化范围
    # print("Pixel 值范围:", X_train.min(), X_train.max())  # 应为 [0.0, 1.0]
    #
    # # 检查 One-hot 编码
    # print("y_train[0] 的 One-hot 编码:", y_train[0])  # 应为类似 [0,0,1,...,0]
    # # 统计原始数据、训练集、验证集的类别分布
    # print("原始数据分布:", np.bincount(y) / len(y))
    # print("训练集分布:", np.bincount(np.argmax(y_train, axis=1)) / len(y_train))
    # print("验证集分布:", np.bincount(np.argmax(y_val, axis=1)) / len(y_val))
    #
    # # 验证前5个样本的标签与图像是否匹配
    # for i in range(10):
    #     plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
    #     true_label = np.argmax(y_train[i])  # 从 One-hot 解码
    #     plt.title(f"True Label: {true_label}")
    #     plt.show()

    # 2. 构建模型
    model = build_model()

    # debug model.py
    # dummy_input = np.random.randn(32, 784)  # 模拟一个 batch 的数据
    # output = model.forward(dummy_input)
    # print("输出层值范围:", output.min(), output.max())  # 应为概率值 [0,1]
    # print("输出层求和:", np.sum(output[0]))  # 应接近 1.0

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