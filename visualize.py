import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_loss_curves(loss_history, val_acc_history):
    """绘制训练损失和验证准确率曲线"""
    plt.figure(figsize=(12, 5))

    # 子图1：训练损失
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    # 子图2：验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(val_acc_history, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.ylim(0, 1)  # 确保准确率在 0-100% 范围内
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """绘制带标注的混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(
        np.argmax(y_true, axis=1),  # 假设 y_true 是 One-hot 编码
        np.argmax(y_pred, axis=1)  # 假设 y_pred 是模型输出的概率分布
    )

    # 可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    # 添加轴标签和数值标注
    if class_names is None:
        class_names = [str(i) for i in range(10)]  # MNIST 默认 0-9
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在单元格中显示数值
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()