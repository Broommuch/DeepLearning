import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# 可视化函数（更新版）
def visualize_results(history, y_true, y_pred, X_val):
    # 训练曲线（损失）
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Loss")
    plt.savefig('loss_curves.png')
    plt.show()
    plt.close()

    # 准确率曲线（新增）
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title("Training and Validation Accuracy")
    plt.savefig('accuracy_curves.png')
    plt.show()
    plt.close()

    # 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    thresh = cm.max() / 2.
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

    # 错误样本
    plt.figure(figsize=(12, 12))
    wrong = np.where(y_pred != y_true)[0][:25]
    for i, idx in enumerate(wrong):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_val[idx].reshape(28, 28), cmap='gray')
        plt.title(f"T:{y_true[idx]}\nP:{y_pred[idx]}", fontsize=8)
        plt.axis('off')
    plt.suptitle("Misclassified Samples")
    plt.tight_layout()
    plt.savefig('misclassified_samples.png')
    plt.show()
    plt.close()