import matplotlib.pyplot as plt

def plot_loss_curves(loss_history, val_acc_history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.show()