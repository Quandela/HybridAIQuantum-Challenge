import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def confusion_matrix_to_file(labels, predictions, path: str):
    # Compute and plot confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(path)
