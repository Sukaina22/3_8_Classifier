from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score

def evaluate_and_report(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = [0, 1] 
    target_names = ["Digit 3", "Digit 8"]
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    print(f"{model_name} Accuracy: {acc:.4f}")

    return acc, y_pred


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    labels = [0, 1] 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

