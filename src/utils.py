import matplotlib.pyplot as plt
import numpy as np

# A function that visualizes sample images from the loaded data.
def show_images(images, labels, count=5):
    plt.figure(figsize=(10, 2))
    for i in range(count):
        image = images[i].reshape(8, 8)
        plt.subplot(1, count, i + 1)
        plt.imshow(image, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()  

# A function that visualizes the separation of data based on two chosen features.
def plot_feature_separation(X, y, fx=0, fy=5):
    plt.figure(figsize=(8, 6))
    for label in np.unique(y):
        mask = y == label
        plt.scatter(
            X[mask, fx], X[mask, fy],
            alpha=0.7,
            label=f"Digit {3 if label == 0 else 8}"
        )

    plt.xlabel(f"Feature {fx}")
    plt.ylabel(f"Feature {fy}")
    plt.title(f"Feature {fx} vs Feature {fy}")
    plt.grid(True)
    plt.legend()
    plt.show()    

# A function that plots the decision boundary of a model given dataset.
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()