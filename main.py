from src.load_data import load_digits_filtered
import matplotlib.pyplot as plt
from src.features import extract_features


X_train, X_test, y_train, y_test = load_digits_filtered(3, 8)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Example labels:", X_train[:5])
print("Example labels:", y_train[:5])



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
    plt.show()  # <--- this goes OUTSIDE the loop




X_train_features = extract_features(X_train)
X_test_features  = extract_features(X_test)

print("Example features:", X_train_features[:5])
print("Example features:", X_test_features[:5])

# Show the first 5 images
show_images(X_train, y_train, count=5)