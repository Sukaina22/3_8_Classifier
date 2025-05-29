from src.load_data import load_digits_filtered
from src.features import extract_features
from sklearn.preprocessing import StandardScaler
from src.models import get_logistic_model
from src.models import get_svm_model
from src.evaluate import evaluate_and_report, plot_confusion_matrix
from src.utils import plot_decision_boundary, show_images
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = load_digits_filtered(3, 8)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Example labels:", X_train[:5])
print("Example labels:", y_train[:5])

X_train_features = extract_features(X_train)
X_test_features  = extract_features(X_test)

# Normalization of features is required; the features values have different ranges.
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)                                             

print("Example features:", X_train_features[:5])
print("Example features:", X_test_features[:5])

show_images(X_train, y_train, count=5)

logistic_model = get_logistic_model()
logistic_model.fit(X_train_features, y_train)

svm_model = get_svm_model()
svm_model.fit(X_train_features, y_train)

scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validated scores:", scores)
print("Mean accuracy:", scores.mean())

scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validated scores:", scores)
print("Mean accuracy:", scores.mean())

logistic_acc, logistic_preds = evaluate_and_report(logistic_model, X_test_features, y_test, model_name="Logistic Regression")

svm_acc, svm_preds = evaluate_and_report(svm_model, X_test_features, y_test, model_name="SVM")



# Optional
plot_confusion_matrix(y_test, logistic_preds, class_names=["Digit 3", "Digit 8"], model_name="Logistic Regression")
plot_confusion_matrix(y_test,  svm_preds, class_names=["Digit 3", "Digit 8"], model_name="SVM")

X_plot = X_train_features[:, [0, 5]]  
y_plot = y_train

log_clf = LogisticRegression().fit(X_plot, y_plot)
svm_clf = SVC(kernel="linear").fit(X_plot, y_plot)

plot_decision_boundary(log_clf, X_plot, y_plot, "Logistic Regression Decision Boundary")
plot_decision_boundary(svm_clf, X_plot, y_plot, "SVM Decision Boundary")








