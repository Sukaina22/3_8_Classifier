Answers of questions 3 and 4 in the task sheet:

3- Based on the below evaluating results of the two models:
Metric Logistic Regression SVM
Accuracy 0.972 0.972
Precision (0) 0.95 0.97
Recall (0) 1.00 0.97
F1-Score (0) 0.97 0.97
Precision (1) 1.00 0.97
Recall (1) 0.94 0.97
F1-Score (1) 0.97 0.97

I can say both models performed well and identically in accuracy and f1 score. However logistic regression had a perfect precision for class 1 but a lower recall while svm had same precision and recall for the two classes.

Therefore, I can tell SVM showed more balanced behavior if we compare the metrics of the 3 and 8 classes. Linear SVM maximizes the margin between classes — just in a straight line/hyperplane, and this gives it strong generalization.

4- In this project, both Logistic Regression and SVM achieved high classification accuracy (~97%) on distinguishing digits 3 and 8. Since the feature space was linearly separable, both models learned similar decision boundaries.

When to Prefer SVM
Clear class separation: SVM performs well when the data is linearly separable.

Small to medium-sized datasets: Efficient and accurate when data size is manageable.

Robustness to outliers: SVM focuses on support vectors, not all data points.

Feature engineering helps: In this project, engineered features allowed the SVM to find a clearer margin.

When to Prefer Logistic Regression
Probability output is needed: Logistic Regression provides class probabilities, useful for thresholds.

Large datasets: Faster and more scalable than SVM on high-volume data.

Overlapping/noisy classes: Can generalize better in fuzzy decision regions.

Simplicity and interpretability: Coefficients are easy to interpret and analyze.

In this project, SVM slightly outperformed Logistic Regression due to the engineered features making class separation more defined — a scenario where SVM’s margin-maximization is advantageous.
