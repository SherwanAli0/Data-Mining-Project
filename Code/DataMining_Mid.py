import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

np.random.seed(42)

data = pd.read_csv('/Users/ipekucar/Desktop/diabetes.csv')

print("Classification on the Original Dataset")
print("Original Dataset")
print(data.head())
print()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model_original = SVC()
svm_model_original.fit(X_train, y_train)
y_pred_svm_original = svm_model_original.predict(X_test)
cm_svm_original = confusion_matrix(y_test, y_pred_svm_original)
accuracy_svm_original = accuracy_score(y_test, y_pred_svm_original)

print("Confusion Matrix for SVM (Original Dataset):")
print(pd.DataFrame(cm_svm_original, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM (Original Dataset):", accuracy_svm_original)
print()

knn_model_original = KNeighborsClassifier()
knn_model_original.fit(X_train, y_train)
y_pred_knn_original = knn_model_original.predict(X_test)
cm_knn_original = confusion_matrix(y_test, y_pred_knn_original)
accuracy_knn_original = accuracy_score(y_test, y_pred_knn_original)

print("Confusion Matrix for KNN (Original Dataset):")
print(pd.DataFrame(cm_knn_original, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN (Original Dataset):", accuracy_knn_original)
print()

print(" Replace Some Values with NaN")
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

print("Dataset with NaN values (After Replacing 0s):")
print(data.head())
print()

print(" Replace NaN Values with Column Means")
data_filled = data.copy()
data_filled[columns_with_zeros] = data_filled[columns_with_zeros].fillna(data_filled[columns_with_zeros].mean())

print("Dataset After Replacing NaN with Column Means (First 5 Rows):")
print(data_filled.head())
print()

print("Classification After Replacing NaN with Column Means")
X_filled = data_filled.iloc[:, :-1]
y_filled = data_filled.iloc[:, -1]

X_train_filled, X_test_filled, y_train_filled, y_test_filled = train_test_split(X_filled, y_filled, test_size=0.2, random_state=42)

svm_model_filled = SVC()
svm_model_filled.fit(X_train_filled, y_train_filled)
y_pred_svm_filled = svm_model_filled.predict(X_test_filled)
cm_svm_filled = confusion_matrix(y_test_filled, y_pred_svm_filled)
accuracy_svm_filled = accuracy_score(y_test_filled, y_pred_svm_filled)

print("Confusion Matrix for SVM (After Replacing NaN with Column Means):")
print(pd.DataFrame(cm_svm_filled, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM (After Replacing NaN with Column Means):", accuracy_svm_filled)
print()

knn_model_filled = KNeighborsClassifier()
knn_model_filled.fit(X_train_filled, y_train_filled)
y_pred_knn_filled = knn_model_filled.predict(X_test_filled)
cm_knn_filled = confusion_matrix(y_test_filled, y_pred_knn_filled)
accuracy_knn_filled = accuracy_score(y_test_filled, y_pred_knn_filled)

print("Confusion Matrix for KNN (After Replacing NaN with Column Means):")
print(pd.DataFrame(cm_knn_filled, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN (After Replacing NaN with Column Means):", accuracy_knn_filled)
print()
