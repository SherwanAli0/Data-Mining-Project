import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)

# load the dataset using relative path
data = pd.read_csv('DataSet/diabetes.csv')

print("=" * 50)
print("STEP 1: Original Dataset")
print("=" * 50)
print(data.head())
print()
print("Dataset shape:", data.shape)
print()
print("Class distribution:")
print(data['Outcome'].value_counts())
print()

# split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features - this is important for SVM and KNN because they use distances
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train SVM on original dataset (with scaling)
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Confusion Matrix for SVM (Original Dataset):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred_svm),
                   index=["Actual 0", "Actual 1"],
                   columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM:", round(accuracy_svm, 4))
print()

# train KNN on original dataset (with scaling)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Confusion Matrix for KNN (Original Dataset):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred_knn),
                   index=["Actual 0", "Actual 1"],
                   columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN:", round(accuracy_knn, 4))
print()

# ============================================================
print("=" * 50)
print("STEP 2: Handling Missing Values")
print("=" * 50)

# in this dataset, 0 values in these columns are actually missing data
columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("Number of 0 values in each column:")
for col in columns_with_zeros:
    print(f"  {col}: {(data[col] == 0).sum()}")
print()

# replace 0s with NaN
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

# fill NaN with column means
data_filled = data.copy()
data_filled[columns_with_zeros] = data_filled[columns_with_zeros].fillna(data_filled[columns_with_zeros].mean())

print("After replacing 0s with column means:")
print(data_filled.head())
print()

# split again with cleaned data
X_filled = data_filled.iloc[:, :-1]
y_filled = data_filled.iloc[:, -1]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_filled, y_filled, test_size=0.2, random_state=42)

# scale the cleaned data
scaler2 = StandardScaler()
X_train_f_scaled = scaler2.fit_transform(X_train_f)
X_test_f_scaled = scaler2.transform(X_test_f)

# train SVM on cleaned dataset
svm_model2 = SVC()
svm_model2.fit(X_train_f_scaled, y_train_f)
y_pred_svm2 = svm_model2.predict(X_test_f_scaled)
accuracy_svm2 = accuracy_score(y_test_f, y_pred_svm2)

print("Confusion Matrix for SVM (Cleaned Dataset):")
print(pd.DataFrame(confusion_matrix(y_test_f, y_pred_svm2),
                   index=["Actual 0", "Actual 1"],
                   columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for SVM:", round(accuracy_svm2, 4))
print()

# train KNN on cleaned dataset
knn_model2 = KNeighborsClassifier()
knn_model2.fit(X_train_f_scaled, y_train_f)
y_pred_knn2 = knn_model2.predict(X_test_f_scaled)
accuracy_knn2 = accuracy_score(y_test_f, y_pred_knn2)

print("Confusion Matrix for KNN (Cleaned Dataset):")
print(pd.DataFrame(confusion_matrix(y_test_f, y_pred_knn2),
                   index=["Actual 0", "Actual 1"],
                   columns=["Predicted 0", "Predicted 1"]))
print("Accuracy for KNN:", round(accuracy_knn2, 4))
print()

# ============================================================
print("=" * 50)
print("STEP 3: Applying SMOTE to Balance the Dataset")
print("=" * 50)

print("Before SMOTE:")
print(y_train_f.value_counts())
print()

# apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_f_scaled, y_train_f)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())
print()

# train SVM on SMOTE balanced data
svm_model3 = SVC()
svm_model3.fit(X_train_smote, y_train_smote)
y_pred_svm3 = svm_model3.predict(X_test_f_scaled)

print("SVM Results (After SMOTE):")
print("Accuracy:", round(accuracy_score(y_test_f, y_pred_svm3), 4))
print("Precision:", round(precision_score(y_test_f, y_pred_svm3), 4))
print("Recall:", round(recall_score(y_test_f, y_pred_svm3), 4))
print("F1-Score:", round(f1_score(y_test_f, y_pred_svm3), 4))
print()

# train KNN on SMOTE balanced data
knn_model3 = KNeighborsClassifier()
knn_model3.fit(X_train_smote, y_train_smote)
y_pred_knn3 = knn_model3.predict(X_test_f_scaled)

print("KNN Results (After SMOTE):")
print("Accuracy:", round(accuracy_score(y_test_f, y_pred_knn3), 4))
print("Precision:", round(precision_score(y_test_f, y_pred_knn3), 4))
print("Recall:", round(recall_score(y_test_f, y_pred_knn3), 4))
print("F1-Score:", round(f1_score(y_test_f, y_pred_knn3), 4))
print()

# ============================================================
print("=" * 50)
print("STEP 4: Confusion Matrix Heatmaps")
print("=" * 50)

# plot confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SVM heatmap
cm_svm = confusion_matrix(y_test_f, y_pred_svm3)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
axes[0].set_title("SVM Confusion Matrix (SMOTE)")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# KNN heatmap
cm_knn = confusion_matrix(y_test_f, y_pred_knn3)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
axes[1].set_title("KNN Confusion Matrix (SMOTE)")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("Output/confusion_matrices.png")
plt.show()

print("Heatmaps saved to Output/confusion_matrices.png")
print()

# ============================================================
print("=" * 50)
print("STEP 5: Summary of All Results")
print("=" * 50)

summary = pd.DataFrame({
    'Model': ['SVM', 'SVM', 'SVM', 'KNN', 'KNN', 'KNN'],
    'Stage': ['Original', 'Cleaned', 'SMOTE', 'Original', 'Cleaned', 'SMOTE'],
    'Accuracy': [
        round(accuracy_svm, 4),
        round(accuracy_svm2, 4),
        round(accuracy_score(y_test_f, y_pred_svm3), 4),
        round(accuracy_knn, 4),
        round(accuracy_knn2, 4),
        round(accuracy_score(y_test_f, y_pred_knn3), 4)
    ]
})

print(summary.to_string(index=False))
print()
print("Done!")
