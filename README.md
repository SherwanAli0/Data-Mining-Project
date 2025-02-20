This is the final version of the Data Mining Project. It builds upon the previous versions by addressing class imbalance using SMOTE and improving model evaluation with additional metrics and hyperparameter tuning.

Dataset
The dataset used in this project is publicly available and can be downloaded from Kaggle.

Steps to Use the Dataset
Go to the Kaggle dataset page.
Download the file diabetes.csv.
Place the file in the data folder within the project directory: /data/diabetes.csv
Run the code as described in the "How to Run" section.
Key Features
SMOTE Implementation:

Class imbalance in the dataset is addressed using SMOTE (Synthetic Minority Oversampling Technique).
Additional Metrics:

Metrics such as Precision, Recall, and F1-Score are calculated for both SVM and KNN classifiers.
Visualizations:

Confusion matrices are visualized as heatmaps for better interpretability.
Code Refinement:

Improved structure and comments for clarity and better maintainability.
Steps in the Code
Original Dataset Analysis:

Load the dataset (diabetes.csv) and display the first 5 rows.
Split the data into training and testing sets (80% training, 20% testing).
Train and evaluate SVM and KNN models on the original dataset.
Display confusion matrices, accuracy scores, and class distributions.
Handle Missing Values:

Replace zeros in specific columns (Glucose, BloodPressure, SkinThickness, Insulin, BMI) with NaN.
Fill NaN values with the mean of each column.
Cleaned Dataset Analysis:

Retrain and evaluate SVM and KNN models on the cleaned dataset.
Generate confusion matrices and accuracy scores.
SMOTE Implementation:

Apply SMOTE to balance the training data for fair evaluation.
Train and evaluate SVM and KNN models on the balanced dataset.
Calculate additional metrics: Precision, Recall, and F1-Score.
Heatmap Visualizations:

Generate heatmaps of confusion matrices for both SVM and KNN classifiers after balancing with SMOTE.
How to Run
Download the dataset as explained in the "Dataset" section.
Place the dataset in the /data folder.
Install the required libraries:
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib

Run the script : python DataMining_Final.py
Outputs
All outputs for this version are saved in the outputs/final-version-output.txt file. Metrics include:
Original Dataset Analysis
Confusion matrices and accuracy scores for SVM and KNN.
Cleaned Dataset Analysis
Confusion matrices and accuracy scores for SVM and KNN after filling missing values.
Balanced Dataset Analysis
Confusion matrices, accuracy scores, precision, recall, and F1-scores for SVM and KNN after applying SMOTE.
Heatmaps
Heatmaps of confusion matrices are displayed during execution.
Example Metrics (Balanced Dataset)
SVM Metrics:

Accuracy: 0.6948
Precision: 0.5606
Recall: 0.6727
F1-Score: 0.6116 KNN Metrics:
Accuracy: 0.6364
Precision: 0.4938
Recall: 0.7273
F1-Score: 0.5882
Limitations of This Version
Hyperparameter tuning is not fully implemented for SVM and KNN models.
Feature importance analysis is not explored.
Advanced visualizations like ROC curves are not included.
Next Steps
In a future version, we could:

Implement hyperparameter tuning for better model performance.
Explore feature importance to understand the contribution of each variable.
Generate advanced visualizations such as ROC and precision-recall curves.
