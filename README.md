# Data Mining Project - Final Version

This final version of the Data Mining Project enhances previous iterations by addressing class imbalance through SMOTE and improving model evaluation with additional metrics and hyperparameter adjustments.

## Dataset
The dataset used in this project is publicly available and can be accessed from Kaggle.

### How to Use the Dataset
1. Visit the Kaggle dataset page.
2. Download `diabetes.csv`.
3. Place the file in the `data/` folder within the project directory: `data/diabetes.csv`.
4. Follow the steps in the "How to Run" section.

## Key Enhancements
### SMOTE for Class Balancing
- Synthetic Minority Oversampling Technique (SMOTE) is used to balance the dataset, reducing bias in model training.

### Expanded Model Evaluation
- Additional performance metrics, including Precision, Recall, and F1-Score, are calculated for both SVM and KNN classifiers.

### Data Visualization
- Confusion matrices are visualized as heatmaps to improve interpretability.

### Code Optimization
- Improved structure with better organization and documentation for clarity and maintainability.

## Steps in the Code
### Initial Analysis
- Load and explore the dataset (`diabetes.csv`).
- Split the data into training (80%) and testing (20%) subsets.
- Train and evaluate SVM and KNN models on the unmodified dataset.
- Generate confusion matrices, accuracy scores, and examine class distribution.

### Handling Missing Data
- Replace zeros in key columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with NaN values.
- Impute missing values using the mean of each respective column.

### Cleaned Dataset Analysis
- Retrain and evaluate SVM and KNN models after cleaning missing values.
- Generate updated confusion matrices and accuracy scores.

### Applying SMOTE
- Implement SMOTE to balance the training dataset.
- Retrain and evaluate SVM and KNN models on the balanced dataset.
- Compute additional performance metrics: Precision, Recall, and F1-Score.

### Visualization
- Generate and display heatmaps of confusion matrices to visualize model performance after SMOTE balancing.

## How to Run the Project
1. Download the dataset as described in the "Dataset" section.
2. Ensure the dataset is located in the `data/` folder.
3. Install the required libraries:
   ```sh
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
   ```
4. Run the script:
   ```sh
   python DataMining_Final.py
   ```

## Outputs
Results from the final version are saved in `outputs/final-results.txt` and include:
- **Initial Dataset Analysis:** Accuracy scores and confusion matrices for SVM and KNN.
- **Post-Cleaning Analysis:** Model performance after handling missing values.
- **Balanced Dataset Analysis:** Accuracy, Precision, Recall, and F1-Scores after SMOTE application.
- **Heatmaps:** Visual representation of confusion matrices.

## Example Metrics (Balanced Dataset)
### SVM Classifier
- Accuracy: 0.695
- Precision: 0.562
- Recall: 0.675
- F1-Score: 0.615

### KNN Classifier
- Accuracy: 0.640
- Precision: 0.495
- Recall: 0.730
- F1-Score: 0.590

## Limitations
- Hyperparameter tuning is only partially implemented.
- Feature importance analysis is not conducted.
- Advanced visualizations like ROC curves are not included.

## Future Improvements
- Implement more thorough hyperparameter tuning to optimize model performance.
- Conduct feature importance analysis to better understand variable significance.
- Include additional visualization techniques such as ROC and precision-recall curves.


