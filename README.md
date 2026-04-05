# Diabetes Prediction using SVM and KNN

A classification project . The goal is to predict whether a patient has diabetes based on medical attributes using Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classifiers.

## Overview

This project walks through a complete machine learning pipeline:
1. **Exploratory analysis** of the Pima Indians Diabetes dataset
2. **Data cleaning** — replacing invalid zero values with column means
3. **Feature scaling** using StandardScaler to normalize the data
4. **Class balancing** using SMOTE to handle the imbalanced dataset
5. **Model evaluation** with accuracy, precision, recall, and F1-score

## Dataset

The [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) from Kaggle contains 768 patient records with 8 medical features (glucose level, blood pressure, BMI, etc.) and a binary outcome indicating diabetes diagnosis.

The dataset is included in the `DataSet/` folder.

## Results

| Model | Stage | Accuracy |
|-------|-------|----------|
| SVM | Original | 73.38% |
| SVM | Cleaned + Scaled | **75.32%** |
| SVM | After SMOTE | 73.38% |
| KNN | Original | 69.48% |
| KNN | Cleaned + Scaled | **74.68%** |
| KNN | After SMOTE | 68.18% |

After applying SMOTE, overall accuracy decreased slightly, but **recall for diabetic patients improved significantly** (SVM: 74.5%, KNN: 81.8%), meaning the models catch more actual positive cases — which matters more in a medical context.

## How to Run

1. Install dependencies:
```
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

2. Run the script from the project root:
```
python Code/DataMining_Final.py
```

## Project Structure
```
Data-Mining-Project/
    Code/
        DataMining_Final.py    # main script
    DataSet/
        diabetes.csv           # dataset
    Output/
        results.txt            # model evaluation results
    LICENSE
    README.md
```

## Key Takeaways
- Feature scaling is essential for distance-based models like SVM and KNN
- SMOTE improves recall for the minority class at the cost of some overall accuracy
- Evaluating models with precision, recall, and F1-score gives a better picture than accuracy alone

## Technologies
- Python 3
- scikit-learn (SVM, KNN, StandardScaler)
- imbalanced-learn (SMOTE)
- seaborn & matplotlib (visualization)
- pandas & numpy

## License
[MIT](LICENSE)
