# Imports
import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"Titanic-Survival-Prediction\titanic.csv")
print("Initial data info:")
data.info()
print("\nMissing values:")
print(data.isnull().sum())

# Data Cleaning and Feature Engineering                                                            
def preprocess_data(df):
    df = df.copy()
    
    # Drop columns with too many missing values or not useful for prediction
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)
    
    # Handle missing values in Embarked
    df["Embarked"].fillna("S", inplace=True)
    
    # Convert categorical variables
    df["Sex"] = df["Sex"].map({'male': 1, "female": 0})
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)
    
    # Handle missing values in Age and Fare
    df = fill_missing_ages(df)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    
    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    
    # Check for any remaining missing values
    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())
    
    return df

# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    return df

data = preprocess_data(data)

# Create Features / Target Variables
X = data.drop(columns=["Survived"])
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

# ML Preprocessing
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning - KNN
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors": list(range(1, 21)),
        "metric": ["euclidean", "manhattan"],
        "weights": ["uniform", "distance"]
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)

# Predictions and evaluate
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
print("Confusion Matrix:")
print(matrix)

# Optional: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Check feature importance by looking at correlation with target
plt.figure(figsize=(10, 8))
correlation = data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.show()