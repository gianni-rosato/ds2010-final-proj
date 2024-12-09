import pandas as pd
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

'''
The program first explores & analyzes the dataset by examining visitor types,
shopping behavior statistics, and seasonal patterns. It processes categorical
variables through one-hot encoding via Pandas and handles the data preparation
for machine learning. Then, it creates and trains a Logistic Regression model
via SciKit to predict whether a visitor will make a purchase, using standardized
features and performing both train-test splitting and cross-validation to
evaluate the model's performance.
'''

def load_data() -> tuple:
    '''
    Load the online shopping dataset from the UCI repository and prepare it for analysis.
    '''
    df = fetch_ucirepo(id=468)
    X: DataFrame = df.data.features
    y: DataFrame = df.data.targets

    # Store original visitor type before encoding
    visitor_types = X['VisitorType'].copy()

    # Convert categorical variables to numeric
    X = pd.get_dummies(X, columns=['Month', 'VisitorType', 'Weekend'])

    return X, y, visitor_types

def analyze_visitor_types(visitor_types, y):
    '''
    Analyze the conversion rates of different visitor types.
    '''
    print("\nVisitor Type Analysis:")
    visitor_conversion = pd.DataFrame({
        'VisitorType': visitor_types,
        'Purchased': y['Revenue']
    }).groupby('VisitorType').agg({
        'Purchased': ['count', 'mean']
    })
    visitor_conversion.columns = ['Total Sessions', 'Conversion Rate']
    print(visitor_conversion)

def explore_data(X, y, visitor_types):
    '''
    Explore the dataset and analyze visitor types, shopping behavior statistics, & seasonal patterns.
    '''
    print("\nDataset Overview:")
    print(f"Number of sessions: {len(X)}")
    print("\nFeature Statistics:")
    print(X.describe())

    # Check for missing values
    missing_values = X.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print("\nNo missing values found in the dataset.")

    # Show class distribution
    print("\nPurchase Distribution:")
    print(y.value_counts(normalize=True))

    # Analyze visitor types
    analyze_visitor_types(visitor_types, y)

    # Shopping behavior stats
    print("\nShopping Behavior Statistics:")
    print(f"Average page value: {X['PageValues'].mean():.2f}")
    print(f"Average time spent: {X['Administrative_Duration'].mean() + X['Informational_Duration'].mean() + X['ProductRelated_Duration'].mean():.2f} seconds")
    print(f"Purchase rate: {(y['Revenue'] == 1).mean():.2%}")

    # Add seasonal purchase patterns
    monthly_cols = X.filter(like='Month_').columns
    monthly_conversion: DataFrame = pd.DataFrame([
        (col.replace('Month_', ''),
            y[X[col] == 1]['Revenue'].mean())
        for col in monthly_cols
    ], columns=['Month', 'Conversion Rate'])
    print("\nSeasonal Purchase Patterns:")
    print(monthly_conversion.sort_values('Conversion Rate', ascending=False))

def create_model(X, y) -> tuple[DataFrame, dict]:
    '''
    Create a Logistic Regression model to predict whether a visitor will make a purchase or not.
    '''
    # Convert to 1D array
    y_1d = y['Revenue'].to_numpy()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_1d, test_size=0.2, random_state=42
    )

    # Train
    model: LogisticRegression = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # model performance
    train_score: float = model.score(X_train, y_train)
    test_score: float = model.score(X_test, y_test)

    print("\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y_1d, cv=5)
    print("\nCross-validation scores:")
    print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # confusion matrix and classification report
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # feature importance
    feature_importance: DataFrame = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)

    print("\nRecommendations for Retailers:")
    for _, row in feature_importance.head().iterrows():
        print(f"- Focus on optimizing {row['Feature'].replace('_', ' ')}: Impact score {row['Importance']:.3f}")

    return feature_importance, {
        'confusion_matrix': conf_matrix,
        'cv_scores': cv_scores
    }

def plot_feature_importance(feature_importance: DataFrame, output_file: str, format: str):
    '''
    Plot the top 10 most important features for purchase prediction.
    '''
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features for Purchase Prediction')
    plt.xlabel('Absolute Coefficient Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_file}.{format}", format=f"{format}", dpi=200)
    plt.close()

def plot_confusion_matrix(conf_matrix, output_file: str, format: str):
    '''
    Plot the confusion matrix for the model.
    '''
    plt.figure(figsize=(8, 6))
    plt.style.use('dark_background')
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"{output_file}.{format}", format=format, dpi=200)
    plt.close()

def plot_seasonal_patterns(X, y, output_file: str, format: str):
    '''
    Plot the monthly conversion rates for the dataset.
    '''
    monthly_cols = X.filter(like='Month_').columns
    monthly_conversion = pd.DataFrame([
        (col.replace('Month_', ''),
            y[X[col] == 1]['Revenue'].mean())
        for col in monthly_cols
    ], columns=['Month', 'Conversion Rate'])

    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    sns.barplot(data=monthly_conversion, x='Month', y='Conversion Rate')
    plt.title('Monthly Conversion Rates')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_file}.{format}", format=format, dpi=200)
    plt.close()

def main():
    try:
        os.mkdir("plots")
    except FileExistsError:
        print("Error: \"plots\" directory already exists. Please clean up manually")
        exit(1)

    print("Online Shopping Purchase Prediction Analysis")
    print("============================================")

    X, y, visitor_types = load_data()
    explore_data(X, y, visitor_types)
    feature_importance, model_metrics = create_model(X, y)

    # Plots
    format: str = "png"
    plot_feature_importance(feature_importance, "plots/feature_importance", format)
    plot_confusion_matrix(model_metrics['confusion_matrix'], "plots/confusion_matrix", format)
    plot_seasonal_patterns(X, y, "plots/seasonal_patterns", format)

if __name__ == "__main__":
    main()
