import pandas as pd
from pandas import DataFrame
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data() -> tuple[DataFrame, DataFrame]:
    # Load the dataset
    # df = pd.read_csv('online_shoppers_intention.csv')
    df = fetch_ucirepo(id=468)

    # Separate features and target
    X: DataFrame = df.data.features
    y: DataFrame = df.data.targets

    # Convert categorical variables to numeric
    X = pd.get_dummies(X, columns=['Month', 'VisitorType', 'Weekend'])

    return X, y

def explore_data(X, y):
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

    # Shopping behavior stats
    print("\nShopping Behavior Statistics:")
    print(f"Average pages per session: {X['PageValues'].mean():.2f}")
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

def create_model(X, y) -> DataFrame:
    # convert to 1D array to avoid warning
    y_1d = y['Revenue'].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_1d, test_size=0.2, random_state=42
    )

    model: LogisticRegression = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    train_score: float = model.score(X_train, y_train)
    test_score: float = model.score(X_test, y_test)

    print("\nModel Performance:")
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Testing accuracy: {test_score:.3f}")

    feature_importance: DataFrame = pd.DataFrame({
        'Feature': X.columns,
        'Importance': abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)

    print("\nRecommendations for Retailers:")
    for _, row in feature_importance.head().iterrows():
        print(f"- Focus on optimizing {row['Feature'].replace('_', ' ')}: Impact score {row['Importance']:.3f}")

    return feature_importance

def analyze_data() -> DataFrame:
    # Load and prepare data

    X, y = load_data()

    # Explore the data
    explore_data(X, y)

    # Conversion rate analysis to see if there are any trends
    # print("\nConversion Rate Analysis:")
    # for month in X.filter(like='Month_').columns:
    #     month_conversion = y[X[month] == 1]['Revenue'].mean()
    #     print(f"{month.replace('Month_', '')}: {month_conversion:.2%}")

    # Create and evaluate model
    feature_importance: DataFrame = create_model(X, y)

    print("\nTop 5 Most Important Features:")
    print(feature_importance.head())

    return feature_importance

def plot_feature_importance(feature_importance: DataFrame, output_file: str, format: str):
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
    plt.title('Top 10 Most Important Features for Purchase Prediction')
    plt.xlabel('Absolute Coefficient Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, format=f"{format}", dpi=200)
    plt.close()

def main():
    try:
        os.mkdir("plots")
    except FileExistsError:
        print("Error: \"plots\" directory already exists. Please clean up manually")
        exit(1)

    print("Online Shopping Purchase Prediction Analysis")
    print("============================================")

    # Analyze data
    feature_importance = analyze_data()

    # Visualize results
    format: str = "svg"
    output_file: str = f"plots/feature_importance.{format}"
    plot_feature_importance(feature_importance, output_file, format)

if __name__ == "__main__":
    main()
