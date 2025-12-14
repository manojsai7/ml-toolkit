"""
Example demonstrating data preprocessing pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_toolkit.data import DataLoader, BasePreprocessor


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    return (data - data.min()) / (data.max() - data.min())


def remove_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers using z-score."""
    z_scores = np.abs((data - data.mean()) / data.std())
    return data[z_scores < threshold]


def main():
    print("Data Preprocessing Example\n")

    # Generate sample data
    data = np.random.randn(1000, 10)
    print(f"Original data shape: {data.shape}")
    print(f"Original data stats: mean={data.mean():.2f}, std={data.std():.2f}")

    # Create preprocessing pipeline
    preprocessor = BasePreprocessor()
    preprocessor.add_step("normalize", normalize)

    # Apply preprocessing
    processed_data = preprocessor.transform(data)
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Processed data stats: mean={processed_data.mean():.2f}, std={processed_data.std():.2f}")

    # Example with pandas DataFrame
    print("\n" + "=" * 50)
    print("DataFrame preprocessing example")

    df = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100) * 10,
            "feature3": np.random.randint(0, 100, 100),
        }
    )

    print("\nOriginal DataFrame:")
    print(df.describe())

    # Custom transform for DataFrame
    def standardize_df(df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[df.columns] = scaler.fit_transform(df)
        return df_scaled

    df_preprocessor = BasePreprocessor()
    df_preprocessor.add_step("standardize", standardize_df)

    df_processed = df_preprocessor.transform(df)
    print("\nProcessed DataFrame:")
    print(df_processed.describe())


if __name__ == "__main__":
    main()
