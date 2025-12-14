"""
Tests for data utilities.
"""

import pytest
import numpy as np
import pandas as pd
import torch

from ml_toolkit.data import DataLoader, BasePreprocessor, CustomDataset


def test_dataloader_csv(tmp_path):
    """Test CSV loading."""
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(csv_file, index=False)

    # Load with DataLoader
    loaded_df = DataLoader.load_csv(csv_file)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_dataloader_numpy(tmp_path):
    """Test NumPy loading."""
    # Create a temporary numpy file
    npy_file = tmp_path / "test.npy"
    arr = np.array([1, 2, 3, 4, 5])
    np.save(npy_file, arr)

    # Load with DataLoader
    loaded_arr = DataLoader.load_numpy(npy_file)
    np.testing.assert_array_equal(arr, loaded_arr)


def test_base_preprocessor():
    """Test BasePreprocessor."""
    data = np.array([1, 2, 3, 4, 5])

    # Define transforms
    def add_one(x):
        return x + 1

    def multiply_two(x):
        return x * 2

    # Create preprocessor
    preprocessor = BasePreprocessor()
    preprocessor.add_step("add_one", add_one)
    preprocessor.add_step("multiply_two", multiply_two)

    # Transform
    result = preprocessor.transform(data)
    expected = (data + 1) * 2
    np.testing.assert_array_equal(result, expected)


def test_custom_dataset():
    """Test CustomDataset."""
    data = torch.randn(10, 5)
    labels = torch.randint(0, 3, (10,))

    dataset = CustomDataset(data, labels)

    assert len(dataset) == 10

    item = dataset[0]
    assert "data" in item
    assert "label" in item
    assert torch.equal(item["data"], data[0])
    assert torch.equal(item["label"], labels[0])


def test_custom_dataset_with_transform():
    """Test CustomDataset with transform."""
    data = torch.randn(10, 5)
    labels = torch.randint(0, 3, (10,))

    def transform(x):
        return x * 2

    dataset = CustomDataset(data, labels, transform=transform)

    item = dataset[0]
    expected = data[0] * 2
    assert torch.equal(item["data"], expected)
