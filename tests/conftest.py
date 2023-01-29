import pandas as pd
import pytest
from sklearn import datasets


@pytest.fixture(scope="function")
def iris_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return the classic iris data in a nice package."""

    d = datasets.load_iris()
    df = (
        pd.DataFrame(data=d.data, columns=d.feature_names)
        .assign(target=d.target)
        .sample(n=20, random_state=42)
    )

    y = df["target"].copy()
    X = df.drop(columns=["target"])

    return X, y


@pytest.fixture(scope="function")
def california() -> tuple[pd.DataFrame, pd.Series]:
    """Return the california housing data in a nice package."""

    d = datasets.fetch_california_housing()
    df = pd.DataFrame(data=d.data, columns=d.feature_names).assign(target=d.target)
    y = df["target"].copy()
    X = df.drop(columns=["target"])

    return X, y
