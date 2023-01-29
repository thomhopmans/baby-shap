""" Unit tests for the Linear explainer.
"""

import numpy as np
import pytest
import scipy
from sklearn import datasets, linear_model

import baby_shap
from baby_shap import maskers


def test_tied_pair():
    np.random.seed(0)
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = baby_shap.LinearExplainer(
        (beta, 0), (mu, Sigma), feature_perturbation="correlation_dependent"
    )
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05


def test_tied_pair_independent():
    np.random.seed(0)
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = baby_shap.LinearExplainer(
        (beta, 0), (mu, Sigma), feature_perturbation="interventional"
    )
    assert np.abs(explainer.shap_values(X) - np.array([1, 0, 0])).max() < 0.05


def test_tied_pair_new():
    np.random.seed(0)
    beta = np.array([1, 0, 0])
    mu = np.zeros(3)
    Sigma = np.array([[1, 0.999999, 0], [0.999999, 1, 0], [0, 0, 1]])
    X = np.ones((1, 3))
    explainer = baby_shap.LinearExplainer(
        (beta, 0), baby_shap.maskers.Impute({"mean": mu, "cov": Sigma})
    )
    assert np.abs(explainer.shap_values(X) - np.array([0.5, 0.5, 0])).max() < 0.05


def test_tied_triple():
    np.random.seed(0)
    beta = np.array([0, 1, 0, 0])
    mu = 1 * np.ones(4)
    Sigma = np.array(
        [
            [1, 0.999999, 0.999999, 0],
            [0.999999, 1, 0.999999, 0],
            [0.999999, 0.999999, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    X = 2 * np.ones((1, 4))
    explainer = baby_shap.LinearExplainer(
        (beta, 0), (mu, Sigma), feature_perturbation="correlation_dependent"
    )
    assert explainer.expected_value == 1
    assert (
        np.abs(
            explainer.shap_values(X) - np.array([0.33333, 0.33333, 0.33333, 0])
        ).max()
        < 0.05
    )


def test_sklearn_linear(california):
    np.random.seed(0)
    Ridge = linear_model.Ridge

    # train linear model
    X, y = california
    X = X[:100]
    y = y[:100]
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_sklearn_linear_old_style(california):
    np.random.seed(0)
    Ridge = linear_model.Ridge

    # train linear model
    X, y = california
    X = X[:100]
    y = y[:100]
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X, feature_perturbation="interventional")
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_sklearn_linear_new(california):
    np.random.seed(0)
    Ridge = pytest.importorskip("sklearn.linear_model").Ridge

    # train linear model
    X, y = california
    X = X[:100]
    y = y[:100]
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X)
    shap_values = explainer(X)
    assert (
        np.abs(
            shap_values.values.sum(1) + shap_values.base_values - model.predict(X)
        ).max()
        < 1e-6
    )
    assert np.abs(shap_values.base_values[0] - model.predict(X).mean()) < 1e-6


def test_sklearn_multiclass_no_intercept(california):
    np.random.seed(0)

    # train linear model
    X, y = california
    X = X[:100]
    y = y[:100]

    # make y multiclass
    multiclass_y = np.expand_dims(y, axis=-1)
    model = linear_model.Ridge(fit_intercept=False)
    model.fit(X, multiclass_y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    explainer.shap_values(X)


def test_perfect_colinear(california):
    X, y = california

    X = X[:100].copy()
    y = y[:100].copy()

    # breakpoint()
    X = X.assign(MedInc=lambda d: d["Population"]) # test duplicated features
    X = X.assign(AveOccup=lambda d: d["Latitude"] - d["Latitude"])  # test multiple colinear features
    X = X.assign(AveBedrms=lambda d: 0) # test null features

    model = linear_model.LinearRegression()
    model.fit(X, y)
    explainer = baby_shap.LinearExplainer(model, masker=maskers.Impute(X))
    shap_values = explainer.shap_values(X)
    assert (
        np.abs(shap_values.sum(1) - model.predict(X) + model.predict(X).mean()).sum()
        < 1e-7
    )


def test_shape_values_linear_many_features():
    np.random.seed(0)

    coef = np.array([1, 2]).T

    # generate linear data
    X = np.random.normal(1, 10, size=(1000, len(coef)))
    y = np.dot(X, coef) + 1 + np.random.normal(scale=0.1, size=1000)

    # train linear model
    model = linear_model.Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X.mean(0).reshape(1, -1))

    values = explainer.shap_values(X)

    assert values.shape == (1000, 2)

    expected = (X - X.mean(0)) * coef
    np.testing.assert_allclose(expected - values, 0, atol=0.01)


def test_single_feature():
    """Make sure things work with a univariate linear regression."""
    Ridge = linear_model.Ridge

    np.random.seed(0)

    # generate linear data
    X = np.random.normal(1, 10, size=(100, 1))
    y = 2 * X[:, 0] + 1 + np.random.normal(scale=0.1, size=100)

    # train linear model
    model = Ridge(0.1)
    model.fit(X, y)

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert np.abs(explainer.expected_value - model.predict(X).mean()) < 1e-6
    assert (
        np.max(np.abs(explainer.expected_value + shap_values.sum(1) - model.predict(X)))
        < 1e-6
    )


def test_sparse():
    """Validate running LinearExplainer on scipy sparse data"""
    # from scipy.special import expit
    LogisticRegression = linear_model.LogisticRegression

    np.random.seed(0)
    n_features = 20
    X, y = datasets.make_multilabel_classification(
        n_samples=100, sparse=True, n_features=n_features, n_classes=1, n_labels=2
    )

    # train linear model
    model = LogisticRegression()
    model.fit(X, y.ravel())

    # explain the model's predictions using SHAP values
    explainer = baby_shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    assert (
        np.max(
            np.abs(
                scipy.special.expit(explainer.expected_value + shap_values.sum(1))
                - model.predict_proba(X)[:, 1]
            )
        )
        < 1e-6
    )


@pytest.mark.parametrize(
    "feature_perturbation,masker",
    [
        (None, baby_shap.maskers.Independent),
        ("interventional", baby_shap.maskers.Independent),
        ("correlation_dependent", baby_shap.maskers.Impute),
    ],
)
def test_feature_perturbation_sets_correct_masker(
    california, feature_perturbation, masker
):
    np.random.seed(0)
    Ridge = linear_model.Ridge

    # train linear model
    X, y = california
    X = X[:100]
    y = y[:100]
    model = Ridge(0.1)
    model.fit(X, y)

    explainer = baby_shap.LinearExplainer(
        model, X, feature_perturbation=feature_perturbation
    )
    assert isinstance(explainer.masker, masker)
