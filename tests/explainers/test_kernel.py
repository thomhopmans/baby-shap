import pathlib

import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn import datasets, linear_model, pipeline, preprocessing

import baby_shap
from baby_shap import explainers

A1A_PATH = pathlib.Path(__file__).parent.resolve() / ".." / "datasets" / "a1a.svmlight"


def a1a():
    """A sparse dataset in scipy csr matrix format."""
    return datasets.load_svmlight_file(A1A_PATH)


def load_iris(display=False, sample=None):
    """Return the classic iris data in a nice package."""

    d = datasets.load_iris()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    df["target"] = d.target

    if sample:
        df = df.sample(n=sample, random_state=42)

    if display:
        return df, [d.target_names[v] for v in d.target]

    target = df["target"].copy()
    df = df.drop(columns=["target"])
    return df, target


def test_null_model_small():
    """Test a small null model."""
    explainer = explainers.KernelExplainer(
        lambda x: np.zeros(x.shape[0]), np.ones((2, 4)), nsamples=100
    )
    e = explainer.explain(np.ones((1, 4)))
    assert np.sum(np.abs(e)) < 1e-8


def test_null_model():
    """Test a larger null model."""
    explainer = explainers.KernelExplainer(
        lambda x: np.zeros(x.shape[0]), np.ones((2, 10)), nsamples=100
    )
    e = explainer.explain(np.ones((1, 10)))
    assert np.sum(np.abs(e)) < 1e-8


def test_front_page_model_agnostic():
    """Test the ReadMe kernel expainer example."""

    # print the JS visualization code to the notebook
    baby_shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *load_iris(), test_size=0.1, random_state=0
    )
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train.to_numpy(), Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = explainers.KernelExplainer(
        svm.predict_proba, X_train, nsamples=100, link="logit"
    )
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    baby_shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0, :],
        X_test.iloc[0, :],
        link="logit",
    )


def test_front_page_model_agnostic_rank():
    """Test the rank regularized explanation of the ReadMe example."""

    # print the JS visualization code to the notebook
    baby_shap.initjs()

    # train a SVM classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *load_iris(), test_size=0.1, random_state=0
    )
    svm = sklearn.svm.SVC(kernel="rbf", probability=True)
    svm.fit(X_train.to_numpy(), Y_train)

    # use Kernel SHAP to explain test set predictions
    explainer = explainers.KernelExplainer(
        svm.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)"
    )
    shap_values = explainer.shap_values(X_test)

    # plot the SHAP values for the Setosa output of the first instance
    baby_shap.force_plot(
        explainer.expected_value[0],
        shap_values[0][0, :],
        X_test.iloc[0, :],
        link="logit",
    )


def test_kernel_shap_with_dataframe():
    """Test with a Pandas DataFrame."""
    np.random.seed(3)

    df_X = pd.DataFrame(np.random.random((10, 3)), columns=list("abc"))
    df_X.index = pd.date_range("2018-01-01", periods=10, freq="D", tz="UTC")

    df_y = df_X.eval("a - 2 * b + 3 * c")
    df_y = df_y + np.random.normal(0.0, 0.1, df_y.shape)

    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(df_X.to_numpy(), df_y)

    explainer = explainers.KernelExplainer(linear_model.predict, df_X, keep_index=True)
    _ = explainer.shap_values(df_X)


def test_kernel_shap_with_a1a_sparse_zero_background():
    """Test with a sparse matrix for the background."""

    X, y = a1a()
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.01, random_state=0
    )
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)

    _, cols = x_train.shape
    shape = 1, cols
    background = sp.sparse.csr_matrix(shape, dtype=x_train.dtype)
    explainer = explainers.KernelExplainer(linear_model.predict, background)
    explainer.shap_values(x_test)


def test_kernel_shap_with_a1a_sparse_nonzero_background():
    """Check with a sparse non zero background matrix."""
    np.set_printoptions(threshold=100000)
    np.random.seed(0)

    X, y = a1a()
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(
        X, y, test_size=0.01, random_state=0
    )
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)

    # Calculate median of background data
    median_dense = sklearn.utils.sparsefuncs.csc_median_axis_0(x_train.tocsc())
    median = sp.sparse.csr_matrix(median_dense)
    explainer = explainers.KernelExplainer(linear_model.predict, median)
    shap_values = explainer.shap_values(x_test)

    def dense_to_sparse_predict(data):
        sparse_data = sp.sparse.csr_matrix(data)
        return linear_model.predict(sparse_data)

    explainer_dense = explainers.KernelExplainer(
        dense_to_sparse_predict, median_dense.reshape((1, len(median_dense)))
    )
    x_test_dense = x_test.toarray()
    shap_values_dense = explainer_dense.shap_values(x_test_dense)

    # Validate sparse and dense result is the same
    assert np.allclose(shap_values, shap_values_dense, rtol=1e-02, atol=1e-01)


def test_kernel_shap_with_high_dim_sparse():
    """Verifies we can run on very sparse data produced from feature hashing."""

    remove = ("headers", "footers", "quotes")
    categories = [
        "alt.atheism",
        "talk.religion.misc",
        "comp.graphics",
        "sci.space",
    ]
    ngroups = sklearn.datasets.fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=remove,
    )
    x_train, x_test, y_train, _ = sklearn.model_selection.train_test_split(
        ngroups.data, ngroups.target, test_size=0.01, random_state=42
    )
    vectorizer = sklearn.feature_extraction.text.HashingVectorizer(
        stop_words="english", alternate_sign=False, n_features=2**16
    )
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)

    # Fit a linear regression model
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(x_train, y_train)

    _, cols = x_train.shape
    shape = 1, cols
    background = sp.sparse.csr_matrix(shape, dtype=x_train.dtype)

    explainer = explainers.KernelExplainer(linear_model.predict, background)
    _ = explainer.shap_values(x_test)


def test_kernel_sparse_vs_dense_multirow_background():
    """Mix sparse and dense matrix values."""

    # train a logistic regression classifier
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(
        *load_iris(), test_size=0.1, random_state=0
    )
    lr = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr.fit(X_train.to_numpy(), Y_train)

    # use Kernel SHAP to explain test set predictions with dense data
    explainer = explainers.KernelExplainer(
        lr.predict_proba, X_train, nsamples=100, link="logit", l1_reg="rank(3)"
    )
    shap_values = explainer.shap_values(X_test)

    X_sparse_train = sp.sparse.csr_matrix(X_train)
    X_sparse_test = sp.sparse.csr_matrix(X_test)

    lr_sparse = sklearn.linear_model.LogisticRegression(solver="lbfgs")
    lr_sparse.fit(X_sparse_train, Y_train)

    # use Kernel SHAP again but with sparse data
    sparse_explainer = explainers.KernelExplainer(
        lr.predict_proba, X_sparse_train, nsamples=100, link="logit", l1_reg="rank(3)"
    )
    sparse_shap_values = sparse_explainer.shap_values(X_sparse_test)

    assert np.allclose(shap_values, sparse_shap_values, rtol=1e-05, atol=1e-05)

    # Use sparse evaluation examples with dense background
    sparse_sv_dense_bg = explainer.shap_values(X_sparse_test)
    assert np.allclose(shap_values, sparse_sv_dense_bg, rtol=1e-05, atol=1e-05)


def test_linear():
    """Tests that KernelExplainer returns the correct result when the model is linear.

    (as per corollary 1 of https://arxiv.org/abs/1705.07874)
    """

    np.random.seed(2)
    x = np.random.normal(size=(200, 3), scale=1)

    # a linear model
    def f(x):
        return x[:, 0] + 2.0 * x[:, 1]

    phi = explainers.KernelExplainer(f, x).shap_values(
        x, l1_reg="num_features(2)", silent=True
    )
    assert phi.shape == x.shape

    # corollary 1
    expected = (x - x.mean(0)) * np.array([1.0, 2.0, 0.0])

    np.testing.assert_allclose(expected, phi, rtol=1e-3)


def test_non_numeric():
    """Test using non-numeric data."""

    # create dummy data
    X = np.array(
        [
            ["A", "0", "0"],
            ["A", "1", "0"],
            ["B", "0", "0"],
            ["B", "1", "0"],
            ["A", "1", "0"],
        ]
    )
    y = np.array([0, 1, 2, 3, 4])

    # build and train the pipeline
    pipe = pipeline.Pipeline(
        [
            ("oneHotEncoder", preprocessing.OneHotEncoder()),
            ("linear", linear_model.LinearRegression()),
        ]
    )
    pipe.fit(X, y)

    # use KernelExplainer
    explainer = baby_shap.KernelExplainer(pipe.predict, X, nsamples=100)
    shap_values = explainer.explain(X[0, :].reshape(1, -1))

    assert (
        np.abs(
            explainer.expected_value
            + shap_values.sum(0)
            - pipe.predict(X[0, :].reshape(1, -1))[0]
        )
        < 1e-4
    )
    assert shap_values[2] == 0

    # tests for shap.KernelExplainer.not_equal
    assert baby_shap.KernelExplainer.not_equal(0, 0) == baby_shap.KernelExplainer.not_equal(
        "0", "0"
    )
    assert baby_shap.KernelExplainer.not_equal(0, 1) == baby_shap.KernelExplainer.not_equal(
        "0", "1"
    )
    assert baby_shap.KernelExplainer.not_equal(0, np.nan) == baby_shap.KernelExplainer.not_equal(
        "0", np.nan
    )
    assert baby_shap.KernelExplainer.not_equal(0, np.nan) == baby_shap.KernelExplainer.not_equal(
        "0", None
    )
    assert baby_shap.KernelExplainer.not_equal(np.nan, 0) == baby_shap.KernelExplainer.not_equal(
        np.nan, "0"
    )
    assert baby_shap.KernelExplainer.not_equal(np.nan, 0) == baby_shap.KernelExplainer.not_equal(
        None, "0"
    )
    assert baby_shap.KernelExplainer.not_equal("ab", "bc")
    assert not baby_shap.KernelExplainer.not_equal("ab", "ab")
    assert baby_shap.KernelExplainer.not_equal(
        pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T13")
    )
    assert not baby_shap.KernelExplainer.not_equal(
        pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T12")
    )
    assert baby_shap.KernelExplainer.not_equal(
        pd.Timestamp("2017-01-01T12"), pd.Timestamp("2017-01-01T13")
    )
    assert baby_shap.KernelExplainer.not_equal(pd.Period("4Q2005"), pd.Period("3Q2005"))
    assert not baby_shap.KernelExplainer.not_equal(pd.Period("4Q2005"), pd.Period("4Q2005"))
