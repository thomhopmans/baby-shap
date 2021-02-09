""" Unit tests for the Permutation explainer.
"""

# pylint: disable=missing-function-docstring
import numpy as np
import pytest
import shap

def test_single_class_independent():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Permutation(model.predict, X)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)

def test_single_class_independent_auto_api():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.Explainer(model.predict, X, algorithm="permutation")
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)

def test_multi_class_independent():
    xgboost = pytest.importorskip('xgboost')
    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    explainer = shap.explainers.Permutation(model.predict_proba, X)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(X[:100])) < 1e6)


def test_single_class_partition():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.Permutation(model.predict, masker)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict(X[:100])) < 1e6)

def test_multi_class_partition():
    xgboost = pytest.importorskip('xgboost')

    # get a dataset on income prediction
    X, y = shap.datasets.adult()
    X = X.iloc[:100]
    y = y[:100]

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    # build an Exact explainer and explain the model predictions on the given dataset
    masker = shap.maskers.Partition(X)
    explainer = shap.explainers.Permutation(model.predict_proba, masker)
    shap_values = explainer(X)

    assert np.max(np.abs(shap_values.base_values + shap_values.values.sum(1) - model.predict_proba(X[:100])) < 1e6)


def test_serialization_permutation():
    import shap
    import xgboost
    import pickle
    import numpy as np
    import tempfile

    # get a dataset on income prediction
    X,y = shap.datasets.adult()

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])

    temp_serialization_file = tempfile.TemporaryFile()
    # Serialization 
    explainer_original.save(temp_serialization_file)
    
    temp_serialization_file.seek(0)

    # Deserialization
    explainer_new = shap.Explainer.load(temp_serialization_file)

    temp_serialization_file.close()

    shap_values_new = explainer_new(X[:1])

    for i in range(len(explainer_original.masker.feature_names)):
        assert explainer_original.masker.feature_names[i] == explainer_new.masker.feature_names[i] 

    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)

def test_serialization_permutation_no_model_or_masker():
    import shap
    import xgboost
    import pickle
    import numpy as np
    import tempfile

    # get a dataset on income prediction
    X,y = shap.datasets.adult()

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])

    temp_serialization_file = tempfile.TemporaryFile()

    # Serialization 
    explainer_original.model.save = None
    explainer_original.masker.save = None
    explainer_original.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # Deserialization
    explainer_new = shap.Explainer.load(temp_serialization_file)

    temp_serialization_file.close()
    
    # manually insert model and masker
    explainer_new.model = explainer_original.model
    explainer_new.masker = explainer_original.masker

    shap_values_new = explainer_new(X[:1])

    for i in range(len(explainer_original.masker.feature_names)):
        assert explainer_original.masker.feature_names[i] == explainer_new.masker.feature_names[i] 

    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)

def test_serialization_permutation_numpy_custom_model_save():
    import shap
    import xgboost
    import pickle
    import numpy as np
    import tempfile

    # get a dataset on income prediction
    X,y = shap.datasets.adult()
    X = X.values

    # train an XGBoost model (but any other model type would also work)
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    explainer_original = shap.Explainer(model.predict_proba, X, algorithm='permutation')
    shap_values_original = explainer_original(X[:1])

    temp_serialization_file = tempfile.TemporaryFile()

    # Serialization 
    explainer_original.model.save = lambda out_file, model: pickle.dump(model, out_file)
    explainer_original.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # Deserialization
    model_loader = lambda in_file: pickle.load(in_file)
    explainer_new = shap.Explainer.load(temp_serialization_file, model_loader = model_loader)

    temp_serialization_file.close()


    shap_values_new = explainer_new(X[:1])

    assert (getattr(explainer_original.masker, "feature_names", None) == None) and (getattr(explainer_original.masker, "feature_names", None) == None)
    assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
    assert type(explainer_original) == type(explainer_new)
    assert type(explainer_original.masker) == type(explainer_new.masker)