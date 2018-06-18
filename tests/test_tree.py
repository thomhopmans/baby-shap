import matplotlib
import numpy as np

matplotlib.use('Agg')
import shap


def test_front_page_xgboost():
    import xgboost
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train XGBoost model
    X, y = shap.datasets.boston()
    model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

def test_front_page_sklearn():
    import sklearn.ensemble
    import shap

    # load JS visualization code to notebook
    shap.initjs()

    # train model
    X, y = shap.datasets.boston()
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # visualize the first prediction's explaination
    shap.force_plot(shap_values[0, :], X.iloc[0, :])

    # visualize the training set predictions
    shap.force_plot(shap_values, X)

    # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
    shap.dependence_plot(5, shap_values, X, show=False)
    shap.dependence_plot("RM", shap_values, X, show=False)

    # summarize the effects of all the features
    shap.summary_plot(shap_values, X, show=False)

def test_xgboost_multiclass():
    import xgboost
    import shap

    # train XGBoost model
    X, Y = shap.datasets.iris()
    model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4)
    model.fit(X, Y)

    # explain the model's predictions using SHAP values (use pred_contrib in LightGBM)
    shap_values = shap.TreeExplainer(model).shap_values(X)

    # ensure plot works for first class
    shap.dependence_plot(0, shap_values[0], X, show=False)

def test_mixed_types():
    import xgboost
    import shap
    import numpy as np

    X,y = shap.datasets.boston()
    X["LSTAT"] = X["LSTAT"].astype(np.int64)
    X["B"] = X["B"].astype(np.bool)
    bst = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 1000)
    shap_values = shap.TreeExplainer(bst).shap_values(X)
    shap.dependence_plot(0, shap_values, X, show=False)

def test_sklearn_multiclass():
    import shap
    from sklearn.ensemble import RandomForestClassifier

    X, y = shap.datasets.iris()
    y[y == 2] = 1
    model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    assert np.abs(shap_values[0][0,0] - 0.05) < 1e-3
    assert np.abs(shap_values[1][0,0] + 0.05) < 1e-3

# def test_lightgbm():
#     import lightgbm
#     import shap
#
#     # train XGBoost model
#     X, y = shap.datasets.boston()
#     model = lightgbm.sklearn.LGBMRegressor()
#     model.fit(X, y)
#
#     # explain the model's predictions using SHAP values
#     shap_values = shap.TreeExplainer(model).shap_values(X)
#
# def test_lightgbm_multiclass():
#     import lightgbm
#     import shap
#
#     # train XGBoost model
#     X, Y = shap.datasets.iris()
#     model = lightgbm.sklearn.LGBMClassifier()
#     model.fit(X, Y)
#
#     # explain the model's predictions using SHAP values
#     shap_values = shap.TreeExplainer(model).shap_values(X)
#
#     # ensure plot works for first class
#     shap.dependence_plot(0, shap_values[0], X, show=False)
