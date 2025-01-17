from sklearn import svm

import baby_shap
from baby_shap import explainers


def test_random_force_plot_mpl_with_data(iris_dataset):
    """Test if force plot with matplotlib works."""

    model = svm.SVC(probability=True)

    # train model
    X, y = iris_dataset
    model.fit(X.to_numpy(), y)

    # explain the model's predictions using SHAP values
    explainer = explainers.KernelExplainer(model=model.predict_proba, data=X)
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explaination
    baby_shap.force_plot(explainer.expected_value[0], shap_values[0], show=False)
