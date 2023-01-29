

<p align="center">
  <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.svg" width="800" />
</p>

---
![example workflow](https://github.com/thomhopmans/baby-shap/actions/workflows/run_tests.yml/badge.svg)

Baby Shap is a stripped and opiniated version of **SHAP (SHapley Additive exPlanations)**, a game theoretic approach to explain the output of any machine learning model by Scott Lundberg. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions (see [papers](#citations) for details and citations). 

**Baby Shap solely implements and maintains the Linear and Kernel Explainer and a limited range of plots, while limiting the number of dependencies, conflicts and raised warnings and errors.**

## Install

Baby SHAP can be installed from either [PyPI](https://pypi.org/project/baby-shap):

<pre>
pip install baby-shap
</pre>

## Model agnostic example with KernelExplainer (explains any function)

Kernel SHAP uses a specially-weighted local linear regression to estimate SHAP values for any model. Below is a simple example for explaining a multi-class SVM on the classic iris dataset.

```python
import baby_shap
from sklearn import datasets, svm, model_selection

# print the JS visualization code to the notebook
baby_shap.initjs()

# train a SVM classifier
d = datasets.load_iris()
X = pd.DataFrame(data=d.data, columns=d.feature_names)
y = d.target

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
clf = svm.SVC(kernel='rbf', probability=True)
clf.fit(X_train.to_numpy(), Y_train)

# use Kernel SHAP to explain test set predictions
explainer = baby_shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance
baby_shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")
```
<p align="center">
  <img width="810" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_instance.png" />
</p>

The above explanation shows four features each contributing to push the model output from the base value (the average model output over the training dataset we passed) towards zero. If there were any features pushing the class label higher they would be shown in red.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset. This is exactly what we do below for all the examples in the iris test set:

```python
# plot the SHAP values for the Setosa output of all instances
baby_shap.force_plot(explainer.expected_value[0], shap_values[0], X_test, link="logit")
```
<p align="center">
  <img width="813" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_dataset.png" />
</p>
