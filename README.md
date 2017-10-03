<p align="center">
  <img src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png" width="400" />
</p>

---

**SHAP (SHapley Additive exPlanations)** explains the output of any machine learning model using expectations and Shapley values. It is the only possible consistent and locally accurate additive feature attribution method based on expectations (see [paper](https://arxiv.org/abs/1705.07874) for details).

## Install

```
pip install shap
```

## LightGBM example

While SHAP values can explain the output of any machine learning model, we have developed high-speed exact algorithms for ensemble tree methods ([paper](https://arxiv.org/abs/1706.06060)). These have been integrated directly into LightGBM, and you can use the `shap` package for visualization in a Jupyter notebook:

```python
import lightgbm
import sklearn.datasets
import shap

# load JS visualization code to notebook
shap.initjs() 

# train LightGBM model
d = sklearn.datasets.load_boston()
bst = lightgbm.train({"learning_rate": 0.01}, lightgbm.Dataset(d.data, label=d.target), 10)

# explain the model's prediction using SHAP values on the first 1000 training data samples
shap_values = bst.predict(d.data[0:1000,:], pred_contrib=True)

# visualize the first prediction's explaination
shap.visualize(shap_values[0,:], feature_names=d.feature_names, data=d.data[0,:])
```

<p align="center">
  <img width="811" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_instance.png" />
</p>

The above explanation shows features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to the model output. Features pushing the prediction higher are shown in red, those pushing the prediction lower are in blue.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset:

```python
# visualize the first 1000 predictions
shap.visualize(shap_values[:1000,:], feature_names=d.feature_names, data=d.data[:1000,:])
```

<p align="center">
  <img width="811" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_dataset.png" />
</p>

To understand how a single feature effects the output of the model we can plot the SHAP value of that feature vs. the value of the feature for all the examples in the training dataset. Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in predicted house price as the average number of rooms per house in an area changes.

```python
# create a SHAP plot to show the effect of a single feature across the whole dataset
shap.plot(d.data[:,5], shap_values[:,5], "Avg. # rooms in home")
```

<p align="center">
  <img width="544" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_shap_plot.png" />
</p>


## Model agnostic example

```python
from shap import KernelExplainer, DenseData, visualize, initjs
from sklearn import datasets,neighbors
from numpy import random, arange

# print the JS visualization code to the notebook
initjs()

# train a k-nearest neighbors classifier on a random subset 
iris = datasets.load_iris()
random.seed(2)
inds = arange(len(iris.target))
random.shuffle(inds)
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target == 0)

# use Shap to explain a single prediction
background = DenseData(iris.data[inds[:100],:], iris.feature_names) # name the features
explainer = KernelExplainer(knn.predict, background, nsamples=100)
x = iris.data[inds[102:103],:]
visualize(explainer.explain(x))
```
<p align="center">
  <img width="810" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_instance.png" />
</p>

The above explanation shows three features each contributing to push the model output from the base value (the average model output over the training dataset we passed) to zero. If there were any features pushing the class label higher they would be shown in red.

If we take many explanations such as the one shown above, rotate them 90 degrees, and then stack them horizontally, we can see explanations for an entire dataset. This is exactly what we do below for all the examples in the iris test set:

```python
# use Shap to explain all test set predictions
visualize([explainer.explain(iris.data[inds[i:i+1],:]) for i in range(100,len(iris.target))])
```
<p align="center">
  <img width="813" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/iris_dataset.png" />
</p>

## Sample notebooks

The notebooks below demonstrate different use cases for Shap. Look inside the notebooks directory of the repository if you want to try playing with the original notebooks yourself. If you have your own notebook you would like to share, or have an improvement to the notebooks below, pull requests are welcome :)

- [**Iris classification**](https://rawgit.com/slundberg/shap/master/notebooks/python/Iris%20classification.html) - A basic demonstration using the popular iris species dataset. It explains predictions from six different models in scikit-learn using Shap.
- [**Census income classification**](https://slundberg.github.io/shap/notebooks/python/Census%20income%20classification.html) - Using the standard adult census income dataset, this notebook trains a random forest classifier using scikit-learn and then explains predictions using Shap.
