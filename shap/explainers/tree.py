import numpy as np
import multiprocessing
import sys
from .explainer import Explainer

have_cext = False
try:
    from .. import _cext
    have_cext = True
except ImportError:
    pass
except:
    print("the C extension is installed...but failed to load!")
    pass

try:
    import xgboost
except ImportError:
    pass
except:
    print("xgboost is installed...but failed to load!")
    pass

try:
    import lightgbm
except ImportError:
    pass
except:
    print("lightgbm is installed...but failed to load!")
    pass

try:
    import catboost
except ImportError:
    pass
except:
    print("catboost is installed...but failed to load!")
    pass


class TreeExplainer(Explainer):
    """Uses the Tree SHAP method to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact (assuming the trees capture the input feature dependencies) method
    to estimate SHAP values for tree models and ensembles of trees. It depends on fast C++
    implementations either inside the externel model package or in the local compiled C extention.
    """

    def __init__(self, model, data = None, model_output = "margin", feature_dependence = "tree_path_dependence", **kwargs):
        """ 
        Parameters
        ----------
        model : Several sklearn, xgboost, lightbgm, and catboost models are supported.
        feature_dependence : Can be "tree_path_dependence" or "independent".
        model_output : Can be "margin", "mse", "logistic", or "logloss".
        """        
        self.model_type = "internal"
        self.less_than_or_equal = False # are threshold comparisons < or <= for this model
        self.base_offset = 0.0
        self.expected_value = None
        self.trees = None
        self.model = None
        self.model_output = model_output
        self.feature_dependence = feature_dependence
        
        # see if the passed model is alerady a list of our Tree objects (in which case no init setup is needed)
        if isinstance(model, list) and isinstance(model[0], Tree):
            self.trees = model
            return

        # parse all the different possible supported model types
        if str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesRegressor'>"):
            self.trees = [Tree(e.tree_) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeRegressor'>"):
            self.trees = [Tree(model.tree_)]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            self.trees = [Tree(model.tree_)]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.forest.ExtraTreesClassifier'>"): # TODO: add unit test for this case
            self.trees = [Tree(e.tree_, normalize=True) for e in model.estimators_]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"): # TODO: add unit test for this case

            # currently we only support the mean estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.MeanEstimator'>"):
                self.base_offset = model.init_.mean
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            scale = len(model.estimators_) * model.learning_rate
            self.trees = [Tree(e.tree_, scaling=scale) for e in model.estimators_[:,0]]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>"):
            
            # currently we only support the logs odds estimator
            if str(type(model.init_)).endswith("ensemble.gradient_boosting.LogOddsEstimator'>"):
                self.base_offset = model.init_.prior
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            scale = len(model.estimators_) * model.learning_rate
            self.trees = [Tree(e.tree_, scaling=scale) for e in model.estimators_[:,0]]
            self.less_than_or_equal = True
        elif str(type(model)).endswith("xgboost.core.Booster'>"):
            if self.feature_dependence == "tree_path_dependence":
                self.model_type = "xgboost"
                self.trees = model
                assert model_output == "margin", "Currently feature_dependence only explains margins"
            elif self.feature_dependence == "independent":
                self.model_type = "trees"
                self.model = model
                self.trees = self.gen_trees(model)
                assert not data is None, "Need to provide a reference set"
                self.data = data
                xgb_ref = xgboost.DMatrix(self.data)
                self.ref_margin_pred = self.model.predict(xgb_ref,output_margin=True)
        elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
            self.model_type = "xgboost"
            self.trees = model.get_booster()
        elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
            self.model_type = "xgboost"
            self.trees = model.get_booster()
        elif str(type(model)).endswith("lightgbm.basic.Booster'>"):
            self.model_type = "lightgbm"
            self.model = model
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMRegressor'>"):
            self.model_type = "lightgbm"
            self.model = model.booster_
        elif str(type(model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            self.model_type = "lightgbm"
            self.model = model.booster_
        elif str(type(model)).endswith("catboost.core.CatBoostRegressor'>"):
            self.model_type = "catboost"
            self.trees = model
        elif str(type(model)).endswith("catboost.core.CatBoostClassifier'>"):
            self.model_type = "catboost"
            self.trees = model
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

    def shap_values(self, X, y=None, tree_limit=-1, approximate=False):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        tree_limit : int
            Limit the number of trees used by the model. By default -1 means no limit.

        approximate : bool
            Run a faster approximate version of Tree SHAP (proposed by Saabas). Only supported for
            XGBoost models right now.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if self.feature_dependence == "tree_path_dependence":
            phi = None
            if self.model_type == "xgboost":
                if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                    X = xgboost.DMatrix(X)
                if tree_limit == -1:
                    tree_limit = 0
                phi = self.trees.predict(X, ntree_limit=tree_limit, pred_contribs=True, approx_contribs=approximate)
            elif self.model_type == "lightgbm":
                assert not approximate, "approximate=True is not supported for LightGBM models!"
                phi = self.model.predict(X, num_iteration=tree_limit, pred_contrib=True)
                if phi.shape[1] != X.shape[1] + 1:
                    phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
            elif self.model_type == "catboost": # thanks to the CatBoost team for implementing this...
                assert not approximate, "approximate=True is not supported for CatBoost models!"
                assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
                if type(X) != catboost.Pool:
                    X = catboost.Pool(X)
                phi = self.trees.get_feature_importance(data=X, fstr_type='ShapValues')

            # note we pull off the last column and keep it as our expected_value
            if phi is not None:
                if len(phi.shape) == 3:
                    self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                    return [phi[:, i, :-1] for i in range(phi.shape[1])]
                else:
                    self.expected_value = phi[0, -1]
                    return phi[:, :-1]

            # convert dataframes
            if str(type(X)).endswith("pandas.core.series.Series'>"):
                X = X.values
            elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
                X = X.values

            assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
            assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

            self.approximate = approximate

            if tree_limit<0 or tree_limit>len(self.trees):
                self.tree_limit = len(self.trees)
            else:
                self.tree_limit = tree_limit

            self.n_outputs = self.trees[0].values.shape[1]
            # single instance
            if len(X.shape) == 1:
                self._current_X = X.reshape(1,X.shape[0])
                self._current_x_missing = np.zeros(X.shape[0], dtype=np.bool)
                phi = self._tree_shap_ind(0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[-1, 0]
                    return phi[:-1, 0]
                else:
                    self.expected_value = [phi[-1, i] for i in range(phi.shape[1])]
                    return [phi[:-1, i] for i in range(self.n_outputs)]

            elif len(X.shape) == 2:
                x_missing = np.zeros(X.shape[1], dtype=np.bool)
                self._current_X = X
                self._current_x_missing = x_missing

                # Only python 3 can serialize a method to send to another process
                if sys.version_info[0] >= 3:
                    pool = multiprocessing.Pool()
                    phi = np.stack(pool.map(self._tree_shap_ind, range(X.shape[0])), 0)
                    pool.close()
                    pool.join()
                else:
                    phi = np.stack(map(self._tree_shap_ind, range(X.shape[0])), 0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[0, -1, 0]
                    return phi[:, :-1, 0]
                else:
                    self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
                    return [phi[:, :-1, i] for i in range(self.n_outputs)]
        # Independence between features
        elif self.feature_dependence == "independent":
            phi_lst = []
            for x_ind in range(0,X.shape[0]):
                phi_lst.append(self.independent_treeshap(X[x_ind,:],y=y).mean(0))
            return(np.array(phi_lst))

    def shap_interaction_values(self, X, tree_limit=-1, **kwargs):

        # shortcut using the C++ version of Tree SHAP in XGBoost and LightGBM
        if self.model_type == "xgboost":
            if not str(type(X)).endswith("xgboost.core.DMatrix'>"):
                X = xgboost.DMatrix(X)
            if tree_limit==-1:
                tree_limit=0
            phi = self.trees.predict(X, ntree_limit=tree_limit, pred_interactions=True)

            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]
        else:

            # lazy build of the trees for lightgbm since we only need them for interaction values right now
            if self.model_type == "lightgbm" and self.trees is None:
                tree_info = self.model.dump_model()["tree_info"]
                self.trees = [Tree(e, scaling=len(tree_info)) for e in tree_info]

            if str(type(X)).endswith("pandas.core.series.Series'>"):
                X = X.values
            elif str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
                X = X.values

            assert str(type(X)).endswith("'numpy.ndarray'>"), "Unknown instance type: " + str(type(X))
            assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

            self.n_outputs = self.trees[0].values.shape[1]

            if tree_limit < 0 or tree_limit > len(self.trees):
                self.tree_limit = len(self.trees)
            else:
                self.tree_limit = tree_limit

            self.n_outputs = self.trees[0].values.shape[1]
            # single instance
            if len(X.shape) == 1:
                self._current_X = X.reshape(1,X.shape[0])
                self._current_x_missing = np.zeros(X.shape[0], dtype=np.bool)
                phi = self._tree_shap_ind_interactions(0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[-1, -1, 0]
                    return phi[:-1, :-1, 0]
                else:
                    self.expected_value = [phi[-1, -1, i] for i in range(phi.shape[2])]
                    return [phi[:-1, :-1, i] for i in range(self.n_outputs)]

            elif len(X.shape) == 2:
                x_missing = np.zeros(X.shape[1], dtype=np.bool)
                self._current_X = X
                self._current_x_missing = x_missing

                # Only python 3 can serialize a method to send to another process
                # TODO: LightGBM models are attached to this object and this seems to cause pool.map to hang
                if sys.version_info[0] >= 3 and self.model_type != "lightgbm":
                    pool = multiprocessing.Pool()
                    phi = np.stack(pool.map(self._tree_shap_ind_interactions, range(X.shape[0])), 0)
                    pool.close()
                    pool.join()
                else:
                    phi = np.stack(map(self._tree_shap_ind_interactions, range(X.shape[0])), 0)

                # note we pull off the last column and keep it as our expected_value
                if self.n_outputs == 1:
                    self.expected_value = phi[0, -1, -1, 0]
                    return phi[:, :-1, :-1, 0]
                else:
                    self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
                    return [phi[:, :-1, :-1, i] for i in range(self.n_outputs)]

    def _tree_shap_ind(self, i):
        phi = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
        phi[-1, :] = self.base_offset * self.tree_limit
        if self.approximate: # only used to mimic Saabas for comparisons right now
            for t in range(self.tree_limit):
                self.approximate_tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi)
        else:
            for t in range(self.tree_limit):
                self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi)
        phi /= self.tree_limit
        return phi

    def _tree_shap_ind_interactions(self, i):
        phi = np.zeros((self._current_X.shape[1] + 1, self._current_X.shape[1] + 1, self.n_outputs))
        phi_diag = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
        for t in range(self.tree_limit):
            self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_diag)
            for j in self.trees[t].unique_features:
                phi_on = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
                phi_off = np.zeros((self._current_X.shape[1] + 1, self.n_outputs))
                self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_on, 1, j)
                self.tree_shap(self.trees[t], self._current_X[i,:], self._current_x_missing, phi_off, -1, j)
                phi[j] += np.true_divide(np.subtract(phi_on,phi_off),2.0)
                phi_diag[j] -= np.sum(np.true_divide(np.subtract(phi_on,phi_off),2.0))
        for j in range(self._current_X.shape[1]+1):
            phi[j][j] = phi_diag[j]
        phi /= self.tree_limit
        return phi

    def tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):
        # start the recursive algorithm
        assert have_cext, "C extension was not built during install!"
        _cext.tree_shap(
            tree.max_depth, tree.children_left, tree.children_right, tree.children_default, tree.features,
            tree.thresholds, tree.values, tree.node_sample_weight,
            x, x_missing, phi, condition, condition_feature, self.less_than_or_equal
        )

    def approximate_tree_shap(self, tree, x, x_missing, phi, condition=0, condition_feature=0):
        """ This is a simple approximation equivelent to the Saabas method.

        It is actually slow because it is in python, but that's fine right now since it is just used
        for benchmark comparisons with Saabas. It would need to be added to tree_shap.h as C++ if we
        wanted it to be high speed.

        x_missing, condition, and condition_feature are currently not used
        """

        def recurse(node):
            i = tree.features[node]
            if i < 0: return
            if x[i] < tree.thresholds[node]:
                child = tree.children_left[node]
            else:
                child = tree.children_right[node]
            phi[i] += tree.values[child] - tree.values[node]
            recurse(child)

        recurse(0)

    def gen_trees(self, model):
        """ Create trees given an XGB model

        Parameters
        ----------
        model : An xgboost model.

        Returns
        -------
        Returns a list of trees for future explanation.
        """
        assert str(type(model)).endswith("xgboost.core.Booster'>")
        model_type = "xgboost"
        xgb_trees = model.get_dump(with_stats = True)
        scale = len(xgb_trees)
        trees = []
        for xgb_tree in xgb_trees:
            nodes = [t.lstrip() for t in xgb_tree[:-1].split("\n")]
            nodes_dict = {}
            for n in nodes: nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
            trees.append(self.create_tree(nodes_dict,scale))
        return(trees)

    def create_tree(self, nodes_dict, scale):
        """ Creates a tree given a dictionary representation of a tree.

        Parameters
        -------
        nodes_dict : A dictionary that contains a single tree.  For example:
        {0: '[f1<0] yes=1,no=2,missing=1,gain=4500,cover=2000',
         1: '[f0<0] yes=3,no=4,missing=3,gain=1000,cover=1000',
         2: '[f0<0] yes=5,no=6,missing=5,gain=1000,cover=1000',
         3: 'leaf=0.5,cover=500',
         4: 'leaf=2.5,cover=500',
         5: 'leaf=-2.5,cover=500',
         6: 'leaf=-0.5,cover=500'}

        Returns
        -------
        A Tree object.
        """
        m = max(nodes_dict.keys())+1
        children_left = -1*np.ones(m,dtype="int32")
        children_right = -1*np.ones(m,dtype="int32")
        children_default = -1*np.ones(m,dtype="int32")
        features = -2*np.ones(m,dtype="int32")
        thresholds = -1*np.ones(m,dtype="float64")
        values = 1*np.ones(m,dtype="float64")
        node_sample_weights = np.zeros(m,dtype="float64")
        values_lst = list(nodes_dict.values())
        keys_lst = list(nodes_dict.keys())
        for i in range(0,len(keys_lst)):
            value = values_lst[i]
            key = keys_lst[i]
            if ("leaf" in value):
                # Extract values
                val = float(value.split("leaf=")[1].split(",")[0])
                node_sample_weight = float(value.split("cover=")[1])
                # Append to lists
                values[key] = val
                node_sample_weights[key] = node_sample_weight
            else:
                c_left = int(value.split("yes=")[1].split(",")[0])
                c_right = int(value.split("no=")[1].split(",")[0])
                c_default = int(value.split("missing=")[1].split(",")[0])
                feat_thres = value.split(" ")[0]
                if ("<" in feat_thres):
                    feature = int(feat_thres.split("<")[0][2:])
                    threshold = float(feat_thres.split("<")[1][:-1])
                if ("=" in feat_thres):
                    feature = int(feat_thres.split("=")[0][2:])
                    threshold = float(feat_thres.split("=")[1][:-1])
                node_sample_weight = float(value.split("cover=")[1].split(",")[0])
                children_left[key] = c_left
                children_right[key] = c_right
                children_default[key] = c_default
                features[key] = feature
                thresholds[key] = threshold
                node_sample_weights[key] = node_sample_weight
        tree_dict = {"children_left":children_left, "children_right":children_right,
                     "children_default":children_default, "feature":features,
                     "threshold":thresholds, "value":values[:,np.newaxis],
                     "node_sample_weight":node_sample_weights}
        return(Tree(tree_dict))

    def independent_treeshap(self,x,y=None):
        """ Recursively calculate Shapley values for a single reference.

        Parameters
        -------
        tree : Current tree object.
        x : Current sample.

        Returns
        -------
        The one reference Shapley value for all features.
        """
        assert have_cext, "C extension was not built during install!"
        x_missing = np.isnan(x)
        feats = range(0, self.data.shape[1])
        phi_final = []
        for tree in self.trees:
            phi = []
            for j in range(self.data.shape[0]):
                r = self.data[j,:]
                r_missing = np.isnan(r)
                out_contribs = np.zeros(x.shape)
                _cext.tree_shap_indep(
                    tree.max_depth, tree.children_left, tree.children_right, 
                    tree.children_default, tree.features, tree.thresholds, 
                    tree.values, x, x_missing, r, r_missing, out_contribs
                )
                phi.append(out_contribs)
            phi_final.append(phi)
        phi = np.array(phi_final).sum(0) # Sum across trees
        # Compute rescale
        if self.model_output == "mse" or self.model_output == "logloss":
            assert not y is None, "Need to provide true label y"
        self.expected_value = self.ref_margin_pred.mean()
        if not self.model_output == "margin":
            margin_pred = self.model.predict(xgboost.DMatrix(x[np.newaxis,:]),output_margin=True)
            if self.model_output == "mse":
                ref_transform_pred = mse(y,self.ref_margin_pred)
                transform_pred = mse(y,margin_pred)
            elif self.model_output == "logistic":
                ref_transform_pred = sigmoid(self.ref_margin_pred)
                transform_pred = sigmoid(margin_pred)
            elif self.model_output == "logloss":
                ref_transform_pred = log_loss(y,sigmoid(self.ref_margin_pred))
                transform_pred = log_loss(y,sigmoid(margin_pred))
            self.expected_value = ref_transform_pred.mean()
            num = transform_pred - ref_transform_pred
            den = margin_pred - self.ref_margin_pred
            rescale = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
            phi = phi * rescale[:,np.newaxis]
        return(phi)
   
# Supported non-linear transforms
def sigmoid(x):
    return(1/(1+np.exp(-x)))

def log_loss(yt,yp):
    return(-(yt*np.log(yp) + (1 - yt)*np.log(1 - yp)))

def mse(yt,yp):
    return(np.square(yt-yp))


class Tree:
    def __init__(self, tree, normalize=False, scaling=1.0):
        if str(type(tree)).endswith("'sklearn.tree._tree.Tree'>"):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            if normalize:
                self.values = (tree.value[:,0,:].T / tree.value[:,0,:].sum(1)).T
            else:
                self.values = tree.value[:,0,:]
            self.values = self.values * scaling

            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)
            self.unique_features = np.unique(self.features)
            self.unique_features = np.delete(self.unique_features, np.where(self.unique_features < 0))

            # we compute the expectations to make sure they follow the SHAP logic
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

        elif type(tree) == dict and 'children_left' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"]
            self.node_sample_weight = tree["node_sample_weight"]
            self.unique_features = np.unique(self.features)
            self.unique_features = np.delete(self.unique_features, np.where(self.unique_features==-1))
            # we compute the expectations to make sure they follow the SHAP logic
            assert have_cext, "C extension was not built during install!"
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )

        elif type(tree) == dict and 'tree_structure' in tree:
            start = tree['tree_structure']
            num_parents = tree['num_leaves']-1
            self.children_left = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_right = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_default = np.empty((2*num_parents+1), dtype=np.int32)
            self.features = np.empty((2*num_parents+1), dtype=np.int32)
            self.thresholds = np.empty((2*num_parents+1), dtype=np.float64)
            self.values = [-2]*(2*num_parents+1)
            self.node_sample_weight = np.empty((2*num_parents+1), dtype=np.float64)
            visited, queue = [], [start]
            while queue:
                vertex = queue.pop(0)
                if 'split_index' in vertex.keys():
                    if vertex['split_index'] not in visited:
                        if 'split_index' in vertex['left_child'].keys():
                            self.children_left[vertex['split_index']] = vertex['left_child']['split_index']
                        else:
                            self.children_left[vertex['split_index']] = vertex['left_child']['leaf_index']+num_parents
                        if 'split_index' in vertex['right_child'].keys():
                            self.children_right[vertex['split_index']] = vertex['right_child']['split_index']
                        else:
                            self.children_right[vertex['split_index']] = vertex['right_child']['leaf_index']+num_parents
                        if vertex['default_left']:
                            self.children_default[vertex['split_index']] = self.children_left[vertex['split_index']]
                        else:
                            self.children_default[vertex['split_index']] = self.children_right[vertex['split_index']]
                        self.features[vertex['split_index']] = vertex['split_feature']
                        self.thresholds[vertex['split_index']] = vertex['threshold']
                        self.values[vertex['split_index']] = [vertex['internal_value']]
                        self.node_sample_weight[vertex['split_index']] = vertex['internal_count']
                        visited.append(vertex['split_index'])
                        queue.append(vertex['left_child'])
                        queue.append(vertex['right_child'])
                else:
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.thresholds[vertex['leaf_index']+num_parents] = -1
                    self.values[vertex['leaf_index']+num_parents] = [vertex['leaf_value']]
                    self.node_sample_weight[vertex['leaf_index']+num_parents] = vertex['leaf_count']
            self.values = np.asarray(self.values)
            self.values = np.multiply(self.values, scaling)
            self.unique_features = np.unique(self.features)
            self.unique_features = np.delete(self.unique_features, np.where(self.unique_features < 0))

            assert have_cext, "C extension was not built during install!" + str(have_cext)
            self.max_depth = _cext.compute_expectations(
                self.children_left, self.children_right, self.node_sample_weight,
                self.values
            )
