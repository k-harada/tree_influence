import numpy as np

from . import util
from ._tree import _Tree


class Tree(object):
    """
    Wrapper for the standardized tree object.

    Note:
        - The Tree object is a binary tree structure.
        - The tree structure is used for predictions and extracting
          structure information.

    Reference
        - https://github.com/scikit-learn/scikit-learn/blob/
            15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/tree/_tree.pyx
    """

    def __init__(self, children_left, children_right, feature, threshold, leaf_vals):
        children_left = np.array(children_left, dtype=np.intp)
        children_right = np.array(children_right, dtype=np.intp)
        feature = np.array(feature, dtype=np.intp)
        threshold = np.array(threshold, dtype=np.float32)
        leaf_vals = np.array(leaf_vals, dtype=np.float32)
        self.tree_ = _Tree(children_left, children_right, feature, threshold, leaf_vals)

    def predict(self, X):
        """
        Return 1d array of leaf values, shape=(X.shape[0]).
        """
        assert X.ndim == 2
        return self.tree_.predict(X)

    def apply(self, X):
        """
        Return 1d array of leaf indices, shape=(X.shape[0],).
        """
        assert X.ndim == 2
        return self.tree_.apply(X)

    def get_leaf_values(self):
        """
        Return 1d array of leaf values, shape=(no. leaves,).
        """
        return self.tree_.get_leaf_values()

    def update_node_count(self, X):
        """
        Update node counts based on the paths taken by x in X.
        """
        assert X.ndim == 2
        self.tree_.update_node_count(X)

    def leaf_path(self, X, output=False, weighted=False):
        """
        Return 2d vector of leaf one-hot encodings, shape=(X.shape[0], no. leaves).
        """
        return self.tree_.leaf_path(X, output=output, weighted=weighted)

    def feature_path(self, X, output=False, weighted=False):
        """
        Return 2d vector of feature one-hot encodings, shape=(X.shape[0], no. nodes).
        """
        return self.tree_.feature_path(X, output=output, weighted=weighted)

    @property
    def node_count_(self):
        return self.tree_.node_count_

    @property
    def leaf_count_(self):
        return self.tree_.leaf_count_


class TreeEnsemble(object):
    """
    Standardized tree-ensemble model.
    """
    def __init__(self, trees, objective, tree_type, bias,
                 learning_rate, l2_leaf_reg, factor):
        """
        Input
            trees: A 1d (or 2d for multiclass) array of Tree objects.
            objective: str, task ("regression", "binary", or "multiclass").
            bias: A single or 1d list (for multiclass) of floats.
                If classification, numbers are in log space.
            tree_type: str, "gbdt" or "rf".
            learning_rate: float, learning rate (GBDT models only).
            l2_leaf_reg: float, leaf regularizer (GBDT models only).
            factor: float, scaler for the redundant class (GBDT multiclass models only)
        """
        assert trees.dtype == np.dtype(object)
        self.trees = trees
        self.objective = objective
        self.tree_type = tree_type
        self.bias = bias
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.factor = factor

        # validate
        if self.objective in ['regression', 'binary']:
            assert self.trees.shape[1] == 1
            assert isinstance(self.bias, float)

        elif self.objective == 'multiclass':
            assert self.trees.shape[1] > 2
            assert len(self.bias) == self.trees.shape[1]

    def __str__(self):
        """
        Return string representation of model.
        """
        params = self.get_params()
        return str(params)

    def get_params(self):
        """
        Return dict. of object parameters.
        """
        params = {}
        params['objective'] = self.objective
        params['tree_type'] = self.tree_type
        params['bias'] = self.bias
        params['learning_rate'] = self.learning_rate
        params['l2_leaf_reg'] = self.l2_leaf_reg
        params['factor'] = self.factor

    def predict(self, X):
        """
        Sums leaf over all trees for each x in X, then apply activation.

        Returns 2d array of leaf values; shape=(X.shape[0], no. class)
        """
        X = util.check_input_data(X)

        # shape=(X.shape[0], no. class)
        pred = np.tile(self.bias, (X.shape[0], 1)).astype(np.float32)

        for boost_idx in range(self.n_boost_):  # per boosting round
            for class_idx in range(self.n_class_):  # per class
                pred[:, class_idx] += self.trees[boost_idx, class_idx].predict(X)

        # transform predictions based on the tree type and objective
        if self.tree_type == 'rf':
            pred /= self.n_boost_

        else:  # gbdt

            if self.objective == 'binary':
                pred = util.sigmoid(pred)

            elif self.objective == 'multiclass':
                pred = util.softmax(pred)

        return pred

    def apply(self, X):
        """
        Returns 3d array of leaf indices; shape=(X.shape[0], no. boost, no. class).
        """
        leaves = np.zeros((X.shape[0], self.n_boost_, self.n_class_), dtype=np.int32)
        for boost_idx in range(self.n_boost_):
            for class_idx in range(self.n_class_):
                leaves[:, boost_idx, class_idx] = self.trees[boost_idx][class_idx].apply(X)
        return leaves

    def get_leaf_values(self):
        """
        Returns 1d array of leaf values of shape=(no. leaves across all trees,).

        Note
            - Multiclass trees are flattened s.t. trees from all classes in one boosting
                iteration come before those in the subsequent boosting iteration.
        """
        return np.concatenate([tree.get_leaf_values() for tree in self.trees.flatten()]).astype(np.float32)

    def get_leaf_counts(self):
        """
        Returns 2d array of leaf counts; shape=(no. boost, no. class).

        Note
            - Multiclass trees are flattened s.t. trees from all classes in one boosting
                iteration come before those in the subsequent boosting iteration.
        """
        leaf_counts = np.zeros((self.n_boost_, self.n_class_), dtype=np.int32)
        for boost_idx in range(self.n_boost_):
            for class_idx in range(self.n_class_):
                leaf_counts[boost_idx, class_idx] = self.trees[boost_idx, class_idx].leaf_count_
        return leaf_counts

    def update_node_count(self, X):
        """
        Increment each node's count for each x in X that passes through each node
            for all trees in the ensemble.
        """
        for tree in self.trees.flatten():
            tree.update_node_count(X)

    @property
    def n_boost_(self):
        """
        Returns no. boosting iterations (no. weak learners per class) in the ensemble.
        """
        return self.trees.shape[0]

    @property
    def n_class_(self):
        """
        Returns no. "sets of trees". E.g. 1 for regression and binary
            since they both only need 1 set of trees for their objective.
            Multiclass needs k >= 3 sets of trees.
        """
        return self.trees.shape[1]
