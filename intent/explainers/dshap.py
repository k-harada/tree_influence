import time
import joblib

import numpy as np
from sklearn.base import clone

from .base import Explainer
from .parsers import util


class DShap(Explainer):
    """
    Explainer that approx. data Shapley values using
    the TMC-Shapley algorithm.

    Global-Influence Semantics
        - Average influence of x_i on itself.
        - Inf.(x_i, x_i) = Avg. L(y_i, f_{w/o x_i}(x_i)) - L(y_i, f(x_i))
            over all possible permutations of the training data.
        - Pos. value means a decrease in loss (a.k.a. proponent, helpful).
        - Neg. value means an increase in loss (a.k.a. opponent, harmful).

    Local-Influence Semantics
        - Inf.(x_i, x_t) = Avg. L(y_i, f_{w/o x_i}(x_t)) - L(y_i, f(x_t))
            over all possible permutations of the training data.
        - Pos. value means a decrease in test loss (a.k.a. proponent, helpful).
        - Neg. value means an increase in test loss (a.k.a. opponent, harmful).

    Reference
        - https://github.com/amiratag/DataShapley

    Paper
        - http://proceedings.mlr.press/v97/ghorbani19c.html

    Note
        - Supports both GBDTs and RFs.
        - No validation set, we are computing loss on training or ONE test example;
            thus, there is no average loss score and use of a `tolerance` parameter
            for early truncation.
            * However, we can use a hard truncation limit via `trunc_frac`.
    """
    def __init__(self, trunc_frac=0.25, n_jobs=1, check_every=10, random_state=1, logger=None):
        """
        Input
            trunc_frac: float, fraction of instances to compute marginals for per iter.
            n_jobs: int, no. iterations / processes to run in parallel.
            check_every: int, no. iterations to run between checking convergence.
            random_state: int, random seed to enhance reproducibility.
            logger: object, If not None, output to logger.
        """
        self.trunc_frac = trunc_frac
        self.n_jobs = n_jobs
        self.check_every = check_every
        self.random_state = random_state
        self.logger = logger

    def fit(self, model, X, y):
        """
        - Convert model to internal standardized tree structures.
        - Perform any initialization necessary for the chosen method.

        Input
            model: tree ensemble.
            X: 2d array of train data.
            y: 1d array of train targets.
        """
        super().fit(model, X, y)
        X, y = util.check_data(X, y, objective=self.model_.objective)

        self.original_model_ = model
        self.objective_ = self.model_.objective
        self.n_class_ = self.model_.n_class_

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        self.loss_fn_ = util.get_loss_fn(self.objective_, self.n_class_, self.model_.factor)
        self.random_loss_ = self._get_random_loss()

        return self

    def get_global_influence(self, check_every=10, stability_tol=0.1):
        """
        - Compute influence of each training example on itself.
        - Provides a global perspective of which training intances
          are most important.

        Return
            - 1d array of shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """
        return self._run_tmc_shapley()

    def get_local_influence(self, X, y):
        """
        - Compute influence of each training instance on the test loss.

        Input
            - X: 2d array of test examples.
            - y: 1d array of test targets.

        Return
            - 2d array of shape=(no. train, X.shape[0]).
                * Arrays are returned in the same order as the training data.
        """
        influence = np.zeros((self.X_train_.shape[0], X.shape[0]))
        for test_idx in range(X.shape[0]):
            influence[:, test_idx] = self._run_tmc_shapley(X_test=X[[test_idx]], y_test=y[[test_idx]])
        return influence

    # private
    def _run_tmc_shapley(self, X_test=None, y_test=None, stability_tol=0.1, n_run=100):
        """
        - Run the TMC-Shapley algorithm until marginal contributions converge.

        Return
            - 1d array of average marginals, shape=(no. train,).
                * Arrays are returned in the same order as the traing data.
        """

        # extract parameters
        original_model = self.original_model_
        X_train = self.X_train_
        y_train = self.y_train_
        loss_fn = self.loss_fn_
        random_loss = self.random_loss_
        truncation_frac = self.trunc_frac
        objective = self.objective_
        n_class = self.n_class_
        random_state = self.random_state
        check_every = self.check_every

        # select no. processes to run in parallel
        if self.n_jobs == -1:
            n_jobs = joblib.cpu_count()

        else:
            assert self.n_jobs >= 1
            n_jobs = min(self.n_jobs, joblib.cpu_count())

        start = time.time()
        if self.logger:
            self.logger.info('\n[INFO] computing approx. data Shapley values...')
            self.logger.info(f'[INFO] no. cpus: {n_jobs:,}...')

        # run TMC-Shapley alg. until convergence
        with joblib.Parallel(n_jobs=n_jobs) as parallel:
            marginals = np.zeros((0, self.X_train_.shape[0]), dtype=np.float32)  # result container
            iteration = 0

            while True:

                results = parallel(joblib.delayed(_run_iteration)
                                                 (original_model, X_train, y_train, loss_fn, random_loss,
                                                  truncation_frac, objective, n_class, random_state,
                                                  iteration, i, X_test, y_test) for i in range(check_every))
                iteration += check_every

                # synchronization barrier
                marginals = np.vstack([marginals, results])  # shape=(iteration, no. train)

                # add up all marginals using axis=0, then divide by their iteration
                all_vals = (np.cumsum(marginals, axis=0) / np.arange(1, len(marginals) + 1).reshape(-1, 1))[-n_run:]

                # diff. between last `n_run` runs and last run, divide by last run, average over all points
                errors = np.mean(np.abs(all_vals[-n_run:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), axis=-1)

                if self.logger:
                    cum_time = time.time() - start
                    self.logger.info(f'[INFO] Iter. {iteration:,}, stability: {np.max(errors):.3f}, '
                                     f'cum. time: {cum_time:.3f}s')
                
                # check convergence
                if np.max(errors) < stability_tol:
                    break

        influence = marginals.mean(axis=0)  # shape=(no. train,)

        return influence

    def _get_random_loss(self):
        """
        Input
            X: 2d array of data.
            y: 1d array of targets.

        Return 1d array of losses resulting from a random guess; shape=(X.shape[0],)
        """
        if self.model_.objective == 'regression':
            loss = 0

        elif self.model_.objective == 'binary':
            loss = -np.log(0.5)

        else:
            assert self.model_.objective == 'multiclass'
            loss = -np.log(1.0 / self.model_.n_class_)

        return loss


def _run_iteration(original_model, X_train, y_train, loss_fn, random_loss,
                   truncation_frac, objective, n_class, finished_iterations,
                   cur_iter, random_state, X_test=None, y_test=None):
    """
    - Run one iteration of the TMC-Shapley algorithm.

    Return
        - 1d array of marginals, shape=(no. train,).

    Note
        - Parallelizable method.
    """
    rng = np.random.default_rng(random_state + finished_iterations + cur_iter)

    # get order of training examples to add
    train_idxs = rng.permutation(y_train.shape[0])  # shape=(no. train,)
    train_idxs = train_idxs[:int(len(train_idxs) * truncation_frac)]  # truncate examples

    # result container
    marginals = np.zeros(X_train.shape[0])  # shape=(no. train,)

    # empty containers
    X_batch = np.zeros((0,) + (X_train.shape[1],), dtype=np.float32)  # shape=(0, no. feature)
    y_batch = np.zeros(0, dtype=np.int32)  # shape=(0,)

    old_loss = random_loss  # tracker
    old_model = None

    # add training examples one at a time to measure the effect of each one
    for train_idx in train_idxs:

        # add example to batch of examples
        X_batch = np.vstack([X_batch, X_train[train_idx].reshape(1, -1)])
        y_batch = np.concatenate([y_batch, y_train[train_idx].reshape(1)])

        # skip batches that do not have enough examples
        if objective == 'regression' and X_batch.shape[0] < 2:
            continue

        elif objective == 'binary' and len(np.unique(y_batch)) < 2:
            continue

        elif objective == 'multiclass' and len(np.unique(y_batch)) < n_class:
            continue

        # train and score
        model = clone(original_model).fit(X_batch, y_batch)

        # global influence
        if X_test is None:
            X_temp = X_train[[train_idx]]
            y_temp = y_train[[train_idx]]

            if old_model is None:
                old_loss = random_loss

            else:
                old_loss = _get_loss(loss_fn, old_model, objective, X=X_temp, y=y_temp)

            loss = _get_loss(loss_fn, model, objective, X=X_temp, y=y_temp)
            marginals[train_idx] = old_loss - loss  # loss(x_i) w/o x_i - loss(x_i) w/ x_i

            old_model = model

        # local influence
        else:
            loss = _get_loss(loss_fn, model, objective, X=X_test, y=y_test)
            marginals[train_idx] = old_loss - loss  # loss(x_t) w/o x_i - loss(x_t) w/ x_i
            old_loss = loss

    return marginals


def _get_loss(loss_fn, model, objective, X, y):
    """
    Return
        - 1d array of individual losses of shape=(X.shape[0],).

    Note
        - Parallelizable method.
    """
    assert X.shape[0] == y.shape[0] == 1

    if objective == 'regression':
        y_pred = model.predict(X)  # shape=(X.shape[0])

    elif objective == 'binary':
        y_pred = model.predict_proba(X)[:, 1]  # 1d arry of pos. probabilities

    else:
        assert objective == 'multiclass'
        y_pred = model.predict_proba(X)  # shape=(X.shape[0], no. class)

    losses = loss_fn(y, y_pred, raw=False)  # shape(X.shape[0],)
    loss = losses[0]

    return loss
