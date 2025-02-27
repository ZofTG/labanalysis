"""
ordinary least squares regression module

a set of functions dedicated to the use of least squares model regression

Classes
---------
PolynomialRegression
    regression model in the form:
            Y = b0 + b1 * fn(X)**1 + ... + bn * fn(X)**n + e


MultiSegmentRegression
    regression model in the form:
            Ys = b0s + b1s * fn(Xs)**1 + ... + bns * fn(Xs)**n + e

        with s denoting a limited continuous interval of X


PowerRegression
    regression model having form:
            Y = b0 + b1 * [fn(X) + b2] ** b3 + e
"""

#! IMPORTS


import copy
import warnings
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    "PolynomialRegression",
    "MultiSegmentRegression",
    "PowerRegression",
]


#! CLASSES


class PolynomialRegression(LinearRegression):
    """
    Ordinary Least Squares regression model in the form:

            Y = b0 + b1 * fn(X)**1 + ... + bn * fn(X)**n + e

    where "b0...bn" are the model coefficients and "fn" is a transform function
    applied elemenwise to each sample of X.

    Parameters
    ----------
    degree: int = 1
        the polynomial order

    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations
        (i.e. data is expected to be centered).

    transform: Callable, default = lambda x: x
        a callable function defining the type of transform to be applied
        elementwise to each input value of X before the extension to the
        required polynomial degree.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        n_targets > 1 and secondly X is sparse or if positive is set to True.
        None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors. See Glossary for more details.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
        This option is only supported for dense arrays.

    Attributes
    ----------
    degree: int
        the polynomial degree

    betas: pandas DataFrame
        a dataframe reporting the regression coefficients for each feature

    additional attributes are described from the mother scikit-learn
    LinearRegression class object:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    _domain = (-np.inf, np.inf)
    _codomain = (-np.inf, np.inf)
    _degree: int
    _names_out: list[str]
    _names_in: list[str]
    _transform: Callable
    _has_intercept: bool
    _betas: pd.DataFrame

    def __init__(
        self,
        degree: int = 1,
        fit_intercept: bool = True,
        transform: Callable = lambda x: x,
        copy_X: bool = True,
        n_jobs: int = 1,
        positive: bool = False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self._degree = degree
        self._transform = transform
        self._has_intercept = fit_intercept

    @property
    def transform(self):
        """return the transform function"""
        return self._transform

    @property
    def degree(self):
        """return the polynomial degree"""
        return self._degree

    @property
    def domain(self):
        """return the domain of this model"""
        return self._domain

    @property
    def codomain(self):
        """return the codomain of this model"""
        return self._codomain

    @property
    def betas(self):
        """return the beta coefficients of the model"""
        return self._betas

    def get_feature_names_in(self):
        """return the input feature names seen at fit time"""
        return self._names_in

    def get_feature_names_out(self):
        """return the output feature names seen at fit time"""
        return self._names_out

    def _simplify(
        self,
        vec: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        label: str = "",
    ):
        """
        internal method to format the entries in the constructor and call
        methods.

        Parameters
        ----------
        vec: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None
            the data to be formatter

        label: str
            in case an array is provided, the label is used to define the
            columns of the output DataFrame.

        Returns
        -------
        dfr: pd.DataFrame
            the data formatted as DataFrame.
        """

        def simplify_array(v: NDArray, l: str):
            if v.ndim == 1:
                d = np.atleast_2d(v).T
            elif v.ndim == 2:
                d = v
            else:
                raise ValueError(v)
            cols = [f"{l}{i}" for i in range(d.shape[1])]
            return pd.DataFrame(d.astype(float), columns=cols)

        if isinstance(vec, pd.DataFrame):
            return vec.astype(float)
        if isinstance(vec, pd.Series):
            return pd.DataFrame(vec).astype(float)
        if isinstance(vec, list):
            return simplify_array(np.array(vec), label)
        if isinstance(vec, np.ndarray):
            return simplify_array(vec, label)
        if np.isreal(vec):
            return simplify_array(np.array([vec]), label)
        raise NotImplementedError(vec)

    def _adjust_degree(
        self,
        xarr: pd.DataFrame,
    ):
        """
        prepare the input to the fit and predict methods

        Parameters
        ----------
        xarr : np.ndarray | pd.DataFrame | pd.Series | list | int | float
           the training data

        Returns
        -------
        xvec: pd.DataFrame | pd.Series
            the transformed features
        """
        feats = PolynomialFeatures(
            degree=self.degree,
            interaction_only=False,
            include_bias=self._has_intercept,
        )
        return pd.DataFrame(
            data=feats.fit_transform(xarr),
            columns=feats.get_feature_names_out(),
            index=xarr.index,
        )

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        yarr: array-like or DataFrame of shape (n_samples,)|(n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.

        Returns
        -------
        self
            the fitted estimator
        """

        # get the input and output variables
        Y = self._simplify(yarr, "Y")
        self._names_out = Y.columns.tolist()
        X = self._adjust_degree(self._simplify(xarr, "X").map(self.transform))
        self._names_in = X.columns.tolist()

        # fit the model
        fitted = super().fit(X, Y)

        # update the betas
        coefs = [np.atleast_2d(fitted.intercept_), fitted.coef_[:, 1:]]
        coefs = np.concatenate(coefs, axis=1).T
        fitted._betas = pd.DataFrame(
            data=coefs,
            index=[f"beta{i}" for i in np.arange(coefs.shape[0])],
            columns=self.get_feature_names_out(),
        )

        return fitted

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        yarr: DataFrame
            the predicted values.
        """
        X = self._adjust_degree(self._simplify(xarr).map(self.transform))
        return pd.DataFrame(
            data=super().predict(X.values),
            columns=self.get_feature_names_out(),
            index=X.index,
        )

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)


class MultiSegmentRegression(PolynomialRegression):
    """
    ordinary polynomial least squares regression splitted on multiple segments

    Parameters
    ----------
    degree: int = 1
        the polynomial degree

    transform: Callable, default = lambda x: x
        a callable function defining the type of transform to be applied
        elementwise to each input value of X before the extension to the
        required polynomial degree.

    n_segments: int = 1
        number of segments to be calculated

    min_samples : int | None
        The minimum number of different samples defining the x axis of each line.
        If not provided, the number of degree + 1 is used.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        n_targets > 1 and secondly X is sparse or if positive is set to True.
        None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors. See Glossary for more details.

    Attributes
    ----------
    degree: int
        the polynomial degree

    betas: pandas DataFrame
        a dataframe reporting the regression coefficients for each feature

    additional attributes are described from the mother scikit-learn
    LinearRegression class object:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    _n_segments: int
    _min_samples: int

    def __init__(
        self,
        degree: int = 1,
        n_lines: int = 1,
        min_samples: int | None = None,
        transform: Callable = lambda x: x,
        copy_X: bool = True,
        n_jobs: int = 1,
        positive: bool = False,
    ):
        super().__init__(
            degree=degree,
            transform=transform,
            fit_intercept=False,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self._n_segments = n_lines
        if min_samples is None:
            self._min_samples = self.degree + 1
        else:
            self._min_samples = min_samples

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)

    @property
    def n_segments(self):
        """the number of lines defining the model"""
        return self._n_segments

    @property
    def min_samples(self):
        """
        return the minimum number of unique values on the x-axis to be used
        for generating each single line of the regression model
        """
        return self._min_samples

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        yarr: array-like or DataFrame of shape (n_samples,)|(n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.

        Returns
        -------
        self
            the fitted estimator
        """
        # format the input data
        X = self._simplify(xarr, "X")
        Y = self._simplify(yarr, "Y")

        # check the inputs
        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array")
        if X.shape[0] != Y.shape[0]:
            msg = "xarr and yarr must have equal sample size"
            raise ValueError(msg)

        # get the unique values
        unique_x = np.unique(X.values.flatten()).astype(float)
        n_unique = len(unique_x)

        # apply the transform
        X = X.map(self.transform)

        # get all the possible combinations of segments
        combs = []
        for i in np.arange(1, self.n_segments):
            start = self.min_samples * i
            stop = n_unique - self.min_samples * (self.n_segments - i)
            combs += [np.arange(start, stop)]
        combs = list(product(*combs))

        # remove those combinations having segments shorter than "min_samples"
        combs = [i for i in combs if np.all(np.diff(i) >= self.min_samples)]

        # generate the crossovers index matrix
        combs = (
            np.zeros((len(combs), 1)),
            np.atleast_2d(combs),
            np.ones((len(combs), 1)) * (n_unique - 1),
        )
        combs = np.hstack(combs).astype(int)

        # iterate each combination to get their regression coefficients,
        # the segments range, and sum of squares
        betas_list = []
        sses_list = []
        for comb in combs:

            # evaluate each segment of the current combination
            combination_sse = 0
            combination_betas_list = []
            y0 = 0
            x0 = np.atleast_1d(X.values[0]).astype(float)
            for i, (i0, i1) in enumerate(zip(comb[:-1], comb[1:])):

                # get x and y samples corresponding to the current segment
                unq_vals = unique_x[np.arange(i0, i1 + 1)]
                index = [np.where(X.values[:, 0] == j)[0] for j in unq_vals]
                index = np.concatenate(index)
                ymat = Y.iloc[index] - y0
                xmat = self._adjust_degree(X.iloc[index] - x0)
                if i == 0:
                    xmat.insert(0, "Intercept", np.ones((xmat.shape[0],)))

                # get the beta coefficients
                bs = (xmat @ np.linalg.inv(xmat.T @ xmat)).T @ ymat

                # update the combination error
                ypred = (xmat @ bs.values + y0).values
                ytrue = ymat.values + y0
                sse = float(((ytrue - ypred) ** 2).sum())
                combination_sse += sse

                # update the coefficients list with the beta values of the
                # current segment
                if i != 0:
                    bs.index = pd.Index([f"beta{j + 1}" for j in range(bs.shape[0])])
                    bs.loc["beta0", bs.columns] = y0
                else:
                    bs.index = pd.Index([f"beta{j}" for j in range(bs.shape[0])])
                bs.loc["alpha0", bs.columns] = x0
                bs.sort_index(inplace=True)
                r0 = -np.inf if i0 == 0 else x0
                r1 = +np.inf if i1 == (n_unique - 1) else unq_vals[-1]
                bs.columns = pd.MultiIndex.from_product(
                    iterables=[ymat.columns.tolist(), [r0], [r1]],
                    names=["FEATURE", "X0", "X1"],
                )
                combination_betas_list += [bs]

                # update offsets
                y0 = ypred[-1]
                x0 = r1

            # merge the combinations betas
            combination_betas = pd.concat(combination_betas_list, axis=1)
            combination_betas.sort_index(axis=1, inplace=True)
            betas_list += [combination_betas]
            sses_list += [combination_sse]

        # get the best combination (i.e. the one with the lowest sse)
        index = np.argmin(sses_list)

        # get the beta coefficients corresponding to the minimum
        # sum of squares error
        self._betas = betas_list[index]

        return self

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        yarr: DataFrame
            the predicted values.
        """
        # check input
        X = self._simplify(xarr, "X").map(self.transform)
        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array")

        # we now prepare the output (empty) dataframe
        feats = self.betas.columns.to_frame().FEATURE.values.astype(str)
        feats = np.unique(feats)
        Y = pd.DataFrame(index=X.index, columns=feats)

        # now we calculate the predicted values for each segment and feature
        for feat, i0, i1 in self.betas.columns:
            idx = np.where((X.values >= i0) & (X.values <= i1))[0]
            coefs = self.betas[[(feat, i0, i1)]].values.astype(float)
            x0 = coefs[0]
            betas = coefs[1:]
            xmat = self._adjust_degree(X.iloc[idx] - x0)
            xmat.insert(0, "Intercept", np.ones((xmat.shape[0],)))
            Y.loc[X.index[idx], [feat]] = (xmat @ betas).values

        return Y


class PowerRegression(PolynomialRegression):
    """
    Regression model having form:

                Y = b0 * fn(X) ** b3 + e

    Parameters
    ----------
    transform: Callable, default = lambda x: x
        a callable function defining the type of transform to be applied
        elementwise to each input value of X before the extension to the
        required polynomial degree.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, default=None
        The number of jobs to use for the computation. This will only provide
        speedup in case of sufficiently large problems, that is if firstly
        n_targets > 1 and secondly X is sparse or if positive is set to True.
        None means 1 unless in a joblib.parallel_backend context. -1 means
        using all processors. See Glossary for more details.

    positive : bool, default=False
        When set to True, forces the coefficients to be positive.
        This option is only supported for dense arrays.

    Attributes
    ----------
    betas: pandas DataFrame
        a dataframe reporting the regression coefficients for each feature

    additional attributes are described from the mother scikit-learn
    LinearRegression class object:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    _domain = (-np.inf, np.inf)
    _codomain = (0, np.inf)

    def __init__(
        self,
        transform: Callable = lambda x: x,
        copy_X: bool = True,
        n_jobs: int = 1,
        positive: bool = False,
    ):
        super().__init__(
            degree=1,
            fit_intercept=True,
            transform=transform,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        yarr: array-like or DataFrame of shape (n_samples,)|(n_samples, n_targets)
            Target values. Will be cast to X’s dtype if necessary.

        Returns
        -------
        self
            the fitted estimator
        """
        # check the inputs
        X = self._simplify(xarr, "X")
        K = X.map(self.transform)
        Y = self._simplify(yarr, "Y")
        if K.shape[1] != 1:
            raise ValueError("xarr must be a 1D array or equivalent set")

        # transform the data
        Yt = (Y).map(np.log)
        Xt = (K).map(np.log)
        fitted = super().fit(Xt, Yt)
        b1 = float(np.e**fitted.intercept_)
        b3 = float(np.squeeze(fitted.coef_)[-1])
        fitted._betas = pd.DataFrame(
            data=[b1, b3],
            index=[f"beta{i}" for i in range(2)],
            columns=Y.columns,
        )

        return fitted

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X: array-like or DataFrame of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        z: DataFrame
            the predicted values.
        """
        # check the inputs
        X = self._simplify(xarr, "X")
        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array or equivalent set")

        # get the predictions
        b0, b1 = self.betas.values.astype(float).flatten()
        return b0 * (X.map(self.transform) ** b1)

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)
