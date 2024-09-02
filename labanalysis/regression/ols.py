"""
ordinary least squares regression module

a set of functions dedicated to the use of least squares model regression

Classes
---------
LinearRegression
    regression model in the form:
            Y = b0 + b1 * X1 + ... + bn * Xn + e


PolynomialRegression
    regression model in the form:
            Y = b0 + b1 * X**1 + ... + bn * X**n + e


PowerRegression
    regression model having form:
            Y = b0 * X1 ** b1 * ... + Xn ** bn + e


EllipseRegression
    Regression tool fitting an ellipse in a 2D space


CircleRegression
    Regression tool fitting a circle in a 2D space
"""

#! IMPORTS

from collections import namedtuple
from itertools import product
from typing import Callable, NamedTuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import copy


__all__ = [
    "PolynomialRegression",
    "MultiSegmentRegression",
    "PowerRegression",
    "EllipseRegression",
    "CircleRegression",
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
    _names_out:list[str]
    _names_in:list[str]
    _transform:Callable
    _has_intercept: bool

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
        rows = len(self.get_feature_names_in()) + 1
        names = self.get_feature_names_out()
        cols = len(names)
        betas = np.zeros((rows, cols))
        betas[0, 0] = self.intercept_
        betas[1:, :] = np.atleast_2d(self.coef_).T
        return pd.DataFrame(
            data=betas,
            index=[f"beta{i}" for i in np.arange(rows)],
            columns=names,
        )

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
            return pd.DataFrame(vec).T.astype(float)
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
        Y = self._simplify(yarr, "Y")
        self._names_out = Y.columns.tolist()
        X = self._adjust_degree(self._simplify(xarr, "X").map(self.transform))
        self._names_in = X.columns.tolist()
        return super().fit(X.values, Y)

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

    min_samples : int = 2
        The minimum number of different samples defining the x axis of each line.

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
    _betas:pd.DataFrame

    def __init__(
        self,
        degree: int = 1,
        n_lines: int = 1,
        min_samples: int = 2,
        transform: Callable = lambda x: x,
        copy_X: bool = True,
        n_jobs: int = 1,
        positive: bool = False,
    ):
        super().__init__(
            degree=degree,
            transform=transform,
            fit_intercept=True,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )
        self._n_segments = n_lines
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

    @property
    def betas(self):
        """coefficients and ranges"""
        return self._betas

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
            y0 = np.atleast_1d(Y.values[0]).astype(float)
            x0 = np.atleast_1d(X.values[0]).astype(float)
            for i, (i0, i1) in enumerate(zip(comb[:-1], comb[1:])):

                # get x and y samples corresponding to the current segment
                unq_vals = unique_x[np.arange(i0, i1 + 1)]
                index = [np.where(X.values[:, 0] == i)[0] for i in unq_vals]
                index = np.concatenate(index)
                xmat = self._adjust_degree(X.iloc[index] - x0)
                ymat = Y.iloc[index] - y0

                # get the beta coefficients
                bs = (xmat @ np.linalg.inv(xmat.T @ xmat)).T @ ymat

                # update the combination error
                ypred = (xmat @ bs.values + y0).values
                ytrue = ymat.values + y0
                sse = float(((ytrue - ypred) ** 2).sum())
                combination_sse += sse

                # update the coefficients list with the beta values of the
                # current segment
                bs.loc[-1, bs.columns] = y0
                bs.loc[-2, bs.columns] = x0
                bs.sort_index(inplace=True)
                cols = ["alpha0"] + [f"beta{i}" for i in range(bs.shape[0] - 1)]
                bs.index = pd.Index(cols)
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

                Y = b0 + b1 * (fn(X) - b2) ** b3 + e

    Parameters
    ----------
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
    betas: pandas DataFrame
        a dataframe reporting the regression coefficients for each feature

    additional attributes are described from the mother scikit-learn
    LinearRegression class object:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    """

    _domain = (-np.inf, np.inf)
    _codomain = (-np.inf, np.inf)
    _betas:pd.DataFrame

    def __init__(
        self,
        fit_intercept: bool = True,
        transform: Callable = lambda x: x,
        copy_X: bool = True,
        n_jobs: int = 1,
        positive: bool = False,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            transform=transform,
            copy_X=copy_X,
            n_jobs=n_jobs,
            positive=positive,
        )

    @property
    def betas(self):
        """return the beta coefficients of the model"""
        return self._betas

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
        X = self._simplify(xarr, "X").map(self.transform)
        Y = self._simplify(yarr, "Y")
        if X.shape[1] != 1:
            raise ValueError("xarr must be a 1D array or equivalent set")

        # get b0 and b2
        b0 = float(np.atleast_1d(0 if not self._has_intercept else (Y.min() - 1)))
        b2 = -float(np.atleast_1d(X.min())) + 1

        # transform the data
        Yt = (Y - b0).map(np.log)
        Xt = (X + b2).map(np.log)
        fitted = super().fit(Xt, Yt)
        b1 = float(np.e**fitted.intercept_)
        b3 = float(np.squeeze(fitted.coef_)[-1])
        fitted._betas = pd.DataFrame(
            data = [b0, b1, b2, b3],
            index = [f"beta{i}" for i in range(4)],
            columns = Y.columns,
        )
        fitted._codomain = (b0, np.inf)
        fitted._domain = (b2, np.inf)
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

        # apply the data transform
        X = X.map(self.transform)

        # check the domain
        if float(X.min().values[0]) < self.domain[0]:
            raise ValueError(f"X values must lie in the [{self.domain}] range.")

        # get the predictions
        b0, b1, b2, b3 = self.betas.values
        return b0 + b1 * (X + b2) ** b3

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)


class EllipseRegression(PolynomialRegression):
    """
    fit an Ellipse to the provided data according to the model:
        a * X**2 + b * XY + c * Y**2 + d * X + e * Y + f = 0

    References
    ----------
    Halir R, Flusser J. Numerically stable direct least squares fitting of
        ellipses. InProc. 6th International Conference in Central Europe on
        Computer Graphics and Visualization. WSCG 1998 (Vol. 98, pp. 125-132).
        Citeseer. https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=DF7A4B034A45C75AFCFF861DA1D7B5CD?doi=10.1.1.1.7559&rep=rep1&type=pdf
    """

    _axis_major: NamedTuple
    _axis_minor: NamedTuple
    _names: list[str] = []
    coef_: np.ndarray
    intercept_: float

    def __init__(
        self,
    ):
        super().__init__(
            fit_intercept=True,
            copy_X=True,
            n_jobs=1,
            positive=False,
        )

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        sample_weight: (
            np.ndarray | pd.DataFrame | pd.Series | list | int | float | None
        ) = None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X: 1D array-like
            x-axis data.

        y: 1D array-like.
            y-axis data.

        sample_weight: array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self
            the fitted estimator
        """
        yvec = self._simplify(yarr, "Y")
        xvec = self._simplify(xarr, "X")
        if yvec.shape[1] != 1 or xvec.shape[1] != 1:
            raise ValueError("'x' and 'y' must be 1D arrays.")

        # quadratic part of the design matrix
        xval = xvec.values.flatten()
        yval = yvec.values.flatten()
        d_1 = np.vstack([xval**2, xval * yval, yval**2]).T

        # linear part of the design matrix
        d_2 = np.vstack([xval, yval, np.ones(len(xval))]).T

        # quadratic part of the scatter matrix
        s_1 = d_1.T @ d_1

        # combined part of the scatter matrix
        s_2 = d_1.T @ d_2

        # linear part of the scatter matrix
        s_3 = d_2.T @ d_2

        # reduced scatter matrix
        cnd = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        trc = -np.linalg.inv(s_3) @ s_2.T
        mat = np.linalg.inv(cnd) @ (s_1 + s_2 @ trc)

        # solve the eigen system
        eigvec = np.linalg.eig(mat)[1]

        # evaluate the coefficients
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
        eiv_pos = eigvec[:, np.nonzero(con > 0)[0]]
        coefs = np.concatenate((eiv_pos, trc @ eiv_pos)).ravel()
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]

        # get the axes angles
        # ref: http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node28.html
        a, c, b = self.betas.values.flatten()[:3]

        # get the axes angles
        if c == 0:
            raise ValueError("coefficient c = 0.")
        m0 = (b - a) / c
        m0 = (m0**2 + 1) ** 0.5 + m0
        if m0 == 0:
            raise ValueError("m0 = 0.")
        m1 = -1 / m0

        # We know that the two axes pass from the centre of the Ellipse
        # and we also know the angle of the major and minor axes.
        # Therefore the intercept of the fitting lines describing the two
        # axes can be found.
        x0, y0 = self.center
        i0 = y0 - x0 * m0
        i1 = y0 - x0 * m1

        # get the crossings between the two axes and the Ellipse
        p0_0, p0_1 = self._get_crossings(slope=m0, intercept=i0)
        p0 = np.vstack(np.atleast_2d(p0_0, p0_1))  # type: ignore
        p1_0, p1_1 = self._get_crossings(slope=m1, intercept=i1)
        p1 = np.vstack(np.atleast_2d(p1_0, p1_1))  # type: ignore

        # get the angle of the two axes
        a0 = float(np.arctan(m0)) * 180 / np.pi
        a1 = float(np.arctan(m1)) * 180 / np.pi

        # get the length of the two axes
        l0 = ((p0_1 - p0_0) ** 2).sum() ** 0.5  # type: ignore
        l1 = ((p1_1 - p1_0) ** 2).sum() ** 0.5  # type: ignore

        # generate the axes
        axis = namedtuple("Axis", ["vertex", "length", "angle", "coef"])
        ax0 = axis(vertex=p0, length=l0, angle=a0, coef=np.array([i0, m0]))
        ax1 = axis(vertex=p1, length=l1, angle=a1, coef=np.array([i1, m1]))
        if l0 > l1:
            self._axis_major = ax0
            self._axis_minor = ax1
        else:
            self._axis_major = ax1
            self._axis_minor = ax0
        return self

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None
            x axis data

        y: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
            y axis adata

        Returns
        -------
        z: ArrayLike
            the predicted values.
        """
        if xarr is None and yarr is None:
            return None
        if xarr is not None:
            v = self._simplify(xarr, "X")
            o = np.atleast_2d([self._get_roots(x=i) for i in v.values])
            self._names = ["Y0", "Y1"]
        elif yarr is not None:
            v = self._simplify(yarr, "Y")
            o = np.atleast_2d([self._get_roots(y=i) for i in v.values])
            self._names = ["X0", "X1"]
        else:
            raise ValueError("x or y must be not None")
        assert v.shape[1] == 1, "Only 1D arrays can be provided."
        return o.astype(float)

    def _get_crossings(
        self,
        slope: int | float,
        intercept: int | float,
    ):
        """
        get the crossings between the provided line and the Ellipse

        Parameters
        ----------
        slope: float
            the slope of the axis line

        intercept: float
            the intercept of the axis line

        Returns
        -------
        p0, p1: tuple
            the coordinates of the crossing points. It returns None if
            the line does not touch the Ellipse.
        """
        a, b, c, d, e, f = self.betas.values.flatten()
        a_ = a + b * slope + c * slope**2
        b_ = b * intercept + 2 * slope * intercept * c + d + e * slope
        c_ = c * intercept**2 + e * intercept + f
        d_ = b_**2 - 4 * a_ * c_
        if d_ < 0:
            return (None, None), (None, None)
        e_ = 2 * a_
        if a_ == 0:
            return (None, None), (None, None)
        f_ = -b_ / e_
        g_ = (d_**0.5) / e_
        x0 = f_ - g_
        x1 = f_ + g_
        return (
            np.array([x0, x0 * slope + intercept]),
            np.array([x1, x1 * slope + intercept]),
        )

    def _solve(
        self,
        a: float,
        b: float,
        c: float,
    ):
        """
        obtain the solutions of a second order polynomial having form:
                a * x**2 + b * x + c = 0

        Parameters
        ----------
        a, b, c: float
            the coefficients of the equation.

        Returns
        -------
        x0, x1: float | None
            the roots of the polynomial. None is returned if the solution
            is impossible.
        """
        d = b**2 - 4 * a * c
        if d < 0:
            raise ValueError("b**2 - 4 * a * c < 0")
        k = (2 * a) ** (-1)
        i = -b * k
        j = k * d**0.5
        return float(i + j), float(i - j)

    def _get_roots(
        self,
        x: float | int | None = None,
        y: float | int | None = None,
    ):
        """
        obtain the roots of a second order polynomial having form:

                a * x**2 + b * x + c = 0

        Parameters
        ----------
        x: float | int | None
            the given x value.

        y: float | int | None
            the given y value.

        Returns
        -------
        x0, x1: float | None
            the roots of the polynomial. None is returned if the solution
            is impossible.
        """
        # get the coefficients
        a_, b_, c_, d_, e_, f_ = self.betas.values.flatten()
        if y is not None and x is None:
            y_ = float(y)
            a, b, c = a_, b_ * y_ + d_, f_ + c_ * y_**2 + e_ * y_
        elif x is not None and y is None:
            x_ = float(x)
            a, b, c = c_, b_ * x_ + e_, f_ + a_ * x_**2 + d_ * x_
        else:
            raise ValueError("Only one 'x' or 'y' must be provided.")

        # get the roots
        return self._solve(a, b, c)

    def is_inside(
        self,
        x: int | float,
        y: int | float,
    ):
        """
        check whether the point (x, y) is inside the Ellipse.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the Ellipse.
        """
        out = self.predict(xarr=x)
        if out is None:
            return False
        if isinstance(out, (pd.DataFrame, pd.Series)):
            out = out.values.astype(float).flatten()
        y0, y1 = out
        return bool((y0 is not None) & (y > min(y0, y1)) & (y <= max(y0, y1)))

    @property
    def axis_major(self):
        """return the axis major of the ellipse"""
        return self._axis_major

    @property
    def axis_minor(self):
        """return the axis major of the ellipse"""
        return self._axis_minor

    @property
    def center(self):
        """
        get the center of the Ellipse as described here:
        https://mathworld.wolfram.com/Ellipse.html

        Returns
        -------
        x0, y0: float
            the coordinates of the centre of the Ellipse.
        """
        a, b, c, d, e = self.betas.values.flatten()[:-1]
        den = b**2 - 4 * a * c
        x = float((2 * c * d - b * e) / den)
        y = float((2 * a * e - b * d) / den)
        return x, y

    @property
    def area(self):
        """
        the area of the Ellipse.

        Returns
        -------
        a: float
            the area of the Ellipse.
        """
        ax1 = self.axis_major.length  # type: ignore
        ax2 = self.axis_minor.length  # type: ignore
        return float(np.pi * ax1 * ax2)

    @property
    def perimeter(self):
        """
        the (approximated) perimeter of the Ellipse as calculated
        by the "infinite series approach".
                P = pi * (a + b) * sum_{n=0...N} (h ** n / (4 ** (n + 1)))
        where:
            h = (a - b) ** 2 / (a ** 2 + b ** 2)
            a = axis major
            b = axis minor
            N = any natural number.

        Note:
        -----
        N is set such as the output measure no longer changes up to the
        12th decimal number.

        Returns
        -------
        p: float
            the approximated perimeter of the ellipse.
        """
        a = self.axis_major.length / 2  # type: ignore
        b = self.axis_minor.length / 2  # type: ignore
        if a == 0 and b == 0:
            raise ValueError("a and b coefficients = 0.")
        h = (a - b) ** 2 / (a**2 + b**2)
        c = np.pi * (a + b)
        p_old = -c
        p = c
        n = 0
        q = 1
        while n == 0 or abs(p_old - p) > 1e-12:
            p_old = p
            n += 1
            q += h**n / 4**n
            p = c * q

        return float(p)

    @property
    def eccentricity(self):
        """
        return the eccentricity parameter of the ellipse.
        """
        b = self.axis_minor.length / 2  # type: ignore
        a = self.axis_major.length / 2  # type: ignore
        if a == 0:
            raise ValueError("coefficient a = 0")
        return float(1 - b**2 / a**2) ** 0.5

    @property
    def foci(self):
        """
        return the coordinates of the foci of the ellipses.

        Returns
        -------
        f0, f1: tuple
            the coordinates of the crossing points. It returns None if
            the line does not touch the ellipse.
        """
        a = self.axis_major.length / 2  # type: ignore
        p = self.axis_major.angle  # type: ignore
        x, y = a * self.eccentricity * np.array([np.cos(p), np.sin(p)])
        x0, y0 = self.center
        return (float(x0 - x), float(y0 - y)), (float(x0 + x), float(y0 + y))

    @property
    def domain(self):
        """
        return the domain of the ellipse.

        Returns
        -------
        x1, x2: float
            the x-axis boundaries of the ellipse.
        """

        # get the roots to for the 2nd order equation to be solved
        a_, b_, c_, d_, e_, f_ = self.betas.values.flatten()
        a = b_**2 - 4 * a_ * c_
        b = 2 * b_ * e_ - 4 * c_ * d_
        c = e_**2 - 4 * c_ * f_

        # solve the equation
        x0, x1 = np.sort(self._solve(a, b, c))
        return float(x0), float(x1)

    @property
    def codomain(self):
        """
        return the codomain of the ellipse.

        Returns
        -------
        y1, y2: float
            the y-axis boundaries of the ellipse.
        """
        # get the roots to for the 2nd order equation to be solved
        a_, b_, c_, d_, e_, f_ = self.betas.values.flatten()
        a = b_**2 - 4 * a_ * c_
        b = 2 * b_ * d_ - 4 * a_ * e_
        c = d_**2 - 4 * a_ * f_

        # solve the equation
        y0, y1 = np.sort(self._solve(a, b, c))
        return float(y0), float(y1)

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)


class CircleRegression(PolynomialRegression):
    """
    generate a circle from the provided data in a least squares sense.

    References
    ----------
    https://lucidar.me/en/mathematics/least-squares-fitting-of-circle/
    """

    coef_: np.ndarray
    intercept_: float
    _names: list[str] = []

    def __init__(
        self,
    ):
        super().__init__(
            fit_intercept=True,
            copy_X=True,
            n_jobs=1,
            positive=False,
        )

    def fit(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
        sample_weight: (
            np.ndarray | pd.DataFrame | pd.Series | list | int | float | None
        ) = None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: 1D array-like
            x-axis data.

        yarr: 1D array-like.
            y-axis data.

        sample_weight: array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self
            the fitted estimator
        """
        yvec = self._simplify(yarr, "Y")
        xvec = self._simplify(xarr, "X")
        if yvec.shape[1] != 1 or xvec.shape[1] != 1:
            raise ValueError("'x' and 'y' must be 1D arrays.")
        x = xvec.values.flatten()
        yarr = yvec.values.flatten()
        i = np.tile(1, len(yarr))
        a = np.vstack(np.atleast_2d(x, yarr, i)).T
        b = np.atleast_2d(x**2 + yarr**2).T
        pinv = np.linalg.pinv
        coefs = pinv(a.T @ a) @ a.T @ b
        self.coef_ = coefs[:-1]
        self.intercept_ = coefs[-1]
        return self

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None
            x axis data

        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
            y axis adata

        Returns
        -------
        zarr: ArrayLike
            the predicted values.
        """
        if xarr is None and yarr is None:
            return None
        if xarr is not None:
            v = self._simplify(xarr, "X")
            o = np.atleast_2d([self._get_roots(x=i) for i in v.values])
            self._names = ["Y0", "Y1"]
        elif yarr is not None:
            v = self._simplify(yarr, "Y")
            o = np.atleast_2d([self._get_roots(y=i) for i in v.values])
            self._names = ["X0", "X1"]
        else:
            raise ValueError("x or y must be not None")
        assert v.shape[1] == 1, "Only 1D arrays can be provided."
        return o.astype(float)

    def _get_roots(
        self,
        x: float | int | None = None,
        y: float | int | None = None,
    ):
        """
        obtain the roots of a second order polynomial having form:
                a * x**2 + b * x + c = 0

        Parameters
        ----------
        x: Union[float, int, None] = None,
            the given x value.

        y: Union[float, int, None] = None,
            the given y value.

        Returns
        -------
        x0, x1: float | None
            the roots of the polynomial.
        """
        # get the coefficients
        x0, y0 = self.center
        r = self.radius
        if y is not None and x is None:
            a, b, c = 1, -2 * x0, x0**2 - r**2 + (float(y) - y0) ** 2
        elif x is not None and y is None:
            a, b, c = 1, -2 * y0, y0**2 - r**2 + (float(x) - x0) ** 2
        else:
            raise ValueError("Only one 'x' or 'y' must be provided.")

        # get the roots
        d = b**2 - 4 * a * c
        if d < 0:
            raise ValueError("b**2 - 4 * a * c < 0")
        if a == 0:
            raise ValueError("coefficient a = 0")
        return float((-b - d**0.5) / (2 * a)), float((-b + d**0.5) / (2 * a))

    def is_inside(
        self,
        x: int | float,
        y: int | float,
    ):
        """
        check whether the point (x, y) is inside the ellipse.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the ellipse.
        """
        out = self.predict(xarr=x)
        if out is None:
            return False
        if isinstance(out, pd.DataFrame | pd.Series):
            out = out.values.astype(float)
        y0, y1 = out
        return bool((y0 is not None) & (y > min(y0, y1)) & (y <= max(y0, y1)))

    @property
    def radius(self):
        """
        get the radius of the circle.

        Returns
        -------
        r: float
            the radius of the circle.
        """
        a, b, c = self.betas.values.flatten()
        return float((4 * c + a**2 + b**2) ** 0.5) * 0.5

    @property
    def center(self):
        """
        get the center of the circle.

        Returns
        -------
        x0, y0: float
            the coordinates of the centre of the cicle.
        """
        a, b = self.betas.values.flatten()[:-1]
        return float(a * 0.5), float(b * 0.5)

    @property
    def area(self):
        """
        the area of the circle.

        Returns
        -------
        a: float
            the area of the circle.
        """
        return float(np.pi * self.radius**2)

    @property
    def perimeter(self):
        """
        the perimeter of the circle.

        Returns
        -------
        p: float
            the perimeter of the circle.
        """
        return float(2 * self.radius * np.pi)

    @property
    def domain(self):
        """
        return the domain of the circle.

        Returns
        -------
        x1, x2: float
            the x-axis boundaries of the circle.
        """
        x = self.center[0]
        r = self.radius
        return float(x - r), float(x + r)

    @property
    def codomain(self):
        """
        return the codomain of the circle.

        Returns
        -------
        y1, y2: float
            the y-axis boundaries of the circle.
        """
        y = self.center[1]
        r = self.radius
        return float(y - r), float(y + r)

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)
