"""
2D and 3D space objects extrapolated by least squares approaches

Classes
---------
Line2D
    line object in a 2D space having general form:
            A * x + B * y + C = 0

Line3D
    line object in a 3D space having general form:
            A * x + B * y + C * z + D = 0
"""

#! IMPORTS


import copy
from math import pi as PI
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import sympy

__all__ = ["GeometricObject", "Line2D", "Line3D", "Circle"]


#! CLASSES


@runtime_checkable
class GeometricObject(Protocol):
    """
    General geometric object class. This object is thought to be extended
    according to the geometric object of interest.

    The standard use case is to extend this class and implement at least the
    method '_expand_dimensions'". Optionally also the methods fit and predict
    might be re-implemented to make better specify the input data.
    """

    _betas: dict[str, float]
    _general_equation: sympy.Eq
    _dimensions: list[str]
    _has_intercept: bool

    @property
    def has_intercept(self):
        """return True if the model has intercept"""
        return self._has_intercept

    @property
    def betas(self):
        """return the betas coefficients"""
        return self._betas

    @property
    def coefs(self):
        """return the labels of the coefficients in the general equation"""
        symbols = [str(i) for i in self.general_equation.free_symbols]
        return sorted([i for i in symbols if i not in self.dimensions])

    @property
    def general_equation(self):
        """return the equation of the model"""
        return self._general_equation

    @property
    def fitted_equation(self):
        """return the equation with the appropriate coefficients"""
        if not self.is_fitted():
            msg = "the fit method has to be used before 'fitted_equation'"
            raise ValueError(msg)
        return self.general_equation.subs(self.betas)

    @property
    def dimensions(self):
        """get the dimensions of the shape"""
        return self._dimensions

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

        def simplify_array(v: np.ndarray, l: str):
            if v.ndim == 1:
                d = np.atleast_2d(v).T
            elif v.ndim == 2:
                d = v
            else:
                raise ValueError(v)
            cols = [f"{l}{i}" for i in range(d.shape[1])]
            return pd.DataFrame(d.astype(float), columns=cols)

        if isinstance(vec, pd.DataFrame):
            out = vec.astype(float)
        elif isinstance(vec, pd.Series):
            out = pd.DataFrame(vec).T.astype(float)
        elif isinstance(vec, list):
            out = simplify_array(np.array(vec), label)
        elif isinstance(vec, np.ndarray):
            out = simplify_array(vec, label)
        elif np.isreal(vec):
            out = simplify_array(np.array([vec]), label)
        else:
            raise NotImplementedError(vec)
        return out.values.astype(float).flatten()

    def is_fitted(self):
        """check if the current objects has the required beta coefficients"""
        return len(self.betas) > 0

    def _check_symbols(self):
        """check the symbols provided"""
        target_symbols = list(self.general_equation.free_symbols)
        target_symbols = np.array(target_symbols).astype(str)
        available_symbols = self.dimensions + list(self.betas.keys())
        missing = [i for i in target_symbols if i not in available_symbols]
        if len(missing) > 0:
            msg = f"General Equation: {self.general_equation}\n"
            msg += f"Missing symbols: {missing}"
            raise ValueError(msg)

    def __str__(self):
        if self.is_fitted():
            return str(self.fitted_equation.args[0].__str__())
        return str(self.general_equation.args[0].__str__())

    def __repr__(self):
        return self.__str__()

    def __init__(
        self,
        general_equation: sympy.Equality,
        dimensions: list[str],
        has_intercept: bool = True,
    ):
        if not isinstance(general_equation, sympy.Equality):
            raise ValueError(f"general_equation must be {sympy.Equality}")
        self._general_equation = general_equation

        if not isinstance(dimensions, list):
            raise ValueError(f"dimensions must be {list} of {str}")
        if not all([isinstance(i, str) for i in dimensions]):
            raise ValueError(f"dimensions must be {list} of {str}")
        self._dimensions = dimensions

        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be a bool")
        self._has_intercept = has_intercept

        self._betas = {}

        # check ghe proper general equation
        if len(self.coefs) != len(list(self.general_equation.args[0].args)):
            msg = "dimensions does not match with the provided number of "
            msg += "coefficients."
            raise ValueError(msg)

    def _expand_dimensions(self, **dimensions: np.ndarray) -> np.ndarray: ...

    def copy(self):
        """create a copy of the current object."""
        return copy.deepcopy(self)

    def predict(
        self,
        **known_dimensions: (
            np.ndarray | list[float] | int | float | pd.Series | pd.DataFrame
        ),
    ):
        """
        predict the unknown dimensions giving all but one other dimensions.

        Parameters
        ----------
        known_dimensions: key-valued ArrayLike
            the input dimensions.

        Returns
        -------
        predicted_dimension: ArrayLike
            the value corresponding to each known dimension pair.
        """
        # check the status of the object
        if not self.is_fitted():
            raise ValueError("this object must be fitted before using 'predict'")
        self._check_symbols()

        # check the input dimensions
        known_dims = {}
        for key, val in known_dimensions.items():
            if key not in self.dimensions:
                raise ValueError(f"valid dimensions are {self.dimensions}")
            known_dims[key] = self._simplify(val)
        if np.sum(np.diff([len(i) for i in list(known_dims.values())])) != 0:
            msg = "All known dimensions must have the same number of samples."
            raise ValueError(msg)
        keys = list(known_dims.keys())
        samples = np.array(list(known_dims.values())).T

        # get the unknown dimension
        unknown_dim = [i for i in self.dimensions if i not in list(known_dims.keys())]
        if len(unknown_dim) != 1:
            raise ValueError("Only one unknown dimension is possible.")
        unknown_dim = unknown_dim[0]

        # fit the values
        out = []
        for vals in samples:
            eq = self.fitted_equation.subs({i: v for i, v in zip(keys, vals)})
            out += [sympy.solve(eq, unknown_dim)]
        out = np.squeeze(out).astype(float)

        # prepare the output
        lbls = [f"{unknown_dim}{i}" for i in np.arange(out.shape[1])]
        return pd.DataFrame(out, columns=lbls)

    def fit(
        self,
        **dimensions_samples: (
            np.ndarray | list[float] | int | float | pd.Series | pd.DataFrame
        ),
    ):
        """
        fit the current object to the provided dimensions

        Parameters
        ----------
        dimensions_samples: key-valued ArrayLike
            the input dimensions.
        """

        # check the inputs
        keys = list(dimensions_samples.keys())
        if not all([i in keys for i in self.dimensions]):
            msg = f"The following dimensions must be provided: {self.dimensions}"
            raise ValueError(msg)
        vals = [self._simplify(j, i) for i, j in dimensions_samples.items()]
        if not all([len(i) for i in vals]):
            msg = "all provided dimensions must have the same number of samples."
            raise ValueError(msg)

        # get the design matrix
        mat = self._expand_dimensions(**{i: v for i, v in zip(keys, vals)})

        # consider the intercept
        if self.has_intercept:
            mat = np.hstack([mat, np.ones((mat.shape[0], 1))])

        # np.hstack performs a loop over all samples and creates
        # a row in J for each x,y,z sample:
        offset = -mat[:, -1]
        mat = mat[:, :-1]
        betas = (np.linalg.pinv(mat.T @ mat) @ mat.T @ offset).astype(float)
        betas = np.round(np.concatenate([betas, [1]]), 15)
        self._betas = {i: float(v) for i, v in zip(self.coefs, betas)}

        return self


class Line2D(GeometricObject):
    """
    line object in a 2D space having general form:

            A * x + B * y + C = 0

    Parameters
    ----------
    has_intercept: bool (default = True)
        if False the C coefficient is excluded.
    """

    def __init__(self, has_intercept: bool = True):
        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be True or False")

        if has_intercept:
            x, y, A, B, C = sympy.symbols("x,y,A,B,C")
            equation = sympy.Equality(A * x + B * y + C, 0)
        else:
            x, y, A, B = sympy.symbols("x,y,A,B")
            equation = sympy.Equality(A * x + B * y, 0)

        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=has_intercept,
        )

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        return {
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
        }

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        out = [self._simplify(dimensions[i], i) for i in ["x", "y"]]
        return np.vstack(np.atleast_2d(*out)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Line2D
            a fitted Line2D object
        """
        return super().fit(x=x, y=y)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")


class Line3D(GeometricObject):
    """
    line object in a 3D space having general form:

            A * x + B * y + C * z + D = 0

    Parameters
    ----------
    has_intercept: bool (default = True)
        if False the D coefficient is excluded.
    """

    def __init__(self, has_intercept: bool = True):
        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be True or False")

        if has_intercept:
            x, y, z, A, B, C, D = sympy.symbols("x,y,z,A,B,C,D")
            equation = sympy.Equality(A * x + B * y + C * z + D, 0)
        else:
            x, y, z, A, B, C = sympy.symbols("x,y,z,A,B,C")
            equation = sympy.Equality(A * x + B * y + C * z, 0)

        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y", "z"],
            has_intercept=has_intercept,
        )

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        return {
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
            "z": (-np.inf, np.inf),
        }

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        out = [self._simplify(dimensions[i], i) for i in ["x", "y", "z"]]
        return np.vstack(np.atleast_2d(*out)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
        z: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        z: np.ndarray | pd.Series | list[float | int] | float | int
            the z-axis coordinates

        Returns
        -------
        fitted: Line3D
            a fitted Line3D object
        """
        return super().fit(x=x, y=y, z=z)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        z: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        z: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the z-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only 2 of the three axes have to be provided.
        """
        if x is None and y is not None and z is not None:
            return super().predict(y=y, z=z)
        if y is None and x is not None and z is not None:
            return super().predict(x=x, z=z)
        if z is None and x is not None and y is not None:
            return super().predict(x=x, y=y)
        raise ValueError("Just one of 'x', 'y' or 'z' must be not None.")


class Circle(GeometricObject):
    """
    circle object in a 2D space having general form:

            x^2 + y^2 + A * x + B * y + C = 0

    Equivalently another expression of the circle is:

            (x - x0)^2 + (y - y0)^2 = r^2

    Where:
        x0 and y0 are the coordinates of the center of the circle
        r is the radius of the circle.

    Hence:
        x0 = -A/2
        y0 = -B/2
        r = (A^2 / 4 + B^2 / 4 - C) ^ 0.5
    """

    @property
    def center(self):
        """return the coordinates of the center of the circle."""
        return -self.betas["A"] / 2, -self.betas["B"] / 2

    @property
    def radius(self):
        """return the radius of the circle"""
        A = self.betas["A"]
        B = self.betas["B"]
        C = self.betas["C"]
        return float(((A**2) / 4 + (B**2) / 4 - C) ** 0.5)

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        if not self.is_fitted():
            raise ValueError("'domains' must be called after fit.")
        x0, y0 = self.center
        r = self.radius
        return {
            "x": (x0 - r, x0 + r),
            "y": (y0 - r, y0 + r),
        }

    @property
    def perimeter(self):
        """return the perimeter of the circle"""
        return 2 * PI * self.radius

    @property
    def area(self):
        """return the area of the circle"""
        return PI * self.radius**2

    def __init__(self):
        x, y, A, B, C = sympy.symbols("x,y,A,B,C")
        equation = sympy.Equality(x ^ 2 + y ^ 2 + A * x + B * y + C, 0)
        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=True,
        )

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        x, y = [self._simplify(dimensions[i], i) for i in ["x", "y"]]
        return np.vstack(np.atleast_2d(x**2, y**2, x, y)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Circle
            a fitted Circle object
        """
        return super().fit(x=x, y=y)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")


'''
class Parabola(GeometricObject):
    """
    'parabola' object in a 2D space having general form:

        A * x^2 + B * y^2 + 2 * (A * B)^0.5 * x * y + C * x + D * y + E = 0

    Parameters
    ----------
    has_intercept: bool (default = True)
        if False the D coefficient is excluded.
    """

    def __init__(self, has_intercept: bool = True):
        if not isinstance(has_intercept, bool):
            raise ValueError("'has_intercept' must be True or False")

        if has_intercept:
            x, y, A, B, C = sympy.symbols("x,y,A,B,C")
            equation = sympy.Equality(A * x + B * y + C, 0)
        else:
            x, y, A, B = sympy.symbols("x,y,A,B")
            equation = sympy.Equality(A * x + B * y, 0)

        super().__init__(
            general_equation=equation,  # type: ignore
            dimensions=["x", "y"],
            has_intercept=has_intercept,
        )

    @property
    def domains(self):
        """return the value limits accepted for each dimension"""
        return {
            "x": (-np.inf, np.inf),
            "y": (-np.inf, np.inf),
        }

    def _expand_dimensions(self, **dimensions: np.ndarray):
        """prepare the design matrix"""
        out = [self._simplify(dimensions[i], i) for i in ["x", "y"]]
        return np.vstack(np.atleast_2d(*out)).T

    def fit(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int,
        y: np.ndarray | pd.Series | list[float | int] | float | int,
    ):
        """
        fit the model with the required data

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int
            the y-axis coordinates

        Returns
        -------
        fitted: Line2D
            a fitted Line2D object
        """
        return super().fit(x=x, y=y)

    def predict(
        self,
        x: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
        y: np.ndarray | pd.Series | list[float | int] | float | int | None = None,
    ):
        """
        get predictions of the not provided coordinate

        Parameters
        ----------
        x: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the x-axis coordinates

        y: np.ndarray | pd.Series | list[float | int] | float | int | None (default=None)
            the y-axis coordinates

        Returns
        -------
        out: np.ndarray
            the coordinates of the axis not provided

        Note
        ----
        only the 'x' or 'y' argument has to be provided.
        """
        if x is None and y is not None:
            return super().predict(y=y)
        if y is None and x is not None:
            return super().predict(x=x)
        raise ValueError("Just 'x' or 'y' must be not None.")


class EllipseRegression(GeometricObject):
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
        d_1 = np.vstack([xval**2, xval * yval, yval**2]).T.astype(float)

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
        self._betas = pd.DataFrame(
            data=np.concatenate([self.coef_, [self.intercept_]]),
            index=["A", "B", "C", "D", "E", "F"],
        )

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


class EllipsoidRegression(GeometricObject):
    """
    Least squares fit to a 3D-ellipsoid

        Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1

    References
    ----------
    http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
    """

    coef: pd.Series

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
        zarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float,
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

        z: 1D array-like.
            z-axis data.

        sample_weight: array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self
            the fitted estimator
        """
        zvec = self._simplify(zarr, "Z")
        yvec = self._simplify(yarr, "Y")
        xvec = self._simplify(xarr, "X")
        if yvec.shape[1] != 1 or xvec.shape[1] != 1 or zvec.shape[1] != 1:
            raise ValueError("'xarr', 'yarr' and 'zarr' must be 1D arrays.")
        if yvec.shape[0] != xvec.shape[0] or zvec.shape[0] != xvec.shape[0]:
            raise ValueError("'xarr', 'yarr' and 'zarr' must have same length.")

        # check the weights
        if sample_weight is not None:
            weights = self._simplify(sample_weight, "W").values
            if yvec.shape[0] != weights.shape[0]:
                msg = "'samples_weight' must have same length or the other "
                msg += "arrays."
                raise ValueError(msg)

        # quadratic part of the design matrix
        xval = xvec.values
        yval = yvec.values
        zval = zvec.values
        J = (
            xval**2,
            yval**2,
            zval**2,
            xval * yval,
            xval * yval,
            yval * zval,
            xval,
            yval,
            zval,
        )
        J = np.hstack(J).astype(float)
        K = np.ones_like(xval)  # column of ones

        # np.hstack performs a loop over all samples and creates
        # a row in J for each x,y,z sample:
        coefs = (np.linalg.inv(J.T @ J) @ J.T @ K).astype(float)
        self._betas = pd.DataFrame(
            data=np.concatenate([coefs, [-1]]),
            index=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        )

    @property
    def _quadratic_matrix(self):
        """return the quadratic matrix from the coefficients"""
        matrix = [
            [self.coef_[0], self.coef_[3] / 2, self.coef_[4] / 2],
            [self.coef_[3] / 2, self.coef_[1], self.coef_[5] / 2],
            [self.coef_[4] / 2, self.coef_[5] / 2, self.coef_[2]],
        ]
        return np.array(matrix).astype(float)

    @property
    def _linear_terms(self):
        """return the linear terms of the coefficients"""
        coefs = [self.coef_[5], self.coef_[6], self.coef_[7]]
        return np.atleast_2d(coefs).T.astype(float)

    @property
    def center(self):
        """return the center of the ellipsoid"""
        ctr = -0.5 * (np.linalg.inv(self._quadratic_matrix) @ self._linear_terms)
        return ctr.astype(float).flatten()

    @property
    def axis_major(self):
        """return the axis major of the ellipse"""
        eivals, eivecs = np.linalg.eigh(self._quadratic_matrix)
        a, b, c = eivecs[:, 1]
        x0, y0, z0 = self.center
        d = -(a * x0 + b * y0 + c * z0)

        # TODO
        # the line is in the form a * x0 + b * y0 + c * z0 + d = 0

    @property
    def axis_minor(self):
        """return the axis major of the ellipse"""
        return self._axis_minor

    def predict(
        self,
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
        zarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
    ):
        """
        Fit the model.

        Parameters
        ----------
        xarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None
            x axis data

        yarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
            y axis adata

        zarr: np.ndarray | pd.DataFrame | pd.Series | list | int | float | None = None,
            z axis adata

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
        z: int | float,
    ):
        """
        check whether the point (x, y, z) is inside the Ellipsoid.

        Parameters
        ----------
        x: float
            the x axis coordinate

        y: float
            the y axis coordinate

        z: float
            the z axis coordinate

        Returns
        -------
        i: bool
            True if the provided point is contained by the Ellipse.
        """
        out = self.predict(xarr=x, yarr=y)
        if out is None:
            return False
        if isinstance(out, (pd.DataFrame, pd.Series)):
            out = out.values.astype(float).flatten()
        z0, z1 = out
        return bool((z0 is not None) & (z > min(z0, z1)) & (z <= max(z0, z1)))

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
'''
