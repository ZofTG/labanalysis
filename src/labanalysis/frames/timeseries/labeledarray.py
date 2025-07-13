"""
labeledarray module
"""

# -*- coding: utf-8 -*-


#! IMPORTS


import numpy as np
import pandas as pd

from ..signalprocessing import fillna as sp_fillna

__all__ = ["LabeledArray"]


class LabeledArray:
    """
    A 2D array with labeled rows and columns, supporting flexible indexing,
    arithmetic operations, and conversion to/from pandas DataFrames.
    """

    def __init__(self, data, index, columns):
        """
        Initialize a LabeledArray.

        Parameters
        ----------
        data : array-like
            2D data array.
        index : list
            Row labels.
        columns : list
            Column labels.

        Raises
        ------
        ValueError
            If the shape of data does not match the length of index and
            columns, or if data cannot be cast to float.
        """
        try:
            self._data = np.array(data, dtype=float)
        except Exception as exc:
            raise ValueError("data must be a 2D array castable to float.") from exc
        if self._data.shape != (len(index), len(columns)):
            raise ValueError("Shape of data does not match length of index and columns")
        self.index = list(index)
        self.columns = list(columns)
        self._row_map = {label: i for i, label in enumerate(self.index)}
        self._col_map = {label: i for i, label in enumerate(self.columns)}
        self._check_consistency()

    def _get_row_indices(self, row_key):
        """Internal: Get row indices from row_key."""
        if isinstance(row_key, (list, np.ndarray)):
            return [self._row_map[r] for r in row_key]
        elif isinstance(row_key, slice):
            start = self._row_map.get(row_key.start, None) if row_key.start else None
            stop = self._row_map.get(row_key.stop, None)
            if stop is not None:
                stop += 1
            return slice(start, stop, row_key.step)
        elif isinstance(row_key, (str, float, int)):
            return self._row_map[row_key]
        else:
            raise KeyError(f"Invalid row key: {row_key}")

    def _get_col_indices(self, col_key):
        """Internal: Get column indices from col_key."""
        if isinstance(col_key, (list, np.ndarray)):
            return [self._col_map[c] for c in col_key]
        elif isinstance(col_key, slice):
            start = self._col_map.get(col_key.start, None) if col_key.start else None
            stop = self._col_map.get(col_key.stop, None)
            if stop is not None:
                stop += 1
            return slice(start, stop, col_key.step)
        elif isinstance(col_key, (str, float, int)):
            return self._col_map[col_key]
        else:
            raise KeyError(f"Invalid column key: {col_key}")

    def __getitem__(self, key, indices=False):
        """
        Get item(s) from the array using labels or indices.

        Parameters
        ----------
        key : label, list, tuple, or slice
            Row/column label(s) or indices.
        indices : bool, optional
            If True, interpret key as indices.

        Returns
        -------
        np.ndarray
            Selected data.
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            row_idx = row_key if indices else self._get_row_indices(row_key)
            col_idx = col_key if indices else self._get_col_indices(col_key)
            return self._data[row_idx, col_idx]
        elif isinstance(key, slice):
            return self._data[key, :]
        elif isinstance(key, str) and key in self.columns:
            return self._data[:, self._col_map[key]]
        else:
            row_idx = key if indices else self._get_row_indices(key)
            return self._data[row_idx, :]  # type: ignore

    def __setitem__(self, key, value, indices=False):
        """
        Set item(s) in the array using labels or indices.

        Parameters
        ----------
        key : label, list, tuple, or slice
            Row/column label(s) or indices.
        value : array-like
            Value(s) to set.
        indices : bool, optional
            If True, interpret key as indices.

        Raises
        ------
        ValueError
            If the value is not a list or array with a length equal to the number of columns.
        """
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key

            if indices:
                row_idx, col_idx = row_key, col_key
            else:
                if row_key not in self._row_map:
                    self.index.append(row_key)
                    self._row_map[row_key] = len(self.index) - 1
                    new_row = np.full((1, self._data.shape[1]), np.nan)
                    self._data = np.vstack([self._data, new_row])

                if col_key not in self._col_map:
                    self.columns.append(col_key)
                    self._col_map[col_key] = len(self.columns) - 1
                    new_col = np.full((self._data.shape[0], 1), np.nan)
                    self._data = np.hstack([self._data, new_col])

                row_idx = self._row_map[row_key]
                col_idx = self._col_map[col_key]

            self._data[row_idx, col_idx] = value
        else:
            row_label = key

            if indices:
                row_idx = row_label
            else:
                if row_label not in self._row_map:
                    self.index.append(row_label)
                    self._row_map[row_label] = len(self.index) - 1
                    new_row = np.full((1, self._data.shape[1]), np.nan)
                    self._data = np.vstack([self._data, new_row])
                row_idx = self._row_map[row_label]

            if (
                isinstance(value, (list, np.ndarray))
                and len(value) == self._data.shape[1]
            ):
                self._data[row_idx, :] = value
            else:
                raise ValueError(
                    "Value must be a list or array with length equal to number of columns"
                )
        self._check_consistency()

    def __getattr__(self, name):
        return self._data.__getattribute__(name)

    def drop(self, label, axis=0):
        """
        Remove a row or column by label.

        Parameters
        ----------
        label : str
            Label to drop.
        axis : int, optional
            0 for row, 1 for column (default: 0).

        Raises
        ------
        KeyError
            If label not found.
        ValueError
            If axis is not 0 or 1.
        """
        if axis == 0:
            if label not in self._row_map:
                raise KeyError(f"Row label '{label}' not found")
            idx = self._row_map[label]
            self.index.pop(idx)
            self._data = np.delete(self._data, idx, axis=0)
        elif axis == 1:
            if label not in self._col_map:
                raise KeyError(f"Column label '{label}' not found")
            idx = self._col_map[label]
            self.columns.pop(idx)
            self._data = np.delete(self._data, idx, axis=1)
        else:
            raise ValueError("Axis must be 0 (row) or 1 (column)")
        self._row_map = {label: i for i, label in enumerate(self.index)}
        self._col_map = {label: i for i, label in enumerate(self.columns)}
        self._check_consistency()

    def fillna(self, value=None, n_regressors=None, inplace=False):
        """
        Return a copy with NaNs replaced by the specified value or using advanced imputation.

        Parameters
        ----------
        value : float or int or None, optional
            Value to use for NaNs. If None, use interpolation or regression.
        n_regressors : int or None, optional
            Number of regressors to use for regression-based imputation. If
            None, use cubic spline interpolation.
        inplace : bool, optional
            If True, fill in place (for DataFrame/Series). If False, return a
            new object.

        Returns
        -------
        LabeledArray
            Filled array.
        """
        filled = pd.DataFrame(
            sp_fillna(
                self.to_dataframe(),
                value=value,
                n_regressors=n_regressors,
                inplace=False,
            )
        )
        if inplace:
            self._data = filled.values.astype(float)
        else:
            return self.__class__(
                filled.values, filled.index.tolist(), filled.columns.tolist()
            )

    def isna(self):
        """
        Return a boolean array indicating NaNs.

        Returns
        -------
        np.ndarray
            Boolean mask of NaNs.
        """
        return np.isnan(self._data)

    def to_dataframe(self):
        """
        Convert to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame representation.
        """
        return pd.DataFrame(self._data, index=self.index, columns=self.columns)

    @classmethod
    def from_dataframe(cls, df):
        """
        Create a LabeledArray from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        LabeledArray
            A LabeledArray created from the DataFrame.
        """
        return cls(df.values, df.index.tolist(), df.columns.tolist())

    def _binary_op(self, other, op, align=True):
        """
        Internal: Perform binary operation with another LabeledArray or
        scalar.

        Parameters
        ----------
        other : LabeledArray or scalar
            The other LabeledArray or scalar to operate with.
        op : callable
            Numpy operation.
        align : bool, optional
            Align on labels (default: True).

        Returns
        -------
        LabeledArray
            The result of the operation as a new LabeledArray.

        Raises
        ------
        TypeError
            If the operand type is unsupported.
        """
        if isinstance(other, LabeledArray):
            if align:
                all_rows = sorted(set(self.index) | set(other.index))
                all_cols = sorted(set(self.columns) | set(other.columns))

                def build_aligned_data(arr):
                    data = np.full((len(all_rows), len(all_cols)), np.nan)
                    for i, row in enumerate(all_rows):
                        if row in arr._row_map:
                            for j, col in enumerate(all_cols):
                                if col in arr._col_map:
                                    data[i, j] = arr._data[
                                        arr._row_map[row], arr._col_map[col]
                                    ]
                    return data

                left_data = build_aligned_data(self)
                right_data = build_aligned_data(other)
                result_data = op(left_data, right_data)
                result = LabeledArray(result_data, all_rows, all_cols)
                result._check_consistency()
                return result
            else:
                result_data = op(self._data, other._data)
                result = LabeledArray(result_data, self.index, self.columns)
            result._check_consistency()
            return result

        elif np.isscalar(other):
            result_data = op(self._data, other)
            result = LabeledArray(result_data, self.index, self.columns)
            result._check_consistency()
            return result
        else:
            raise TypeError("Unsupported operand type")

    def __add__(self, other):
        """Element-wise addition."""
        return self._binary_op(other, np.add)

    def __iadd__(self, other):
        """In-place element-wise addition."""
        result = self._binary_op(other, np.add)
        self._data = result._data
        return self

    def __radd__(self, other):
        """Right element-wise addition."""
        return self._binary_op(other, np.add)

    def __sub__(self, other):
        """Element-wise subtraction."""
        return self._binary_op(other, np.subtract)

    def __isub__(self, other):
        """In-place element-wise subtraction."""
        result = self._binary_op(other, np.subtract)
        self._data = result._data
        return self

    def __rsub__(self, other):
        """Right element-wise subtraction."""
        # right subtraction: other - self
        return self._binary_op(other, lambda x, y: np.subtract(y, x))

    def __mul__(self, other):
        """Element-wise multiplication."""
        return self._binary_op(other, np.multiply)

    def __imul__(self, other):
        """In-place element-wise multiplication."""
        result = self._binary_op(other, np.multiply)
        self._data = result._data
        return self

    def __rmul__(self, other):
        """Right element-wise multiplication."""
        return self._binary_op(other, np.multiply)

    def __truediv__(self, other):
        """Element-wise division."""
        return self._binary_op(other, np.divide)

    def __itruediv__(self, other):
        """In-place element-wise division."""
        result = self._binary_op(other, np.divide)
        self._data = result._data
        return self

    def __rtruediv__(self, other):
        """Right element-wise division."""
        # right division: other / self
        return self._binary_op(other, lambda x, y: np.divide(y, x))

    def __pow__(self, other):
        """Element-wise exponentiation."""
        return self._binary_op(other, np.power, align=False)

    def __ipow__(self, other):
        """In-place element-wise exponentiation."""
        result = self._binary_op(other, np.power, align=False)
        self._data = result._data
        return self

    def __rpow__(self, other):
        """Right element-wise exponentiation."""
        # right power: other ** self
        return self._binary_op(other, lambda x, y: np.power(y, x), align=False)

    def __repr__(self):
        """String representation of the LabeledArray."""
        header = "\t" + "\t".join(map(str, self.columns))
        rows = "\n".join(
            f"{idx}\t"
            + "\t".join(f"{val:.2f}" if not np.isnan(val) else "nan" for val in row)
            for idx, row in zip(self.index, self._data)
        )
        return f"{header}\n{rows}"

    def strip(self, axis=0, inplace=False):
        """
        Remove leading/trailing rows or columns that are all NaN.

        Parameters
        ----------
        axis : int, optional
            0 for rows, 1 for columns (default: 0).
        inplace : bool, optional
            If True, modifies self. If False, returns a new stripped object.

        Returns
        -------
        LabeledArray or None
            Stripped array if inplace is False, otherwise None.
        """
        if inplace:
            obj = self
        else:
            obj = self.copy()

        if axis == 0:
            mask = np.all(np.isnan(obj._data), axis=1)
            start = 0
            while start < len(mask) and mask[start]:
                start += 1
            end = len(mask)
            while end > start and mask[end - 1]:
                end -= 1
            obj._data = obj._data[start:end]
            obj.index = obj.index[start:end]
            obj._row_map = {label: i for i, label in enumerate(obj.index)}
        elif axis == 1:
            mask = np.all(np.isnan(obj._data), axis=0)
            start = 0
            while start < len(mask) and mask[start]:
                start += 1
            end = len(mask)
            while end > start and mask[end - 1]:
                end -= 1
            obj._data = obj._data[:, start:end]
            obj.columns = obj.columns[start:end]
            obj._col_map = {label: i for i, label in enumerate(obj.columns)}
        obj._check_consistency()

        if not inplace:
            return obj

    def _check_consistency(self):
        """Internal: Check if data shape matches index/columns."""
        if self._data.shape != (len(self.index), len(self.columns)):
            raise ValueError(
                "Inconsistent shape: data shape does not match index and columns length"
            )

    def copy(self):
        """
        Return a deep copy of the LabeledArray.

        Returns
        -------
        LabeledArray
            A new LabeledArray object with the same data, index, and columns.
        """
        return LabeledArray(
            data=self._data.copy(),
            index=self.index,
            columns=self.columns,
        )

    def apply(self, func, axis=0, inplace=False, *args, **kwargs):
        """
        Apply a function to the underlying data.

        Parameters
        ----------
        func : callable or ProcessingPipeline
            Function, class, or method to apply to the data, or a ProcessingPipeline.
        axis : int, optional
            0 to apply by row, 1 to apply by column (default: 0).
        inplace : bool, optional
            If True, modifies self. If False, returns a new object.
        *args, **kwargs : additional arguments to pass to func.

        Returns
        -------
        LabeledArray or result of func
            If inplace is False, returns a new LabeledArray with the function applied.
            If inplace is True, returns None.
        """
        if hasattr(func, "apply") and callable(getattr(func, "apply", None)):
            return func.apply(self, inplace=inplace, *args, **kwargs)
        obj = self if inplace else self.copy()
        if axis == 0:
            result = np.apply_along_axis(func, axis, obj._data, *args, **kwargs)
            if result.ndim == 1:
                arr = self.__class__(
                    data=result[:, np.newaxis],
                    index=obj.index,
                    columns=["applied"],
                )
            else:
                arr = self.__class__(data=result, index=obj.index, columns=obj.columns)
        elif axis == 1:
            result = np.apply_along_axis(func, axis, obj._data, *args, **kwargs)
            if result.ndim == 1:
                arr = self.__class__(
                    data=result[np.newaxis, :],
                    index=["applied"],
                    columns=obj.columns,
                )
            else:
                arr = self.__class__(data=result, index=obj.index, columns=obj.columns)
        else:
            raise ValueError("axis must be 0 (row) or 1 (column)")
        if inplace:
            self._data = arr._data
            self.index = arr.index
            self.columns = arr.columns
            self._row_map = getattr(arr, "_row_map", self._row_map)
            self._col_map = getattr(arr, "_col_map", self._col_map)
            return self
        return arr
