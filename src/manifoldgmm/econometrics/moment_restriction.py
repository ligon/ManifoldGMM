from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..geometry import Manifold, ManifoldPoint

GiMap = Callable[..., Any]
JacobianMap = Callable[..., Any]


def _default_argument_adapter(argument: Any) -> Any:
    if isinstance(argument, ManifoldPoint):
        return argument.value
    return argument


def _default_array_adapter(argument: Any) -> np.ndarray:
    if isinstance(argument, ManifoldPoint):
        base = argument.value
    else:
        base = argument
    array = np.asarray(base, dtype=float)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)


class MomentRestriction:
    """
    Hansen-style moment restriction ``g_i(θ)`` bound to a dataset.

    Parameters
    ----------
    gi_map:
        Callable implementing the observation-level moment function ``g_i``.
        It receives the parameter (possibly adapted by ``argument_adapter``)
        and, if supplied, the dataset.
    data:
        Optional dataset captured alongside the moment function. The object is
        forwarded as the second positional argument when calling ``gi_map`` or
        ``jacobian_map`` if those callables accept it.
    jacobian_map:
        Optional callable returning the average Jacobian ``D\\bar g_N(θ)``.
    manifold:
        Manifold describing the parameter space. When provided and the caller
        passes ambient coordinates, they are projected through
        :class:`~manifoldgmm.geometry.ManifoldPoint`.
    argument_adapter:
        Callable applied to the raw argument before dispatching to
        ``gi_map``/``jacobian_map``. Defaults to returning the ambient value.
    array_adapter:
        Callable producing a flat :class:`numpy.ndarray` from the argument used
        to evaluate ``gi_map``. This is employed to infer the parameter
        dimension when required.
    """

    def __init__(
        self,
        gi_map: GiMap,
        *,
        data: Any | None = None,
        jacobian_map: JacobianMap | None = None,
        manifold: Manifold | None = None,
        argument_adapter: Callable[[Any], Any] | None = None,
        array_adapter: Callable[[Any], np.ndarray] | None = None,
    ):
        self._gi_map = gi_map
        self._data = data
        self._jacobian_map = jacobian_map
        self.manifold = manifold

        self._argument_adapter = argument_adapter or _default_argument_adapter
        self._array_adapter = array_adapter or _default_array_adapter

        self._num_moments: int | None = None
        self._num_observations: int | None = None
        self._parameter_dimension: int | None = None
        self._observation_counts: np.ndarray | None = None

    @property
    def data(self) -> Any | None:
        """Dataset used by the moment restriction."""

        return self._data

    @property
    def num_moments(self) -> int | None:
        """Number of stacked moments ``ℓ`` if observed."""

        return self._num_moments

    @property
    def num_observations(self) -> int | None:
        """Largest available observation count across the sample."""

        return self._num_observations

    @property
    def parameter_dimension(self) -> int | None:
        """Ambient dimension of the parameter vector."""

        return self._parameter_dimension

    @property
    def observation_counts(self) -> np.ndarray | None:
        """
        Observation counts per moment (shape ``(ℓ,)``).

        The vector reflects missing-data adjustments via ``DataMat.count`` when
        available.
        """

        return None if self._observation_counts is None else self._observation_counts.copy()

    def gi(self, theta: Any) -> Any:
        """
        Evaluate the observation-level moments ``g_i(θ)``.

        Parameters
        ----------
        theta:
            Parameter value for which to evaluate the restriction. The argument
            may already be a :class:`ManifoldPoint`; otherwise, if a ``manifold``
            was supplied, it is projected onto that manifold.
        """

        argument = self._prepare_argument(theta)
        moments = self._call_with_optional_data(self._gi_map, argument)
        self._update_metadata(argument, moments)
        return moments

    def g_bar(self, theta: Any) -> Any:
        """
        Sample average ``\\bar g_N(θ)`` (shape ``(ℓ,)`` or column equivalent).
        """

        moments = self.gi(theta)
        return self._mean(moments)

    def gN(self, theta: Any) -> Any:
        """Alias for :meth:`g_bar` preserving classical notation."""

        return self.g_bar(theta)

    def omega_hat(self, theta: Any, *, centered: bool = True) -> Any:
        """
        Empirical covariance estimate ``Ω̂(θ)`` (shape ``(ℓ, ℓ)``).

        Parameters
        ----------
        theta:
            Evaluation point.
        centered:
            When ``True`` (default) subtract ``\\bar g_N(θ)`` before scaling.
        """

        moments = self.gi(theta)
        counts_obj = self._count(moments)
        scale = counts_obj ** 0.5

        if centered:
            mean = self._mean(moments)
            centered_moments = self._subtract_mean(moments, mean)
        else:
            centered_moments = moments

        scaled = self._divide_columns(centered_moments, scale)
        return scaled.T @ scaled

    def Omega_hat(self, theta: Any, *, centered: bool = True) -> Any:
        """Alias retaining the Ω̂ notation."""

        return self.omega_hat(theta, centered=centered)

    def jacobian(self, theta: Any) -> Any:
        """
        Average Jacobian ``D\\bar g_N(θ)`` if available.

        Raises
        ------
        NotImplementedError
            When the instance was built without ``jacobian_map``.
        """

        if self._jacobian_map is None:
            raise NotImplementedError(
                "MomentRestriction was constructed without jacobian_map; "
                "supply one or override jacobian()."
            )
        argument = self._prepare_argument(theta)
        return self._call_with_optional_data(self._jacobian_map, argument)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_argument(self, theta: Any) -> Any:
        if isinstance(theta, ManifoldPoint):
            if self.manifold is not None and theta.manifold is not self.manifold:
                raise ValueError("theta belongs to a different manifold instance")
            raw_argument = theta
        elif self.manifold is not None:
            raw_argument = ManifoldPoint(self.manifold, theta)
        else:
            raw_argument = theta
        return self._argument_adapter(raw_argument)

    def _call_with_optional_data(self, fn: Callable[..., Any], argument: Any) -> Any:
        if self._data is None:
            return fn(argument)
        try:
            return fn(argument, self._data)
        except TypeError:
            return fn(argument)

    def _update_metadata(self, argument: Any, moments: Any) -> None:
        if self._parameter_dimension is None:
            parameter_array = self._array_adapter(argument)
            self._parameter_dimension = int(parameter_array.size)

        counts_obj = self._count(moments)
        counts_array = np.asarray(counts_obj, dtype=float).reshape(-1)
        self._observation_counts = counts_array
        self._num_moments = int(counts_array.size)
        if counts_array.size:
            self._num_observations = int(np.nanmax(counts_array))

    @staticmethod
    def _mean(moments: Any) -> Any:
        try:
            return moments.mean(axis=0)
        except AttributeError:
            array = np.asarray(moments, dtype=float)
            if array.ndim == 1:
                array = array[:, np.newaxis]
            return np.nanmean(array, axis=0)

    @staticmethod
    def _count(moments: Any) -> Any:
        try:
            return moments.count(axis=0)
        except AttributeError:
            array = np.asarray(moments, dtype=float)
            if array.ndim == 1:
                array = array[:, np.newaxis]
            mask = np.isnan(array)
            return (~mask).sum(axis=0)

    @staticmethod
    def _subtract_mean(moments: Any, mean: Any) -> Any:
        if isinstance(moments, np.ndarray):
            mean_array = np.asarray(mean, dtype=float)
            if moments.ndim == 1:
                moments = moments[:, np.newaxis]
            if mean_array.ndim == 1:
                mean_array = mean_array.reshape(1, -1)
            return moments - mean_array
        return moments - mean

    @staticmethod
    def _divide_columns(moments: Any, scale: Any) -> Any:
        if isinstance(moments, np.ndarray):
            scale_array = np.asarray(scale, dtype=float)
            if scale_array.ndim == 0:
                return moments / scale_array
            if moments.ndim == 1:
                moments = moments[:, np.newaxis]
            return moments / scale_array.reshape(1, -1)
        return moments / scale


__all__ = ["MomentRestriction"]

