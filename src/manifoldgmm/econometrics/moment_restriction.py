from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from datamat import DataMat, DataVec

from ..autodiff import jacobian_operator
from ..autodiff.jax_backend import JacobianOperator
from ..geometry import Manifold, ManifoldPoint

GiMap = Callable[..., Any]
JacobianMap = Callable[..., Any]

if TYPE_CHECKING:  # pragma: no cover - typing assistance
    import jax
    import jax.numpy as jnp

try:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when JAX missing
    jax = cast(Any, None)
    jnp = cast(Any, None)
    JAX_AVAILABLE = False


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
        Backwards-compatibility alias for the vectorized moment map ``g``. New
        code should prefer the explicit ``g`` or ``gi_jax`` keyword arguments.
    g:
        Callable implementing the vectorized moment function ``g``. It receives
        the parameter and the entire dataset (if supplied).
    gi_jax:
        Observation-level JAX-compatible moment function ``g_i``. When
        provided, :class:`MomentRestriction` automatically vectorizes it across
        observations. Requires ``backend='jax'``.
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
    backend:
        ``"numpy"`` (default) uses NumPy/pandas semantics. ``"jax"`` enables
        JAX-friendly operations so the restriction can participate in
        autodiff-backed optimizers. Requires the optional JAX dependency.
    parameter_labels:
        Optional labels (sequence, DataVec/DataMat, or nested structure) naming
        each parameter coordinate. These are flattened, validated against the
        manifold dimension, and exposed via :attr:`parameter_labels` for use in
        inference outputs.
    """

    def __init__(
        self,
        gi_map: GiMap | None = None,
        *,
        g: Callable[[Any, Any], Any] | None = None,
        gi_jax: Callable[[Any, Any], Any] | None = None,
        data: Any | None = None,
        jacobian_map: JacobianMap | None = None,
        manifold: Manifold | None = None,
        argument_adapter: Callable[[Any], Any] | None = None,
        array_adapter: Callable[[Any], Any] | None = None,
        backend: str = "numpy",
        parameter_labels: Any | None = None,
    ):
        self._data = data
        self._jacobian_map = jacobian_map
        self.manifold = manifold
        self._moments_reconstructor: Callable[[Any, bool], Any] | None = None
        self._data_array: Any | None = None
        self._raw_parameter_labels = parameter_labels
        self._parameter_labels: tuple[str, ...] | None = None

        backend_normalized = backend.lower()
        if backend_normalized not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'")
        if backend_normalized == "jax" and not JAX_AVAILABLE:
            raise RuntimeError(
                "MomentRestriction with backend='jax' requires JAX to be installed. "
                "Install ManifoldGMM with the 'jax' extra or specify backend='numpy'."
            )
        self._backend_kind = backend_normalized
        self._xp: Any
        self._linalg: Any
        if backend_normalized == "jax":
            self._xp = jnp
            self._linalg = jnp.linalg
        else:
            self._xp = np
            self._linalg = np.linalg

        self._argument_adapter = argument_adapter or _default_argument_adapter
        if array_adapter is None:
            self._array_adapter: Callable[[Any], Any] = (
                self._default_backend_array_adapter
            )
        else:
            self._array_adapter = array_adapter

        vectorized_map = g or gi_map
        if g is not None and gi_map is not None:
            raise ValueError("Provide only one of 'g' or positional gi_map")
        if vectorized_map is not None and gi_jax is not None:
            raise ValueError("Provide either 'g' or 'gi_jax', not both")

        if vectorized_map is not None:
            self._gi_map = vectorized_map
        else:
            if gi_jax is None:
                raise ValueError("Supply either 'g' (vectorized) or 'gi_jax'")
            if backend_normalized != "jax":
                raise ValueError("gi_jax requires backend='jax'")

            self._data_array = self._normalize_dataset(data)
            if self._data_array is not None:
                self._data_array = jnp.asarray(self._data_array)

            def vectorized(theta: Any, dataset: Any | None = None) -> Any:
                array = self._normalize_dataset(dataset)
                if array is None:
                    if self._data_array is None:
                        raise ValueError("Dataset must be provided when using gi_jax")
                    array = self._data_array
                array = jnp.asarray(array)
                return jax.vmap(lambda obs: gi_jax(theta, obs))(array)

            self._gi_map = vectorized

        self._num_moments: int | None = None
        self._num_observations: int | None = None
        self._parameter_dimension: int | None = None
        self._parameter_shape: tuple[int, ...] | None = None
        self._moment_shape: tuple[int, ...] | None = None
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
    def parameter_labels(self) -> tuple[str, ...] | None:
        """Flattened parameter labels if provided."""

        return self._parameter_labels

    @property
    def observation_counts(self) -> np.ndarray | None:
        """
        Observation counts per moment (shape ``(ℓ,)``).

        The vector reflects missing-data adjustments via ``DataMat.count`` when
        available.
        """

        return (
            None
            if self._observation_counts is None
            else self._observation_counts.copy()
        )

    @property
    def parameter_shape(self) -> tuple[int, ...] | None:
        """Structured shape of the parameter vector if known."""

        return self._parameter_shape

    def gi(self, theta: Any) -> Any:
        """Observation-level moments ``g_i(θ)``."""

        backend_moments = self._evaluate_backend(theta)
        if self._moments_reconstructor is not None:
            return self._moments_reconstructor(backend_moments, False)
        return backend_moments

    def g_bar(self, theta: Any) -> Any:
        """
        Sample average ``\\bar g_N(θ)`` (shape ``(ℓ,)`` or column equivalent).
        """

        backend_moments = self._evaluate_backend(theta)
        mean = self._mean(backend_moments)
        if self._moments_reconstructor is not None:
            return self._moments_reconstructor(mean, True)
        return mean

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

        moments = self._evaluate_backend(theta)
        counts_obj = self._count(moments)
        scale = counts_obj**0.5

        if centered:
            mean = self._mean(moments)
            centered_moments = self._subtract_mean(moments, mean)
        else:
            centered_moments = moments

        scaled = self._divide_columns(centered_moments, scale)
        omega = scaled.T @ scaled
        return self._ensure_psd(omega)

    def Omega_hat(self, theta: Any, *, centered: bool = True) -> Any:
        """Alias retaining the Ω̂ notation."""

        return self.omega_hat(theta, centered=centered)

    def jacobian(self, theta: Any) -> Any:
        """
        Average Jacobian ``D\\bar g_N(θ)`` projected to the manifold tangent space.

        Raises
        ------
        NotImplementedError
            When the instance was built without ``jacobian_map``.
        """

        return self.jacobian_operator(theta)

    def tangent_basis(self, theta: Any, *, tol: float = 1e-12) -> list[Any]:
        """
        Construct a basis for the tangent space ``T_θ\\mathcal{M}``.

        Parameters
        ----------
        theta:
            Evaluation point, either as a :class:`ManifoldPoint` or ambient
            coordinates compatible with the wrapped manifold.
        tol:
            Numerical tolerance used to discard zero-norm or duplicate directions.

        Returns
        -------
        list
            Tangent directions (matching the structure of ``theta``) that span
            ``T_θ\\mathcal{M}``.
        """

        point = self._maybe_point(theta)
        if point is None:
            raise ValueError(
                "tangent_basis requires a manifold-aware evaluation point. "
                "Instantiate MomentRestriction with a manifold."
            )

        manifold_wrapper = self.manifold
        if manifold_wrapper is None:
            raise ValueError(
                "MomentRestriction must define a manifold to compute tangent bases."
            )

        dim_attr = (
            getattr(manifold_wrapper.data, "dim", None)
            if manifold_wrapper.data is not None
            else None
        )
        if callable(dim_attr):
            target_dimension = int(dim_attr())
        elif dim_attr is not None:
            target_dimension = int(dim_attr)
        else:
            parameter_array = self._array_adapter(point.value)
            target_dimension = int(parameter_array.size)

        def zero_like(obj: Any) -> Any:
            if isinstance(obj, tuple):
                return tuple(zero_like(component) for component in obj)
            if isinstance(obj, list):
                return [zero_like(component) for component in obj]
            return np.zeros_like(np.asarray(obj, dtype=float))

        def ambient_basis(obj: Any) -> Any:
            if isinstance(obj, tuple):
                for index, component in enumerate(obj):
                    for component_basis in ambient_basis(component):
                        blocks = [zero_like(part) for part in obj]
                        blocks[index] = component_basis
                        yield tuple(blocks)
                return
            if isinstance(obj, list):
                for index, component in enumerate(obj):
                    for component_basis in ambient_basis(component):
                        blocks = [zero_like(part) for part in obj]
                        blocks[index] = component_basis
                        yield blocks
                return

            array = np.asarray(obj, dtype=float)
            if array.size == 0:
                return
            if array.ndim == 0:
                basis = np.zeros_like(array, dtype=float)
                basis[...] = 1.0
                yield basis
                return
            for multi_index in np.ndindex(array.shape):
                basis = np.zeros_like(array, dtype=float)
                basis[multi_index] = 1.0
                yield basis

        def flatten(obj: Any) -> np.ndarray:
            if isinstance(obj, tuple | list):
                parts = [flatten(component) for component in obj]
                if parts:
                    return np.concatenate(parts)
                return np.array([], dtype=float)
            return np.asarray(obj, dtype=float).reshape(-1)

        basis: list[Any] = []
        normalised_vectors: list[np.ndarray] = []
        for candidate in ambient_basis(point.value):
            projected = point.project_tangent(candidate)
            flat = flatten(projected)
            norm = np.linalg.norm(flat)
            if norm <= tol:
                continue
            direction = flat / norm
            if any(
                np.linalg.norm(direction - existing) <= tol
                for existing in normalised_vectors
            ):
                continue
            basis.append(projected)
            normalised_vectors.append(direction)
            if len(basis) >= target_dimension:
                break

        if len(basis) != target_dimension:
            raise RuntimeError(
                f"Expected to construct {target_dimension} tangent directions; "
                f"collected {len(basis)}."
            )

        return basis

    def jacobian_matrix(
        self, theta: Any, *, basis: list[Any] | None = None, tol: float = 1e-12
    ) -> np.ndarray:
        """
        Return ``D\bar g_N(θ)`` as a dense matrix in the canonical tangent basis.

        Parameters
        ----------
        theta:
            Evaluation point.
        basis:
            Optional tangent basis to use. By default the canonical basis from
            :meth:`tangent_basis` is employed.
        tol:
            Numerical tolerance passed to :meth:`tangent_basis` when ``basis`` is
            ``None``.

        Returns
        -------
        numpy.ndarray
            Matrix whose columns contain the action of the Jacobian on each basis
            vector. The number of rows equals the flattened moment dimension.
        """

        operator = self.jacobian(theta)
        basis_vectors = (
            basis if basis is not None else self.tangent_basis(theta, tol=tol)
        )

        columns: list[np.ndarray] = []
        for direction in basis_vectors:
            image = operator.matvec(direction)
            columns.append(np.asarray(image, dtype=float).reshape(-1))

        if not columns:
            return np.zeros((0, 0), dtype=float)
        return np.column_stack(columns)

    def jacobian_operator(self, theta: Any, *, euclidean: bool = False) -> Any:
        """
        Return the Jacobian evaluated at ``theta``.

        Parameters
        ----------
        theta:
            Evaluation point for the restriction.
        euclidean:
            When ``True`` return the raw Euclidean Jacobian (requires
            ``jacobian_map``). Otherwise return a
            :class:`~manifoldgmm.autodiff.jax_backend.JacobianOperator`
            that acts on the manifold tangent space.

        Raises
        ------
        NotImplementedError
            If ``euclidean=False`` and ``jacobian_map`` is not provided.
        """

        if euclidean:
            if self._jacobian_map is None:
                raise NotImplementedError(
                    "MomentRestriction does not define a jacobian_map; "
                    "raw Euclidean Jacobian unavailable."
                )
            argument = self._prepare_argument(theta)
            return self._call_with_optional_data(self._jacobian_map, argument)

        point = self._maybe_point(theta)
        if self._jacobian_map is None:
            if point is None:
                raise NotImplementedError(
                    "Autodiff Jacobian requires a manifold-aware evaluation point."
                )
            return jacobian_operator(self._autodiff_moment_function(), point)

        argument = self._prepare_argument(theta)
        matrix = self._call_with_optional_data(self._jacobian_map, argument)
        return self._jacobian_operator_from_matrix(matrix, point)

    @classmethod
    def from_datamat(  # noqa: D401
        cls,
        gi_datamat: Callable[[Any, DataMat], DataMat],
        *,
        data: DataMat,
        jacobian_datamat: Callable[[Any, DataMat], DataMat | np.ndarray] | None = None,
        manifold: Manifold | None = None,
        backend: str = "numpy",
        argument_adapter: Callable[[Any], Any] | None = None,
        parameter_labels: Any | None = None,
    ) -> MomentRestriction:
        """Construct a restriction while keeping DataMat as the user-facing type."""

        if backend == "jax" and jacobian_datamat is None:
            raise ValueError("jacobian_datamat must be provided when backend='jax'")

        adapter = _DataMatMomentAdapter(
            data=data,
            backend=backend,
            gi_datamat=gi_datamat,
            jacobian_datamat=jacobian_datamat,
        )

        def gi_backend(argument: Any, _unused: Any = None) -> Any:
            return adapter.backend_moments(argument)

        jacobian_backend: JacobianMap | None
        if jacobian_datamat is not None:

            def jacobian_backend(argument: Any, _unused: Any = None) -> Any:
                return adapter.backend_jacobian(argument)

        else:
            jacobian_backend = None

        restriction = cls(
            g=gi_backend,
            data=data,
            jacobian_map=jacobian_backend,
            manifold=manifold,
            argument_adapter=argument_adapter,
            backend=backend,
            parameter_labels=parameter_labels,
        )
        restriction._moments_reconstructor = adapter.restore
        return restriction

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

    def _evaluate_backend(self, theta: Any) -> Any:
        argument = self._prepare_argument(theta)
        moments = self._call_with_optional_data(self._gi_map, argument)
        if not self._is_jax_backend:
            self._update_metadata(argument, moments)
        return moments

    def _update_metadata(self, argument: Any, moments: Any) -> None:
        if self._parameter_dimension is None:
            parameter_array = self._array_adapter(argument)
            self._parameter_dimension = int(parameter_array.size)
            base_argument = (
                argument.value if isinstance(argument, ManifoldPoint) else argument
            )
            self._parameter_shape = np.asarray(base_argument, dtype=float).shape
            if (
                self._raw_parameter_labels is not None
                and self._parameter_labels is None
            ):
                flat_labels = self._flatten_parameter_labels(self._raw_parameter_labels)
                if len(flat_labels) != self._parameter_dimension:
                    raise ValueError(
                        "parameter_labels length does not match parameter dimension "
                        f"({len(flat_labels)} vs {self._parameter_dimension})"
                    )
                self._parameter_labels = tuple(flat_labels)

        counts_obj = self._count(moments)
        counts_array = np.asarray(counts_obj, dtype=float).reshape(-1)
        self._observation_counts = counts_array
        self._num_moments = int(counts_array.size)
        if counts_array.size:
            self._num_observations = int(np.nanmax(counts_array))
        if self._moment_shape is None:
            mean = self._mean(moments)
            self._moment_shape = np.asarray(mean, dtype=float).shape

    def _mean(self, moments: Any) -> Any:
        if not self._is_jax_backend:
            try:
                return moments.mean(axis=0)
            except AttributeError:
                array = np.asarray(moments, dtype=float)
                if array.ndim == 1:
                    array = array[:, np.newaxis]
                return np.nanmean(array, axis=0)

        xp = self._xp
        array = xp.asarray(moments)
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        return xp.nanmean(array, axis=0)

    def _count(self, moments: Any) -> Any:
        if not self._is_jax_backend:
            try:
                return moments.count(axis=0)
            except AttributeError:
                array = np.asarray(moments, dtype=float)
                if array.ndim == 1:
                    array = array[:, np.newaxis]
                mask = np.isnan(array)
                return (~mask).sum(axis=0)

        xp = self._xp
        array = xp.asarray(moments)
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        mask = xp.isnan(array)
        return xp.sum(xp.logical_not(mask), axis=0)

    def _subtract_mean(self, moments: Any, mean: Any) -> Any:
        if not self._is_jax_backend and isinstance(moments, np.ndarray):
            mean_array = np.asarray(mean, dtype=float)
            if moments.ndim == 1:
                moments = moments[:, np.newaxis]
            if mean_array.ndim == 1:
                mean_array = mean_array.reshape(1, -1)
            return moments - mean_array

        xp = self._xp
        array = xp.asarray(moments)
        mean_array = xp.asarray(mean)
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        if mean_array.ndim == 1:
            mean_array = mean_array.reshape((1, -1))
        return array - mean_array

    def _divide_columns(self, moments: Any, scale: Any) -> Any:
        if not self._is_jax_backend and isinstance(moments, np.ndarray):
            scale_array = np.asarray(scale, dtype=float)
            if scale_array.ndim == 0:
                return moments / scale_array
            if moments.ndim == 1:
                moments = moments[:, np.newaxis]
            return moments / scale_array.reshape(1, -1)

        xp = self._xp
        array = xp.asarray(moments)
        scale_array = xp.asarray(scale)
        if scale_array.ndim == 0:
            return array / scale_array
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        return array / scale_array.reshape((1, -1))

    def _ensure_psd(self, matrix: Any) -> Any:
        """
        Symmetrise and clip eigenvalues to keep ``matrix`` numerically PSD.

        Notes
        -----
        This is a lightweight safeguard implemented with dense linear algebra.
        A future enhancement could wrap Ω̂ in a dedicated PSD manifold and rely
        on its retraction routines instead.
        """

        if not self._is_jax_backend and isinstance(matrix, np.ndarray):
            return _project_psd_numpy(matrix)

        if not self._is_jax_backend:
            try:
                array = np.asarray(matrix, dtype=float)
                projected = _project_psd_numpy(array)
                return matrix.__class__(
                    projected, index=matrix.index, columns=matrix.columns
                )
            except AttributeError:
                return _project_psd_numpy(np.asarray(matrix, dtype=float))

        array = self._xp.asarray(matrix)
        return _project_psd_backend(array, self._xp, self._linalg)

    def _default_backend_array_adapter(self, argument: Any) -> Any:
        if isinstance(argument, ManifoldPoint):
            base = argument.value
        else:
            base = argument
        if self._is_jax_backend:
            return self._xp.asarray(base)
        return np.asarray(base, dtype=float)

    def _normalize_dataset(self, dataset: Any | None) -> Any | None:
        if dataset is None:
            return None
        if isinstance(dataset, DataMat):
            return np.asarray(dataset, dtype=float)
        if isinstance(dataset, pd.DataFrame):
            return dataset.to_numpy(dtype=float)
        return dataset

    @property
    def _is_jax_backend(self) -> bool:
        return self._backend_kind == "jax"

    def _maybe_point(self, theta: Any) -> ManifoldPoint | None:
        if isinstance(theta, ManifoldPoint):
            return theta
        if self.manifold is None:
            return None
        return ManifoldPoint(self.manifold, theta)

    def _reshape_parameter(self, flat: np.ndarray) -> np.ndarray:
        if self._parameter_shape is None:
            return flat
        return flat.reshape(self._parameter_shape)

    def _reshape_moment(self, flat: np.ndarray) -> np.ndarray:
        if self._moment_shape is None:
            return flat
        return flat.reshape(self._moment_shape)

    def _jacobian_operator_from_matrix(
        self, matrix: Any, point: ManifoldPoint | None
    ) -> JacobianOperator:
        matrix_array = np.asarray(matrix, dtype=float)
        if matrix_array.ndim == 1:
            matrix_array = matrix_array.reshape(1, -1)
        elif matrix_array.ndim > 2:
            matrix_array = matrix_array.reshape(matrix_array.shape[0], -1)
        rows, cols = matrix_array.shape

        def matvec(tangent: Any) -> Any:
            tangent_array = np.asarray(
                self._array_adapter(tangent), dtype=float
            ).reshape(cols)
            result = matrix_array @ tangent_array
            return self._reshape_moment(result)

        def T_matvec(covector: Any) -> Any:
            covector_array = np.asarray(covector, dtype=float).reshape(rows)
            gradient_flat = matrix_array.T @ covector_array
            reshaped = self._reshape_parameter(gradient_flat)
            if point is not None:
                return point.project_tangent(reshaped)
            return reshaped

        return JacobianOperator(shape=(rows, cols), matvec=matvec, T_matvec=T_matvec)

    def _flatten_parameter_labels(self, labels: Any) -> list[str]:
        if isinstance(labels, DataVec):
            return [str(idx) for idx in labels.index]
        if isinstance(labels, DataMat):
            flattened: list[str] = []
            for row_label in labels.index:
                for col_label in labels.columns:
                    flattened.append(f"{col_label}[{row_label}]")
            return flattened
        if isinstance(labels, list | tuple):
            collected: list[str] = []
            for item in labels:
                collected.extend(self._flatten_parameter_labels(item))
            return collected
        return [str(labels)]

    def _autodiff_moment_function(self) -> Callable[[ManifoldPoint], Any]:
        def moment_average(point: ManifoldPoint) -> Any:
            raw_argument = self._argument_adapter(point)
            moments = self._call_with_optional_data(self._gi_map, raw_argument)
            return self._mean(moments)

        return moment_average


__all__ = ["MomentRestriction"]


def _project_psd_numpy(matrix: np.ndarray) -> np.ndarray:
    """Return a PSD projection of ``matrix`` while preserving symmetry."""

    return _project_psd_backend(matrix, np, np.linalg)


def _project_psd_backend(matrix: Any, xp: Any, linalg: Any) -> Any:
    symmetrised = 0.5 * (matrix + xp.swapaxes(matrix, -1, -2))
    eigenvalues, eigenvectors = linalg.eigh(symmetrised)
    clipped = xp.clip(eigenvalues, 0.0, None)
    diag = xp.diag(clipped)
    return eigenvectors @ diag @ xp.swapaxes(eigenvectors, -1, -2)


class _DataMatMomentAdapter:
    """Bridge between DataMat moments and array-based backends."""

    def __init__(
        self,
        *,
        data: DataMat,
        backend: str,
        gi_datamat: Callable[[Any, DataMat], DataMat],
        jacobian_datamat: Callable[[Any, DataMat], DataMat | np.ndarray] | None,
    ) -> None:
        self._data = data
        self._backend = backend
        self._gi_datamat = gi_datamat
        self._jacobian_datamat = jacobian_datamat
        self._columns = None
        self._row_index = data.index

    def backend_moments(self, argument: Any) -> Any:
        moments = self._gi_datamat(argument, self._data)
        if not isinstance(moments, DataMat):
            moments = DataMat(moments)
        if self._columns is None:
            self._columns = moments.columns
            self._row_index = moments.index
        values = moments.to_numpy(dtype=float)
        if self._backend == "jax" and JAX_AVAILABLE:
            return jnp.asarray(values)
        return values

    def backend_jacobian(self, argument: Any) -> Any:
        if self._jacobian_datamat is None:
            raise RuntimeError("jacobian_datamat is required for this adapter")
        jac = self._jacobian_datamat(argument, self._data)
        if isinstance(jac, DataMat):
            values = jac.to_numpy(dtype=float)
        else:
            values = np.asarray(jac, dtype=float)
        if self._backend == "jax" and JAX_AVAILABLE:
            return jnp.asarray(values)
        return values

    def restore(self, array: Any, aggregate: bool) -> DataMat:
        values = np.asarray(array, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if aggregate:
            index = pd.RangeIndex(stop=values.shape[0], name="aggregate")
        else:
            expected = (
                len(self._row_index) if self._row_index is not None else values.shape[0]
            )
            if values.shape[0] == expected and self._row_index is not None:
                index = self._row_index
            else:
                index = pd.RangeIndex(stop=values.shape[0])
        return DataMat(values, index=index, columns=self._columns)
