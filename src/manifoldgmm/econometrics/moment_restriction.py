from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datamat import DataMat, DataVec

from ..autodiff import jacobian_operator
from ..autodiff.jax_backend import JacobianOperator
from ..geometry import Manifold, ManifoldPoint

GiMap = Callable[..., Any]
JacobianMap = Callable[..., Any]


class _VmapVectorizer:
    """Picklable callable that vectorizes a per-observation JAX moment function."""

    def __init__(
        self, gi_jax: Callable[..., Any], restriction: MomentRestriction
    ) -> None:
        self._gi_jax = gi_jax
        self._restriction = restriction

    def __call__(self, theta: Any, dataset: Any | None = None) -> Any:
        array = self._restriction._normalize_dataset(dataset)
        if array is None:
            if self._restriction._data_array is None:
                raise ValueError("Dataset must be provided when using gi_jax")
            array = self._restriction._data_array
        array = jnp.asarray(array)
        gi_jax = self._gi_jax
        return jax.vmap(lambda obs: gi_jax(theta, obs))(array)


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
        autodiff-backed optimizers. Requires JAX to be installed.
    parameter_labels:
        Optional labels (sequence, DataVec/DataMat, or nested structure) naming
        each parameter coordinate. These are flattened, validated against the
        manifold dimension, and exposed via :attr:`parameter_labels` for use in
        inference outputs.
    weights:
        Optional per-observation weights ``w_i`` with ``E[w_i] = 1`` used for
        bootstrap-style reweighting of the moment mean.  See :attr:`weights`.
    clusters:
        Optional observation-level cluster identifiers (1-D array-like of length
        ``N``).  When supplied, :meth:`omega_hat` aggregates centered moment
        contributions to per-cluster sums before forming :math:`\\hat\\Omega`,
        yielding a cluster-robust estimate that drives the two-step weighting,
        the sandwich SEs (``GMMResult.tangent_covariance``), and the Hansen J
        through the same locus.  Cluster labels may be any hashable type; they
        are normalised to integer codes internally.  Default ``None`` reproduces
        the i.i.d. behaviour bit-for-bit.  See :meth:`with_clusters`.
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
        weights: Any | None = None,
        clusters: Any | None = None,
    ):
        self._data = data
        self._jacobian_map = jacobian_map
        self.manifold = manifold
        self._moments_reconstructor: Callable[[Any, bool], Any] | None = None
        self._data_array: Any | None = None
        self._raw_parameter_labels = parameter_labels
        self._parameter_labels: tuple[str, ...] | None = None
        self._weights: Any | None = weights
        self._clusters: Any | None = clusters
        # Integer codes derived from ``self._clusters`` (cached lazily); paired
        # with the number of unique clusters ``G``.  Recomputed whenever
        # ``_clusters`` changes (see ``with_clusters``).
        self._cluster_codes: np.ndarray | None = None
        self._num_clusters: int | None = None
        # Phase B-minimal (PR #49): the v2 GMM synthesis path attaches
        # the DGP here so omega_hat can delegate to
        # dgp.sample_distribution.moment_covariance(...).  None for v1
        # callers; the omega_hat ``getattr(self, "_dgp", None)`` check
        # falls through to the existing v1 formula when this is None.
        self._dgp: Any = None

        backend_normalized = backend.lower()
        if backend_normalized not in {"numpy", "jax"}:
            raise ValueError("backend must be 'numpy' or 'jax'")
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

            self._gi_map = _VmapVectorizer(gi_jax, self)

        self._num_moments: int | None = None
        self._num_observations: int | None = None
        self._parameter_dimension: int | None = None
        self._parameter_shape: tuple[int, ...] | None = None
        self._moment_shape: tuple[int, ...] | None = None
        self._observation_counts: np.ndarray | None = None
        self._metadata_argument: Any | None = None
        self._metadata_moments: Any | None = None

    @property
    def data(self) -> Any | None:
        """Dataset used by the moment restriction."""

        return self._data

    @property
    def weights(self) -> Any | None:
        """Bootstrap weights applied to observations (read-only).

        When ``None`` (default) the restriction uses equal weights, i.e.,
        the ordinary sample mean.  When set, the mean becomes

        .. math::

            \\bar g^*_N(\\theta) = \\frac{1}{n}\\sum_{i=1}^n w_i\\, g_i(\\theta).

        Division by :math:`n` (not :math:`\\sum w_i`) ensures unbiasedness
        when :math:`E[w_i] = 1` (Davidson & Flachaire, 2008).
        """

        return self._weights

    def _set_weights(self, weights: Any) -> MomentRestriction:
        """Internal counterpart to :meth:`with_weights` that does not warn.

        Used by library-internal call sites (notably
        :mod:`manifoldgmm.econometrics.bootstrap`) where the v1
        weighted-clone idiom remains the active code path.  Public
        callers should construct the v2 DGP-side equivalent;
        :meth:`with_weights` exposes that recommendation as a
        :class:`DeprecationWarning`.
        """

        import copy

        clone = copy.copy(self)
        clone._weights = weights
        return clone

    def with_weights(self, weights: Any) -> MomentRestriction:
        """Return a shallow copy carrying the supplied bootstrap weights.

        .. deprecated:: 0.4
            Sampling-design state (weights, cluster ids) now lives on
            the DGP's :class:`~dgp_protocol.SamplingDesign`.  Use
            ``EmpiricalDGP(observation=X, sampling=IIDSampling(weights=...))``
            (or ``ClusteredSampling(cluster_ids=..., weights=...)``) with
            ``GMM(moment_func=g, dgp=dgp, ...)``.  This method emits a
            :class:`DeprecationWarning` for one minor release and will
            be removed in v0.5.  See ``docs/design/v2_dgp.org`` and
            ManifoldGMM issue #47.

        The returned instance shares the dataset, manifold, moment map, and
        all cached metadata with the original --- only the weights differ.
        This makes it cheap to create bootstrap replicates without redundant
        re-initialization.

        Parameters
        ----------
        weights:
            Observation-level weights with ``E[w_i] = 1``.  Shape should
            broadcast against ``(n,)``.
        """

        import warnings

        warnings.warn(
            "MomentRestriction.with_weights is deprecated and will be "
            "removed in v0.5.  Sampling-design state (weights, cluster "
            "ids) now belongs to the DGP's SamplingDesign.  Construct the "
            "v2 equivalent: EmpiricalDGP(observation=X, "
            "sampling=IIDSampling(weights=...)) (or "
            "ClusteredSampling(cluster_ids=..., weights=...)) with "
            "GMM(moment_func=g, dgp=dgp, ...).  See "
            "docs/design/v2_dgp.org and issue #47.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._set_weights(weights)

    @property
    def clusters(self) -> Any | None:
        """Observation-level cluster ids associated with the moments.

        When ``None`` (default) :meth:`omega_hat` uses the i.i.d. estimator.
        When set, the centered moment contributions are aggregated to
        per-cluster sums before forming :math:`\\hat\\Omega`, producing a
        cluster-robust covariance that automatically propagates to the
        two-step weighting, the sandwich SEs, and the Hansen J.
        """

        return self._clusters

    def _set_clusters(self, cluster_ids: Any) -> MomentRestriction:
        """Internal counterpart to :meth:`with_clusters` that does not warn.

        Used by library-internal call sites where the v1 clustered-
        clone idiom remains the active code path (see
        :meth:`_set_weights` for the same pattern on the weights side).
        Public callers should construct the v2 DGP-side equivalent;
        :meth:`with_clusters` exposes that recommendation as a
        :class:`DeprecationWarning`.
        """

        import copy

        clone = copy.copy(self)
        clone._clusters = cluster_ids
        # Drop any cached integer codes inherited from ``self`` so the clone
        # recomputes them lazily against the new assignment.
        clone._cluster_codes = None
        clone._num_clusters = None
        return clone

    def with_clusters(self, cluster_ids: Any) -> MomentRestriction:
        """Return a shallow copy carrying the supplied cluster identifiers.

        .. deprecated:: 0.4
            Cluster identifiers now live on the DGP's
            :class:`~dgp_protocol.SamplingDesign`.  Use
            ``EmpiricalDGP(observation=X, sampling=ClusteredSampling(cluster_ids=ids))``
            with ``GMM(moment_func=g, dgp=dgp, ...)`` instead.  This
            method emits a :class:`DeprecationWarning` for one minor
            release and will be removed in v0.5.  See
            ``docs/design/v2_dgp.org`` and ManifoldGMM issue #47.

        Mirrors :meth:`with_weights`: the returned instance shares the
        dataset, manifold, moment map, weights, and cached metadata with the
        original --- only the cluster assignment differs.  Pass ``None`` to
        revert to the i.i.d. estimator.

        Parameters
        ----------
        cluster_ids:
            1-D array-like of length ``N`` carrying a cluster label for each
            observation.  Labels may be of any hashable type and are
            internally normalised to integer codes.
        """

        import warnings

        warnings.warn(
            "MomentRestriction.with_clusters is deprecated and will be "
            "removed in v0.5.  Sampling-design state (cluster ids, "
            "weights) now belongs to the DGP's SamplingDesign.  "
            "Construct the v2 equivalent: EmpiricalDGP(observation=X, "
            "sampling=ClusteredSampling(cluster_ids=ids)) with "
            "GMM(moment_func=g, dgp=dgp, ...).  See "
            "docs/design/v2_dgp.org and issue #47.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._set_clusters(cluster_ids)

    @property
    def num_moments(self) -> int | None:
        """Number of stacked moments ``ℓ`` if observed."""

        self._ensure_metadata()
        return self._num_moments

    @property
    def num_observations(self) -> int | None:
        """Largest available observation count across the sample."""

        self._ensure_metadata()
        return self._num_observations

    @property
    def parameter_dimension(self) -> int | None:
        """Ambient dimension of the parameter vector."""

        self._ensure_metadata()
        return self._parameter_dimension

    @property
    def parameter_labels(self) -> tuple[str, ...] | None:
        """Flattened parameter labels if provided."""

        self._ensure_metadata()
        return self._parameter_labels

    @property
    def observation_counts(self) -> np.ndarray | None:
        """
        Observation counts per moment (shape ``(ℓ,)``).

        The vector reflects missing-data adjustments via ``DataMat.count`` when
        available.
        """

        self._ensure_metadata()
        return (
            None
            if self._observation_counts is None
            else self._observation_counts.copy()
        )

    @property
    def parameter_shape(self) -> tuple[int, ...] | None:
        """Structured shape of the parameter vector if known."""

        self._ensure_metadata()
        return self._parameter_shape

    def gi(self, theta: Any) -> Any:
        """Observation-level moments ``g_i(θ)``."""

        backend_moments = self._evaluate_backend(theta)
        if self._moments_reconstructor is not None:
            return self._moments_reconstructor(backend_moments, False)
        return backend_moments

    def moment_contributions(self, theta: Any) -> Any:
        """Return the per-observation moment matrix ``(N, ℓ)`` at ``theta``.

        This is the raw backend array consumed by :meth:`g_bar` and
        :meth:`omega_hat`: each row ``i`` carries the moment vector
        ``g_i(θ)``.  Callers should treat the returned object as read-only;
        in-place mutation will silently corrupt cached metadata used by
        subsequent calls.  Equivalent to :meth:`gi` but returns the raw
        backend matrix without the optional moments reconstructor wrapping.
        """

        return self._evaluate_backend(theta)

    def g_bar(self, theta: Any) -> Any:
        r"""
        Scaled sample average :math:`\frac{1}{\sqrt{N_k}}\sum_i g_{ik}(\theta)`
        (shape ``(\ell,)`` or column equivalent).

        Each moment *k* is divided by :math:`\sqrt{N_k}` where :math:`N_k` is
        the number of non-missing observations for that moment, so that the
        resulting vector is :math:`O(1)` under the null.
        """

        backend_moments = self._evaluate_backend(theta)
        mean = self._mean(backend_moments)
        counts = self._count(backend_moments)
        sqrt_n = counts**0.5
        scaled = mean * sqrt_n
        if self._moments_reconstructor is not None:
            return self._moments_reconstructor(scaled, True)
        return scaled

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

        Notes
        -----
        When the restriction carries cluster ids (see :meth:`with_clusters`),
        the centered, per-moment-rescaled contributions are aggregated to
        per-cluster sums :math:`S_c = \\sum_{i\\in c}(g_{ik}-\\bar g_k)/\\sqrt{N_k}`
        before forming :math:`\\hat\\Omega = S^{\\top}S`.  With every cluster
        of size one this is byte-identical to the i.i.d. estimator
        (``XᵀX`` is permutation invariant).

        Phase B-minimal: when a DGP is attached to this restriction
        (``self._dgp`` set, typical for v2-constructed GMMs), the
        computation is delegated to
        ``self._dgp.sample_distribution.moment_covariance(theta,
        self.gi_jax, centered=centered)`` -- the closed-form formula on
        the DGP's :class:`SamplingDesign`.  The DGP-side formula is
        byte-parity with the v1 formula below on shared inputs
        (verified by the DGP_Protocol parity tests at 1e-12 tolerance),
        so this delegation is observationally equivalent while
        single-sourcing the sampling-design knowledge to the DGP.
        See ManifoldGMM issue #47 for the larger
        ``with_clusters`` / ``with_weights`` deprecation that follows.
        """

        dgp = getattr(self, "_dgp", None)
        if dgp is not None and hasattr(dgp, "_sd_moment_covariance"):
            # ``self._gi_map`` is the user's vectorized moment callable
            # ``(theta, data) -> (N, k)``; this is what the v2 GMM
            # synthesis path always sets (via ``g=moment_func``).  For
            # v1 callers it could be a ``_VmapVectorizer`` wrapper, but
            # those don't have ``_dgp`` attached (no delegation fires).
            #
            # ``_prepare_argument`` mirrors v1's ``_evaluate_backend``:
            # unwraps ``ManifoldPoint``, applies ``_argument_adapter``,
            # so the user's moment function receives the raw parameter
            # array (not the manifold-wrapped form).
            #
            # We dispatch directly to ``_sd_moment_covariance`` (not via
            # ``dgp.sample_distribution.moment_covariance``) so that
            # ``AnalyticUnavailable`` falls through here to the v1
            # formula below -- the SampleDistribution view would otherwise
            # catch it and pursue an adaptive-MC fallback, which is
            # slow, non-deterministic, and breaks JAX tracing for
            # ParametricDGP draws that return traced arrays.
            from dgp_protocol import AnalyticUnavailable

            adapted_theta = self._prepare_argument(theta)
            try:
                return dgp._sd_moment_covariance(
                    adapted_theta, self._gi_map, centered=centered
                )
            except AnalyticUnavailable:
                pass  # fall through to v1 formula below

        moments = self._evaluate_backend(theta)
        counts_obj = self._count(moments)
        scale = counts_obj**0.5

        if centered:
            mean = self._mean(moments)
            centered_moments = self._subtract_mean(moments, mean)
        else:
            centered_moments = moments

        scaled = self._divide_columns(centered_moments, scale)
        if self._clusters is None:
            omega = scaled.T @ scaled
        else:
            grouped = self._group_sum(scaled)
            omega = grouped.T @ grouped
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

            # Pre-allocate template to avoid repeated zeros_like calls
            template = np.zeros_like(array, dtype=float)
            for multi_index in np.ndindex(array.shape):
                template[multi_index] = 1.0
                yield template.copy()
                template[multi_index] = 0.0

        def flatten(obj: Any) -> np.ndarray:
            if isinstance(obj, tuple | list):
                parts = [flatten(component) for component in obj]
                if parts:
                    return np.concatenate(parts)
                return np.array([], dtype=float)
            return np.asarray(obj, dtype=float).reshape(-1)

        basis: list[Any] = []
        # Optimization: Euclidean manifolds have the ambient basis as tangent basis
        if "Euclidean" in manifold_wrapper.name:
            basis = list(ambient_basis(point.value))
            if len(basis) == target_dimension:
                return basis
            basis = []
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

        Notes
        -----
        When the underlying :class:`JacobianOperator` exposes
        ``matrix_in_basis`` (the JAX autodiff path), the matrix is computed
        with a single batched ``jax.vmap`` over the linearised JVP closure
        rather than ``len(basis)`` sequential ``matvec`` calls.  This is the
        primary cost win for problems with large ``N`` and many tangent
        directions: each ``matvec`` invocation carries non-trivial Python-
        side dispatch overhead that the batched form amortises.  The
        loop-based fallback preserves the previous behaviour when no
        batched implementation is registered (e.g., the explicit
        ``jacobian_map`` path).
        """

        operator = self.jacobian(theta)
        basis_vectors = (
            basis if basis is not None else self.tangent_basis(theta, tol=tol)
        )

        if not basis_vectors:
            return np.zeros((0, 0), dtype=float)

        # Fast path: batched JVPs in one vmap call.
        if operator.matrix_in_basis is not None:
            return operator.matrix_in_basis(basis_vectors)

        # Fallback: sequential matvec loop (explicit jacobian_map path).
        columns: list[np.ndarray] = []
        for direction in basis_vectors:
            image = operator.matvec(direction)
            columns.append(np.asarray(image, dtype=float).reshape(-1))

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
            raw = self._call_with_optional_data(self._jacobian_map, argument)
            return self._scale_jacobian_rows(raw, theta)

        point = self._maybe_point(theta)
        if self._jacobian_map is None:
            if point is None:
                raise NotImplementedError(
                    "Autodiff Jacobian requires a manifold-aware evaluation point."
                )
            return jacobian_operator(self._autodiff_moment_function(), point)

        argument = self._prepare_argument(theta)
        matrix = self._call_with_optional_data(self._jacobian_map, argument)
        scaled_matrix = self._scale_jacobian_rows(matrix, theta)
        return self._jacobian_operator_from_matrix(scaled_matrix, point)

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
        self._metadata_argument = argument
        self._metadata_moments = moments
        if not self._is_jax_backend:
            self._update_metadata(argument, moments)
        return moments

    def _metadata_ready(self) -> bool:
        labels_ready = (
            self._raw_parameter_labels is None or self._parameter_labels is not None
        )
        return (
            self._parameter_dimension is not None
            and self._parameter_shape is not None
            and self._num_moments is not None
            and labels_ready
        )

    def _ensure_metadata(self) -> None:
        if self._metadata_ready():
            return
        argument, moments = self._obtain_metadata_evaluation()
        self._update_metadata(argument, moments)
        self._metadata_moments = None

    def _obtain_metadata_evaluation(self) -> tuple[Any, Any]:
        if self._metadata_argument is not None and self._metadata_moments is not None:
            return self._metadata_argument, self._metadata_moments

        theta_probe = self._metadata_probe_theta()
        argument = self._prepare_argument(theta_probe)
        moments = self._call_with_optional_data(self._gi_map, argument)
        self._metadata_argument = argument
        self._metadata_moments = moments
        return argument, moments

    def _metadata_probe_theta(self) -> Any:
        if self.manifold is not None:
            try:
                return self.manifold.random_point()
            except AttributeError as exc:
                raise RuntimeError(
                    "Cannot infer parameter metadata lazily because the supplied "
                    "manifold does not expose random_point(); evaluate the "
                    "restriction with a representative parameter first."
                ) from exc
        raise RuntimeError(
            "Cannot infer parameter metadata lazily without a manifold providing "
            "random_point(); evaluate the restriction with a representative "
            "parameter first."
        )

    def _update_metadata(self, argument: Any, moments: Any) -> None:
        if self._parameter_dimension is None:
            parameter_array = self._array_adapter(argument)
            self._parameter_dimension = int(parameter_array.size)
            base_argument = (
                argument.value if isinstance(argument, ManifoldPoint) else argument
            )
            if isinstance(base_argument, tuple | list):
                self._parameter_shape = None
            else:
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
        if self._weights is not None:
            return self._weighted_mean(moments)

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

    def _weighted_mean(self, moments: Any) -> Any:
        r"""Compute the weighted sample mean.

        .. math::

            \bar g^*_N(\theta) = \frac{1}{n}\sum_{i=1}^n w_i\, g_i(\theta)

        Division by :math:`n` (not :math:`\sum w_i`) preserves unbiasedness
        when :math:`E[w_i] = 1`.  Uses ``nansum`` so that missing values are
        handled consistently with the unweighted path.
        """

        xp = self._xp
        array = xp.asarray(moments)
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        n = array.shape[0]
        w = xp.asarray(self._weights, dtype=array.dtype).reshape(n, 1)
        return xp.nansum(w * array, axis=0) / n

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

    def _resolve_cluster_codes(self, num_observations: int) -> tuple[np.ndarray, int]:
        """Return cached integer cluster codes ``(codes, G)`` for ``self._clusters``.

        Labels of any hashable type are accepted; they are normalised to
        contiguous integer codes ``0..G-1`` via :func:`numpy.unique` (with
        ``return_inverse=True``) on the first call and cached on the
        instance.  Raises :class:`ValueError` if the length does not match
        ``num_observations``.
        """

        if self._cluster_codes is not None and self._num_clusters is not None:
            cached = self._cluster_codes
            if cached.size != num_observations:
                raise ValueError(
                    "Cluster ids length mismatch: cached cluster codes have "
                    f"length {cached.size} but moment matrix carries "
                    f"{num_observations} observations."
                )
            return cached, int(self._num_clusters)

        raw = np.asarray(self._clusters)
        if raw.ndim != 1:
            raw = raw.reshape(-1)
        if raw.size != num_observations:
            raise ValueError(
                "Cluster ids length mismatch: expected "
                f"{num_observations} entries (one per observation), got "
                f"{raw.size}."
            )

        unique, codes = np.unique(raw, return_inverse=True)
        codes_int = np.asarray(codes, dtype=np.int64)
        self._cluster_codes = codes_int
        self._num_clusters = int(unique.size)
        return codes_int, self._num_clusters

    def _group_sum(self, scaled: Any) -> Any:
        """Aggregate row contributions of ``scaled`` by cluster.

        Returns an array of shape ``(G, ℓ)`` containing
        :math:`\\sum_{i\\in c} \\text{scaled}_{i,k}` for each cluster ``c`` and
        moment ``k``.  ``NaN`` entries are treated as zero contribution so
        that the existing per-moment ``1/sqrt(N_k)`` normalisation (which
        already ignores missing observations) carries through cleanly.
        """

        xp = self._xp
        if not self._is_jax_backend:
            array = np.asarray(scaled, dtype=float)
            if array.ndim == 1:
                array = array[:, np.newaxis]
            codes, num_clusters = self._resolve_cluster_codes(array.shape[0])
            cleaned = np.where(np.isnan(array), 0.0, array)
            grouped = np.zeros((num_clusters, cleaned.shape[1]), dtype=cleaned.dtype)
            np.add.at(grouped, codes, cleaned)
            return grouped

        array = xp.asarray(scaled)
        if array.ndim == 1:
            array = array[:, xp.newaxis]
        codes, num_clusters = self._resolve_cluster_codes(int(array.shape[0]))
        cleaned = xp.where(xp.isnan(array), xp.zeros_like(array), array)

        try:
            from jax.ops import segment_sum
        except ImportError as exc:  # pragma: no cover - jax always present on jax path
            raise RuntimeError(
                "JAX backend requested for cluster-aware omega_hat but "
                "jax.ops.segment_sum is unavailable."
            ) from exc

        codes_backend = xp.asarray(codes)
        return segment_sum(cleaned, codes_backend, num_segments=num_clusters)

    def _scale_jacobian_rows(self, matrix: Any, theta: Any = None) -> Any:
        r"""Multiply each row *k* of a Jacobian by :math:`\sqrt{N_k}`.

        User-supplied Jacobians are derivatives of the sample *mean*; this
        rescaling makes them consistent with the :math:`1/\sqrt{N_k}` convention
        used by :meth:`g_bar`.
        """

        if self._observation_counts is None:
            if theta is not None:
                # Evaluate moments to populate observation counts;
                # _evaluate_backend caches the result so _ensure_metadata
                # can finalise without a redundant evaluation.
                self._evaluate_backend(theta)
            self._ensure_metadata()
        counts = self._observation_counts
        if counts is None:
            return matrix

        matrix_array = np.asarray(matrix, dtype=float)
        sqrt_n = np.sqrt(counts).reshape(-1)
        if matrix_array.ndim == 1:
            return matrix_array * sqrt_n
        # matrix shape is (ℓ, p) — scale each row k by √N_k
        return matrix_array * sqrt_n[:, np.newaxis]

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

        if isinstance(base, tuple | list):
            flattened: list[np.ndarray] = []
            for component in base:
                component_array = np.asarray(component, dtype=float).reshape(-1)
                flattened.append(component_array)
            if not flattened:
                concatenated = np.array([], dtype=float)
            else:
                concatenated = np.concatenate(flattened)
            return self._xp.asarray(concatenated)

        if self._is_jax_backend:
            return self._xp.asarray(base)
        return np.asarray(base, dtype=float)

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------
    def __getstate__(self) -> dict[str, Any]:
        """Exclude unpicklable module references (_xp, _linalg)."""
        state = self.__dict__.copy()
        state.pop("_xp", None)
        state.pop("_linalg", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore module references from the serialised backend kind."""
        self.__dict__.update(state)
        if self._backend_kind == "jax":
            self._xp = jnp
            self._linalg = jnp.linalg
        else:
            self._xp = np
            self._linalg = np.linalg

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

    def format_parameter(self, theta: Any) -> Any:
        """
        Return ``theta`` enriched with parameter labels when available.

        When :class:`MomentRestriction` was constructed with ``parameter_labels``,
        this helper converts flat array outputs (e.g., from optimizers) into a
        :class:`DataVec` or :class:`DataMat` instance carrying the supplied labels.
        Structured parameters (lists/tuples) are returned unchanged.
        """

        self._ensure_metadata()
        if self._raw_parameter_labels is None or isinstance(theta, tuple | list):
            return theta

        try:
            array = np.asarray(theta, dtype=float)
        except Exception:
            return theta

        labels_obj = self._raw_parameter_labels
        expected_dim = self._parameter_dimension
        flat = array.reshape(-1)
        if expected_dim is not None and flat.size != expected_dim:
            return theta

        if isinstance(labels_obj, DataVec):
            return DataVec(
                flat,
                index=labels_obj.index,
                dtype=array.dtype,
                name=labels_obj.name,
            )
        if isinstance(labels_obj, DataMat):
            values = array.reshape(labels_obj.shape)
            return DataMat(
                values,
                index=labels_obj.index,
                columns=labels_obj.columns,
                dtype=array.dtype,
            )
        if isinstance(labels_obj, list | tuple) and all(
            isinstance(label, str) for label in labels_obj
        ):
            return DataVec(
                flat, index=list(labels_obj), dtype=array.dtype, name="theta"
            )

        return theta

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
            mean = self._mean(moments)
            counts = self._count(moments)
            sqrt_n = counts**0.5
            return mean * sqrt_n

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
        if self._backend == "jax":
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
        if self._backend == "jax":
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
