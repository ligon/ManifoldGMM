"""Moment wild bootstrap for GMM confidence regions on manifolds.

This module implements a wild-bootstrap procedure that resamples *moment errors*
rather than observation pairs, re-estimates the parameter on the manifold for
each bootstrap replicate, and uses geodesic Mahalanobis distances to construct
confidence regions that live on the manifold itself.

The approach follows Davidson & Flachaire (2008) in preferring the Rademacher
distribution for bootstrap weights, and Bhattacharya & Patrangenaru (2005) for
the geodesic distance formulation on Riemannian manifolds.

Key classes
-----------
MomentWildBootstrap
    Head-node orchestrator: generates tasks, collects results, computes
    critical values and membership tests.
BootstrapTask
    Self-contained, serializable task suitable for dispatch to a cluster worker.
BootstrapResult
    Lightweight payload returned by each worker.

Weight generators
-----------------
rademacher_weights
    Shifted Rademacher: w_i in {0, 2} with equal probability.
mammen_weights
    Mammen two-point distribution matching skewness.
exponential_weights
    Exp(1) weights (Bayesian bootstrap).
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # Optional dependency for richer pickling support
    import cloudpickle
except ImportError:  # pragma: no cover - optional
    cloudpickle = None

from ..geometry import ManifoldPoint
from .gmm import GMM, FixedWeighting, GMMResult, PenaltyStrategy
from .moment_restriction import MomentRestriction

# -----------------------------------------------------------------------
# Weight generators
# -----------------------------------------------------------------------


def rademacher_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Shifted Rademacher bootstrap weights.

    Each weight is drawn independently as

    .. math::

        w_i = 1 + \epsilon_i, \quad \epsilon_i \in \{-1, +1\}
        \text{ with equal probability},

    so that :math:`w_i \in \{0, 2\}`.

    Statistical properties: :math:`E[w] = 1`, :math:`\operatorname{Var}[w] = 1`,
    :math:`E[(w - 1)^3] = 0`.

    Preferred over Mammen on the basis of Davidson & Flachaire (2008): the
    attempt to match the third moment of the residual distribution adds noise
    to second-moment estimation without compensating gains.

    Parameters
    ----------
    n : int
        Number of observations (weights to generate).
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` with values in ``{0, 2}``.
    """

    epsilon = 2 * rng.integers(0, 2, size=n) - 1  # {-1, +1}
    return (1 + epsilon).astype(float)


def mammen_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Mammen two-point bootstrap weights.

    The distribution takes values

    .. math::

        w = \begin{cases}
        1 - \frac{\sqrt{5} - 1}{2} & \text{with probability } p = \frac{\sqrt{5} + 1}{2\sqrt{5}}, \\
        1 + \frac{\sqrt{5} + 1}{2} & \text{with probability } 1 - p.
        \end{cases}

    Statistical properties: :math:`E[w] = 1`, :math:`\operatorname{Var}[w] = 1`,
    :math:`E[(w - 1)^3] = 1`.

    The non-zero skewness matches the third moment of the (unknown) residual
    distribution, yielding an asymptotic refinement under certain conditions.
    However, Davidson & Flachaire (2008) caution that this adds noise in
    second-moment estimation for moderate samples.

    Parameters
    ----------
    n : int
        Number of observations.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.
    """

    sqrt5 = np.sqrt(5.0)
    p = (sqrt5 + 1.0) / (2.0 * sqrt5)
    val_low = 1.0 - (sqrt5 - 1.0) / 2.0
    val_high = 1.0 + (sqrt5 + 1.0) / 2.0
    draws = rng.random(n)
    return np.where(draws < p, val_low, val_high)


def exponential_weights(n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Exponential(1) bootstrap weights (Bayesian bootstrap).

    Each weight is drawn independently from the standard exponential
    distribution.

    Statistical properties: :math:`E[w] = 1`, :math:`\operatorname{Var}[w] = 1`.

    Parameters
    ----------
    n : int
        Number of observations.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` with positive real values.
    """

    return rng.exponential(scale=1.0, size=n)


def rademacher_signs(n: int, rng: np.random.Generator) -> np.ndarray:
    r"""Pure :math:`\pm 1` Rademacher signs.

    Distinct from :func:`rademacher_weights` (which returns shifted
    weights in :math:`\{0, 2\}` for the fit-replication bootstrap);
    these are the mean-zero sign-flip weights used by the wild
    bootstrap of *centred* moment contributions.  Used by
    :func:`k_statistic_bootstrap_for_result` when testing
    ``H0: theta = theta_0`` -- the centred g_i have approximately mean
    zero under H0, so multiplying by mean-zero signs yields a
    bootstrap whose g_bar* is approximately mean zero too.

    Statistical properties: :math:`E[s] = 0`, :math:`\operatorname{Var}[s] = 1`,
    :math:`s^2 = 1` always.

    Parameters
    ----------
    n : int
        Number of signs to draw.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)`` with values in ``{-1.0, +1.0}``.
    """

    return (2 * rng.integers(0, 2, size=n) - 1).astype(float)


_WEIGHT_GENERATORS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "rademacher": rademacher_weights,
    "mammen": mammen_weights,
    "exponential": exponential_weights,
}


# -----------------------------------------------------------------------
# BootstrapResult
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Lightweight return payload from a single bootstrap replicate.

    Attributes
    ----------
    task_id : int
        Integer identifying the replicate.
    seed : int
        RNG seed used to generate weights.
    theta_star : ManifoldPoint
        Re-estimated parameter on the manifold.
    criterion_value : float
        GMM criterion :math:`J_N(\\hat\\theta^*)` at the bootstrap estimate.
    converged : bool
        Whether the optimizer reported convergence.
    """

    task_id: int
    seed: int
    theta_star: ManifoldPoint
    criterion_value: float
    converged: bool


# -----------------------------------------------------------------------
# BootstrapTask
# -----------------------------------------------------------------------


@dataclass
class BootstrapTask:
    """Self-contained, serializable task for a single bootstrap replicate.

    Each task carries everything needed to re-estimate the parameter on a
    cluster worker without additional context.

    **Data-size note:** each serialized task carries one copy of the dataset
    (:math:`n \\times p \\times 8` bytes).  This is acceptable for
    :math:`n \\approx 10^4`--:math:`10^5`.  For larger datasets, broadcast the
    data once per node (e.g., ``ray.put`` or ``dask.scatter``) and use
    lightweight task stubs.  A ``scatter_tasks()`` API can be added when needed.

    Attributes
    ----------
    restriction : MomentRestriction
        The base moment restriction (unweighted).
    weighting_matrix : Any
        Fixed weighting matrix :math:`W` evaluated at :math:`\\hat\\theta`.
    initial_point : Any
        Starting point for the optimizer (typically :math:`\\hat\\theta`).
    seed : int
        RNG seed for weight generation.
    weight_scheme : str or callable
        Either the name of a built-in distribution (``"rademacher"``,
        ``"mammen"``, ``"exponential"``) or a callable
        ``(n, rng) -> np.ndarray`` of shape ``(n,)`` with ``E[w] = 1``.
        Callables ride through the pickled payload, so user-defined schemes
        do not depend on parent-process mutations of the module-level
        registry surviving the fork to a loky worker.
    optimizer_class : type or None
        Optimizer class to use (default: pymanopt TrustRegions).
    optimizer_kwargs : dict
        Extra keyword arguments forwarded to the optimizer constructor.
    task_id : int
        Integer identifying this replicate.
    cluster_codes : numpy.ndarray or None
        Integer cluster codes of length ``N`` (one per observation), or
        ``None`` for the i.i.d. scheme.  When provided, one wild weight is
        drawn per unique cluster and broadcast to its member rows.
    num_clusters : int or None
        Number of unique clusters ``G`` (matches
        ``cluster_codes.max() + 1``).  Required when ``cluster_codes`` is set.
    """

    restriction: MomentRestriction
    weighting_matrix: Any
    initial_point: Any
    seed: int
    weight_scheme: str | Callable[[int, np.random.Generator], np.ndarray]
    optimizer_class: type | None
    optimizer_kwargs: dict[str, Any]
    task_id: int
    cluster_codes: np.ndarray | None = None
    num_clusters: int | None = None
    # Parameter-space penalty propagated from the original ``GMMResult``
    # (#29).  When non-``None``, the replicate ``GMM`` is constructed
    # with this penalty so the bootstrap targets the *penalised*
    # estimator's sampling distribution; dropping it would cause
    # replicates to drift to the unpenalised optimum, which on a
    # weakly-identified design (cf. K-Aggregators exp-link runaway)
    # lives in a different basin than the point estimate.
    # ``MomentWildBootstrap.tasks`` reads ``gmm_result.penalty``
    # automatically, so callers only see this field if they build
    # tasks by hand.
    penalty: PenaltyStrategy | Callable[[Any], Any] | None = None

    def run(self) -> BootstrapResult:
        """Execute the bootstrap replicate (worker entry point).

        Steps
        -----
        1. Resolve the weight generator from ``weight_scheme`` (callable
           used directly, string looked up in the built-in registry).
        2. Draw ``N`` i.i.d. weights, or ``G`` cluster-constant weights
           broadcast via ``cluster_codes`` when set.
        3. Create a weighted clone via ``restriction.with_weights(w)``,
           chaining ``with_clusters`` when the parent restriction carries
           a cluster assignment.
        4. Construct a ``GMM`` estimator with ``FixedWeighting(W)`` and run it.
        5. Return a lightweight ``BootstrapResult``.
        """

        rng = np.random.default_rng(self.seed)

        scheme = self.weight_scheme
        generator: Callable[[int, np.random.Generator], np.ndarray]
        if callable(scheme):
            generator = scheme
        else:
            try:
                generator = _WEIGHT_GENERATORS[scheme]
            except KeyError:
                raise ValueError(
                    f"Unknown weight scheme {scheme!r}; "
                    f"choose from {sorted(_WEIGHT_GENERATORS)} "
                    f"or pass a callable (n, rng) -> np.ndarray."
                ) from None

        n = self.restriction.num_observations
        if n is None:
            raise RuntimeError(
                "Cannot determine observation count from restriction; "
                "evaluate the restriction at a point first."
            )

        if self.cluster_codes is None:
            weights = generator(n, rng)
        else:
            codes = self.cluster_codes
            if codes.shape != (n,):
                raise ValueError(
                    f"cluster_codes shape {codes.shape} does not match "
                    f"num_observations ({n},)"
                )
            if self.num_clusters is None:
                raise RuntimeError(
                    "cluster_codes was supplied without num_clusters; "
                    "construct BootstrapTask via MomentWildBootstrap."
                )
            cluster_weights = generator(self.num_clusters, rng)
            weights = np.asarray(cluster_weights)[codes]

        weighted_restriction = self.restriction.with_weights(weights)
        if self.restriction.clusters is not None:
            weighted_restriction = weighted_restriction.with_clusters(
                self.restriction.clusters
            )

        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs.setdefault("verbosity", 0)

        gmm = GMM(
            weighted_restriction,
            weighting=FixedWeighting(self.weighting_matrix),
            optimizer=self.optimizer_class,
            initial_point=self.initial_point,
            penalty=self.penalty,
        )

        result = gmm.estimate(optimizer_kwargs=optimizer_kwargs)

        converged = bool(result.optimizer_report.get("converged", False))

        return BootstrapResult(
            task_id=self.task_id,
            seed=self.seed,
            theta_star=result.theta_point,
            criterion_value=result.criterion_value,
            converged=converged,
        )

    def to_bytes(self) -> bytes:
        """Serialize this task to bytes using pickle (cloudpickle fallback).

        Returns
        -------
        bytes
            Serialized representation.
        """

        try:
            return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError):
            if cloudpickle is None:
                raise
            return cloudpickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> BootstrapTask:
        """Deserialize a task from bytes.

        Parameters
        ----------
        data : bytes
            Serialized task (from :meth:`to_bytes`).

        Returns
        -------
        BootstrapTask
        """

        try:
            obj = pickle.loads(data)
        except Exception:
            if cloudpickle is None:
                raise
            obj = cloudpickle.loads(data)
        if not isinstance(obj, BootstrapTask):
            raise TypeError("Deserialized object is not a BootstrapTask")
        return obj


# Module-level helper for joblib workers: loky pickles by name, so
# ``delayed(BootstrapTask.run)`` must dispatch through a top-level
# callable rather than a bound method.
def _run_bootstrap_task(task: BootstrapTask) -> BootstrapResult:
    return task.run()


# -----------------------------------------------------------------------
# Tangent coordinate helpers
# -----------------------------------------------------------------------


def _tangent_coordinates(
    manifold: Any,
    base_value: Any,
    tangent_vector: Any,
    basis: list[Any],
    restriction: MomentRestriction,
) -> np.ndarray:
    r"""Project a tangent vector onto a basis using the Riemannian inner product.

    For each basis element :math:`e_j` the coordinate is

    .. math::

        \xi_j = \langle e_j,\; v \rangle_{\hat\theta}

    where :math:`v` is ``tangent_vector`` and the inner product is obtained
    from ``manifold.data.inner_product``.

    Parameters
    ----------
    manifold
        Manifold wrapper (must have ``manifold.data.inner_product``).
    base_value
        Ambient representation of the base point.
    tangent_vector
        Tangent vector at the base point.
    basis
        List of tangent basis vectors at the base point.
    restriction
        MomentRestriction used as a fallback for the array adapter.

    Returns
    -------
    numpy.ndarray
        Coordinate vector of shape ``(len(basis),)``.
    """

    inner = getattr(manifold.data, "inner_product", None)
    if inner is None:
        # Fallback: flatten and use Euclidean inner product
        def _flatten(v: Any) -> np.ndarray:
            return np.asarray(restriction._array_adapter(v), dtype=float).reshape(-1)

        tv_flat = _flatten(tangent_vector)
        coords = np.array([tv_flat @ _flatten(e) for e in basis])
        return coords

    coords = np.array([float(inner(base_value, e_j, tangent_vector)) for e_j in basis])
    return coords


# -----------------------------------------------------------------------
# Geodesic Mahalanobis distance
# -----------------------------------------------------------------------


def geodesic_mahalanobis_distance(
    result: GMMResult,
    point: ManifoldPoint | Any,
    *,
    covariance: np.ndarray | None = None,
) -> float:
    r"""Squared geodesic Mahalanobis distance from an estimate to a point.

    Computes

    .. math::

        d^2 = \xi^\top \Sigma^{-1} \xi

    where :math:`\xi` are the tangent coordinates of
    :math:`\operatorname{Log}_{\hat\theta}(\theta_0)` and :math:`\Sigma` is
    the tangent-space covariance of the GMM estimator.

    Under standard regularity conditions the statistic :math:`d^2` is
    asymptotically :math:`\chi^2_p` where :math:`p` is the manifold
    dimension, providing the basis for both asymptotic and bootstrap
    confidence regions.

    Parameters
    ----------
    result : GMMResult
        Completed estimation result providing :math:`\hat\theta`, the
        tangent basis, and the covariance.
    point : ManifoldPoint or array-like
        The candidate parameter value.
    covariance : numpy.ndarray or None
        Tangent-space covariance matrix.  If ``None``, uses
        ``result.tangent_covariance()``.

    Returns
    -------
    float
        Squared geodesic Mahalanobis distance :math:`d^2 \ge 0`.

    References
    ----------
    Bhattacharya, R. & Patrangenaru, V. (2005). Large sample theory of
    intrinsic and extrinsic sample means on manifolds. *Annals of
    Statistics*, 33(1), 1--29.
    """

    from ..utils.numeric import ridge_inverse

    theta_hat = result.theta_point
    restriction = result.restriction
    manifold = restriction.manifold
    if manifold is None:
        raise ValueError("Restriction must define a manifold for geodesic distances")

    if not isinstance(point, ManifoldPoint):
        point = ManifoldPoint(manifold, point)

    # Tangent basis and covariance
    basis = restriction.tangent_basis(theta_hat)

    if covariance is None:
        cov_mat = result.tangent_covariance().to_numpy(dtype=float)
    else:
        cov_mat = np.asarray(covariance, dtype=float)

    cov_inv, _ = ridge_inverse(cov_mat)

    # Log map (or fallback to ambient difference + projection)
    log_fn = getattr(manifold.data, "log", None)
    if log_fn is not None:
        log_vec = log_fn(theta_hat.value, point.value)
    else:
        diff = _ambient_difference(theta_hat.value, point.value)
        log_vec = theta_hat.project_tangent(diff)

    xi = _tangent_coordinates(manifold, theta_hat.value, log_vec, basis, restriction)
    return float(xi @ cov_inv @ xi)


def _ambient_difference(base: Any, target: Any) -> Any:
    """Compute the ambient-space difference ``target - base``."""

    if isinstance(base, tuple | list):
        return type(base)(
            np.asarray(t) - np.asarray(b) for b, t in zip(base, target, strict=False)
        )
    return np.asarray(target) - np.asarray(base)


# -----------------------------------------------------------------------
# MomentWildBootstrap
# -----------------------------------------------------------------------


class MomentWildBootstrap:
    r"""Head-node orchestrator for moment wild bootstrap confidence regions.

    Given a ``GMMResult``, this class generates bootstrap replicates by
    resampling moment *errors* (not observation pairs), re-estimating on the
    manifold :math:`\mathcal{M}`, and computing geodesic Mahalanobis distances
    from the original estimate.

    The bootstrap critical value for a :math:`(1 - \alpha)` confidence region
    is the :math:`(1 - \alpha)` quantile of the squared geodesic distances

    .. math::

        d^2_b = \operatorname{Log}_{\hat\theta}(\hat\theta^*_b)^\top
                \Sigma^{-1}
                \operatorname{Log}_{\hat\theta}(\hat\theta^*_b).

    References
    ----------
    - Davidson, R. & Flachaire, E. (2008). The wild bootstrap, tamed at last.
      *Journal of Econometrics*, 146(1), 162--169.
    - Bhattacharya, R. & Patrangenaru, V. (2005). Large sample theory of
      intrinsic and extrinsic sample means on manifolds. *Annals of Statistics*,
      33(1), 1--29.

    Parameters
    ----------
    gmm_result : GMMResult
        Completed estimation result providing :math:`\hat\theta`, the
        weighting matrix, and the moment restriction.
    n_bootstrap : int
        Number of bootstrap replicates.
    weight_scheme : str or callable, default ``"rademacher"``
        Either the name of a built-in scheme (``"rademacher"``, ``"mammen"``,
        ``"exponential"``) or a callable ``(n, rng) -> np.ndarray`` of shape
        ``(n,)`` with ``E[w] = 1``.  Callables are pickled into the task
        payload, avoiding the loky-worker hazard of relying on parent-process
        mutations of the module-level registry surviving a fork.
    clusters : array-like or None, default ``None``
        Optional cluster ids of length ``N``.  When supplied (or inherited
        from ``gmm_result.restriction.clusters`` if that is non-``None``),
        each replicate draws one wild weight per unique cluster and
        broadcasts it to the cluster's member rows.  Replicate restrictions
        also inherit the cluster assignment so their :math:`\hat\Omega` is
        cluster-robust.  With clusters of size 1 this reduces to the
        per-observation scheme byte-for-byte.
    base_seed : int, default 0
        Base random seed; replicate *b* uses seed ``base_seed + b``.
    optimizer_class : type or None
        Optimizer class forwarded to each task.
    optimizer_kwargs : dict or None
        Extra optimizer keyword arguments.

    Notes
    -----
    Under the cluster bootstrap, the effective sample size is the number
    of clusters ``G``, not ``N``.  ``n_bootstrap`` should be set with this
    in mind: with small ``G`` (say, < 30) the bootstrap distribution is
    coarse and quantile estimates noisy regardless of how many replicates
    are taken.  When ``weight_scheme=`` is a callable, it must be picklable
    by the chosen executor (cloudpickle is used as a fallback by
    :meth:`BootstrapTask.to_bytes`; loky workers carry their own pickling
    semantics, so closures over unpicklable state should be avoided).

    Estimand under penalty (#19 MR1, #29)
    -------------------------------------
    When ``gmm_result`` carries a non-``None`` ``penalty`` (any of the
    :class:`~manifoldgmm.econometrics.gmm.PenaltyStrategy`-shaped
    inputs to :class:`~manifoldgmm.econometrics.gmm.GMM`), each
    replicate's re-fit applies the same penalty.  The resulting
    bootstrap distribution therefore characterises uncertainty around
    :math:`\hat\theta_{\text{pen}}` -- the **penalised** estimator,
    which is itself an asymptotically biased estimator of
    :math:`\theta_0`.  This is the correct construction for inference
    *about* the reported point estimate, **not** a frequentist
    sandwich CI for :math:`\theta_0`.  Bias-aware methods (out of
    scope for this class) would be needed there; see #19's scope
    section for the open question.

    Dropping the penalty on replicates -- the pre-#29 behaviour -- was
    a real bug: on a weakly-identified design (the K-Aggregators
    exp-link runaway is the motivating case) the unpenalised optimum
    lives in a different basin from
    :math:`\hat\theta_{\text{pen}}`, so unpenalised replicates do not
    characterise uncertainty around the point estimate at all.
    """

    def __init__(
        self,
        gmm_result: GMMResult,
        n_bootstrap: int = 199,
        *,
        weight_scheme: (
            str | Callable[[int, np.random.Generator], np.ndarray]
        ) = "rademacher",
        clusters: Any | None = None,
        base_seed: int = 0,
        optimizer_class: type | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(weight_scheme, str):
            if weight_scheme not in _WEIGHT_GENERATORS:
                raise ValueError(
                    f"Unknown weight_scheme {weight_scheme!r}; "
                    f"choose from {sorted(_WEIGHT_GENERATORS)} "
                    f"or pass a callable (n, rng) -> np.ndarray."
                )
        elif not callable(weight_scheme):
            raise TypeError(
                "weight_scheme must be a str name or a callable "
                "(n, rng) -> np.ndarray; got "
                f"{type(weight_scheme).__name__}."
            )

        self._gmm_result = gmm_result
        self._n_bootstrap = n_bootstrap
        self._weight_scheme = weight_scheme
        self._base_seed = base_seed
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = dict(optimizer_kwargs or {})

        # Resolve cluster assignment: explicit `clusters=` wins; otherwise
        # inherit from the restriction so the natural usage (clustered
        # GMMResult -> clustered bootstrap) just works.  We carry the
        # resolved ids in one place (`self._cluster_ids`) and attach them
        # to the task-side restriction in `tasks()`.
        restriction = gmm_result.restriction
        cluster_source = clusters if clusters is not None else restriction.clusters

        if cluster_source is None:
            self._cluster_ids: Any | None = None
            self._cluster_codes: np.ndarray | None = None
            self._num_clusters: int | None = None
        else:
            cluster_arr = np.asarray(cluster_source)
            if cluster_arr.ndim != 1:
                raise ValueError(
                    "clusters must be a 1-D array of length N; "
                    f"got shape {cluster_arr.shape}."
                )
            n_obs = restriction.num_observations
            if n_obs is not None and cluster_arr.shape[0] != n_obs:
                raise ValueError(
                    f"clusters length ({cluster_arr.shape[0]}) does not "
                    f"match restriction.num_observations ({n_obs})."
                )
            _, codes = np.unique(cluster_arr, return_inverse=True)
            self._cluster_ids = cluster_source
            self._cluster_codes = codes.astype(np.int64, copy=False)
            self._num_clusters = (
                int(self._cluster_codes.max()) + 1
                if self._cluster_codes.size > 0
                else 0
            )

        # Extract a fixed weighting matrix W evaluated at theta_hat
        weighting: Any = gmm_result.weighting
        if weighting is None:
            raise ValueError(
                "MomentWildBootstrap requires a GMMResult carrying a weighting "
                "strategy; got None."
            )
        theta_hat = gmm_result.theta_point
        if hasattr(weighting, "matrix") and callable(weighting.matrix):
            self._W = np.asarray(weighting.matrix(theta_hat), dtype=float)
        elif callable(weighting):
            self._W = np.asarray(weighting(theta_hat), dtype=float)
        else:
            self._W = np.asarray(weighting, dtype=float)

        self._results: list[BootstrapResult] = []
        self._distances: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------

    def tasks(self) -> list[BootstrapTask]:
        """Generate one :class:`BootstrapTask` per bootstrap replicate.

        Returns
        -------
        list of BootstrapTask
            Tasks indexed from 0 to ``n_bootstrap - 1``.
        """

        restriction = self._gmm_result.restriction
        # Attach the resolved cluster ids to the task-side restriction if
        # the bootstrap is operating in clustered mode and the original
        # restriction did not already carry the same assignment.  This
        # ensures `BootstrapTask.run`'s `with_clusters` chaining produces a
        # cluster-robust replicate Omega for every replicate.
        if self._cluster_ids is not None and restriction.clusters is None:
            restriction = restriction.with_clusters(self._cluster_ids)

        theta_hat = self._gmm_result.theta_point
        # Use the ambient value as initial point for each replicate
        initial_point = theta_hat.value
        # Propagate the original fit's parameter penalty (#29).  When
        # the result was produced by ``GMM(..., penalty=...)`` each
        # replicate must re-fit *with* that penalty -- otherwise the
        # bootstrap distribution targets the unpenalised optimum
        # rather than ``theta_hat_pen``.  ``None`` (no penalty) is
        # bit-identical to the pre-#29 behaviour.
        penalty = self._gmm_result.penalty

        return [
            BootstrapTask(
                restriction=restriction,
                weighting_matrix=self._W,
                initial_point=initial_point,
                seed=self._base_seed + b,
                weight_scheme=self._weight_scheme,
                optimizer_class=self._optimizer_class,
                optimizer_kwargs=self._optimizer_kwargs,
                task_id=b,
                cluster_codes=self._cluster_codes,
                num_clusters=self._num_clusters,
                penalty=penalty,
            )
            for b in range(self._n_bootstrap)
        ]

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def collect(self, results: list[BootstrapResult]) -> None:
        """Ingest worker results.

        Parameters
        ----------
        results : list of BootstrapResult
            Bootstrap results from completed tasks.
        """

        self._results.extend(results)
        self._distances = None  # invalidate cache

    # ------------------------------------------------------------------
    # Sequential execution (debugging)
    # ------------------------------------------------------------------

    def run_sequential(self) -> list[BootstrapResult]:
        """Execute all tasks sequentially in the current process.

        Intended for debugging and small-scale testing. For production use,
        prefer :meth:`run_parallel`, or dispatch the tasks from
        :meth:`tasks` to a cluster scheduler.

        Returns
        -------
        list of BootstrapResult
        """

        task_list = self.tasks()
        results = [task.run() for task in task_list]
        self.collect(results)
        return results

    def run_parallel(
        self,
        *,
        n_jobs: int = -1,
        backend: str = "loky",
        verbose: int = 0,
        **joblib_kwargs: Any,
    ) -> list[BootstrapResult]:
        """Execute all tasks in parallel via joblib.

        Each :class:`BootstrapTask` is independent and self-contained
        (it carries its own slice of data, the weighting matrix at
        ``theta_hat``, and the seed used to draw weights), so the
        replicates fan out across worker processes without shared
        state.  JAX cost-function caches are populated once per worker
        during the first replicate; replicate work amortises across
        the remaining tasks on that worker.

        Parameters
        ----------
        n_jobs : int, default -1
            Number of worker processes.  ``-1`` uses all cores; any
            positive integer caps the pool.  Passed through to
            ``joblib.Parallel``.
        backend : str, default "loky"
            joblib backend.  ``"loky"`` (default) spawns independent
            processes via cloudpickle, which side-steps the GIL and
            handles JAX-traced closures cleanly.  ``"threading"`` is
            convenient for small / IO-bound replicates but will not
            scale because JAX holds the GIL during compilation and
            most cost evaluations.  ``"multiprocessing"`` is similar
            to loky without the cloudpickle fallback.
        verbose : int, default 0
            joblib verbosity level.
        **joblib_kwargs
            Forwarded to ``joblib.Parallel`` (e.g. ``batch_size``,
            ``pre_dispatch``).

        Returns
        -------
        list of BootstrapResult
            Replicate results in task order.  Same shape as
            :meth:`run_sequential`; for ``backend="loky"`` with a
            fixed ``base_seed`` the per-replicate RNG seeds are
            deterministic, so the collected ``theta_star`` values
            match the sequential run replicate-for-replicate.
        """

        from joblib import Parallel, delayed

        task_list = self.tasks()
        results = list(
            Parallel(
                n_jobs=n_jobs,
                backend=backend,
                verbose=verbose,
                **joblib_kwargs,
            )(delayed(_run_bootstrap_task)(task) for task in task_list)
        )
        self.collect(results)
        return results

    # ------------------------------------------------------------------
    # Geodesic distances
    # ------------------------------------------------------------------

    def geodesic_distances(self, *, covariance: np.ndarray | None = None) -> np.ndarray:
        r"""Compute geodesic Mahalanobis distances for collected replicates.

        For each replicate *b*, the distance is

        .. math::

            d^2_b = \xi_b^\top \Sigma^{-1} \xi_b

        where :math:`\xi_b` are the tangent coordinates of
        :math:`\operatorname{Log}_{\hat\theta}(\hat\theta^*_b)` and
        :math:`\Sigma` is the tangent covariance.

        Parameters
        ----------
        covariance : numpy.ndarray or None
            Tangent-space covariance matrix.  If ``None``, uses
            ``GMMResult.tangent_covariance()``.

        Returns
        -------
        numpy.ndarray
            Squared geodesic distances, shape ``(n_collected,)``.
        """

        if self._distances is not None and covariance is None:
            return self._distances.copy()

        if not self._results:
            return np.array([], dtype=float)

        result = self._gmm_result

        distances = np.empty(len(self._results), dtype=float)
        for i, br in enumerate(self._results):
            distances[i] = geodesic_mahalanobis_distance(
                result,
                br.theta_star,
                covariance=covariance,
            )

        if covariance is None:
            self._distances = distances.copy()

        return distances

    # ------------------------------------------------------------------
    # Critical values and membership
    # ------------------------------------------------------------------

    def critical_value(self, alpha: float = 0.05) -> float:
        r"""Return the :math:`(1-\alpha)` quantile of bootstrap distances.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        float
            Critical value :math:`c_{1-\alpha}`.
        """

        d2 = self.geodesic_distances()
        if d2.size == 0:
            raise ValueError("No bootstrap results collected; run tasks first")
        return float(np.quantile(d2, 1.0 - alpha))

    def in_confidence_region(
        self, point: ManifoldPoint | Any, alpha: float = 0.05
    ) -> bool:
        r"""Test whether ``point`` belongs to the confidence region.

        The point is in the :math:`(1-\alpha)` confidence region if its
        geodesic Mahalanobis distance from :math:`\hat\theta` is at most
        the bootstrap critical value.

        Parameters
        ----------
        point : ManifoldPoint or array-like
            Candidate parameter.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
        """

        cv = self.critical_value(alpha)
        d2 = geodesic_mahalanobis_distance(self._gmm_result, point)
        return d2 <= cv

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a summary of collected bootstrap results.

        Returns
        -------
        dict
            Keys: ``n_collected``, ``n_converged``, ``distances_quantiles``
            (10%, 50%, 90%, 95%, 99%).
        """

        n_collected = len(self._results)
        n_converged = sum(1 for r in self._results if r.converged)

        info: dict[str, Any] = {
            "n_collected": n_collected,
            "n_converged": n_converged,
        }

        if n_collected > 0:
            d2 = self.geodesic_distances()
            quantiles = [0.10, 0.50, 0.90, 0.95, 0.99]
            info["distances_quantiles"] = {
                f"q{int(q * 100)}": float(np.quantile(d2, q)) for q in quantiles
            }
        else:
            info["distances_quantiles"] = {}

        return info


# -----------------------------------------------------------------------
# Bootstrap K-statistic (#25)
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class KStatBootstrapResult:
    """Output of :func:`k_statistic_bootstrap_for_result`.

    Attributes
    ----------
    K_observed, S_observed, J_observed:
        Point estimates of K, S, J at ``theta_0`` computed on the data.
        Match what :meth:`GMMResult.k_statistic` would return for the
        same ``theta_0`` (the bootstrap doesn't change these; it only
        produces a reference distribution against which to compare them).
    K_bootstrap, S_bootstrap, J_bootstrap:
        Arrays of bootstrap replicates of K*, S*, J* under H0:
        ``theta = theta_0``.  Each has length ``n_replicates``.
    df_K, df_S:
        Degrees of freedom for the asymptotic chi^2 reference
        distributions of K and S (``p`` and ``ell - p`` respectively,
        matching :class:`KStatisticResult`).
    p_K_bootstrap, p_S_bootstrap:
        Percentile p-values from the bootstrap distributions:
        ``(1 + sum(K_bootstrap >= K_observed)) / (1 + n_replicates)``
        (the +1's give the conventional small-sample-corrected upper
        bound on the p-value).
    p_K_asymptotic, p_S_asymptotic:
        Reference chi^2-based p-values from the same observed
        statistics; reported alongside the bootstrap p-values so a
        caller can inspect the gap between the two reference
        distributions.
    n_replicates:
        Number of bootstrap replicates run.
    method:
        ``"iid"`` when no cluster structure was found, ``"cluster_wild"``
        otherwise.
    cluster_info:
        ``{"num_clusters": G, "source": "argument" | "restriction"}``
        when clustered; ``None`` when ``method == "iid"``.
    """

    K_observed: float
    S_observed: float
    J_observed: float
    K_bootstrap: np.ndarray
    S_bootstrap: np.ndarray
    J_bootstrap: np.ndarray
    df_K: int
    df_S: int
    p_K_bootstrap: float
    p_S_bootstrap: float
    p_K_asymptotic: float
    p_S_asymptotic: float
    n_replicates: int
    method: str
    cluster_info: Mapping[str, Any] | None


def k_statistic_bootstrap_for_result(
    result: GMMResult,
    *,
    theta_0: Any,
    n_replicates: int = 200,
    cluster_index: Any | None = None,
    ridge_condition: float = 1e8,
    rng: np.random.Generator | int | None = None,
) -> KStatBootstrapResult:
    r"""Cluster-wild bootstrap of the Kleibergen K-statistic at ``theta_0``.

    See #25 for motivation and design.  Sketch:

    1. Compute the per-observation moment matrix :math:`g_i(\theta_0)`
       and centre it.
    2. Compute the data-side ingredients (``Omega_hat``, ``D``,
       ``Omega_hat^{-1}``, ``(D' Omega^{-1} D)^{-1}``) once; cache the
       projection ``P = Omega^{-1} D (D' Omega^{-1} D)^{-1} D' Omega^{-1}``
       so each replicate's :math:`K^* = g_bar^{*\top} P g_bar^*` is a
       single quadratic form.
    3. For each replicate, draw :math:`\pm 1` Rademacher signs (one per
       cluster, broadcast to observations), compute weighted
       :math:`g_bar^*`, and evaluate :math:`K^*`, :math:`J^*`,
       :math:`S^* = J^* - K^*` using the cached projection.

    Omega is held fixed at its sample value across replicates -- the
    standard wild-bootstrap recipe for test statistics.  Recomputing
    ``Omega^*`` per replicate would add Monte Carlo noise without
    statistical gain; under :math:`\pm 1` Rademacher cluster signs
    ``Omega^*`` (centred) is approximately equal to ``Omega`` anyway.

    Penalty independence: ``K(theta_0)`` is a pure function of
    ``(restriction, theta_0, data)`` and the bootstrap inherits this
    property -- the function produces identical p-values on penalised
    and unpenalised ``GMMResult`` instances fit on the same data.

    Parameters
    ----------
    result:
        The ``GMMResult`` whose restriction supplies the moment
        function, cluster structure (if any), and tangent basis at
        ``theta_0``.
    theta_0:
        Hypothesised parameter value under H0.  Required (no default);
        the bootstrap of K at the estimator itself (``theta_0=None``)
        would conflate with #21's open derivation.
    n_replicates:
        Number of bootstrap replicates.  Default 200, matching the
        order of magnitude callers typically use for percentile
        inference.
    cluster_index:
        Optional override for the cluster assignment.  When ``None``
        (default), falls back to ``result.restriction.clusters``; if
        that is also ``None``, the bootstrap is per-observation iid
        (each observation gets its own sign).
    ridge_condition:
        Target condition number passed to
        :func:`~manifoldgmm.utils.numeric.ridge_inverse` for the data-
        side inversions.  See :meth:`GMMResult.k_statistic`'s
        docstring for the ``cond >> ridge_condition`` workaround.
    rng:
        ``numpy.random.Generator``, integer seed, or ``None`` for a
        fresh default RNG.

    Returns
    -------
    KStatBootstrapResult

    References
    ----------
    Kleibergen, F. (2005). "Testing Parameters in GMM Without
    Assuming that They Are Identified." *Econometrica*, 73(4),
    1103--1123.

    Davidson, R. and Flachaire, E. (2008). "The wild bootstrap, tamed
    at last." *Journal of Econometrics*, 146(1), 162--169.
    """

    from scipy.stats import chi2 as chi2_dist

    from ..utils.numeric import ridge_inverse

    restriction = result.restriction

    # Resolve theta_0 to a ManifoldPoint matching the result's manifold.
    if isinstance(theta_0, ManifoldPoint):
        eval_point: ManifoldPoint = theta_0
    else:
        eval_point = ManifoldPoint(result._theta.manifold, theta_0)

    # 1. Per-observation moment matrix at theta_0; shape (N, ell).
    # ``moment_contributions`` returns a 1-D array of length N when ``ell == 1``;
    # promote to (N, 1) so downstream matmul indexing is uniform.
    g_matrix = np.asarray(restriction.moment_contributions(eval_point), dtype=float)
    if g_matrix.ndim == 1:
        g_matrix = g_matrix.reshape(-1, 1)
    if g_matrix.ndim != 2:
        raise ValueError(
            "moment_contributions must return a 1-D (length N, ell==1) or "
            f"2-D (N, ell) matrix; got shape {g_matrix.shape!r}."
        )
    N, ell = g_matrix.shape

    # 2. Cluster structure: explicit override > restriction.clusters > iid.
    if cluster_index is not None:
        labels = np.asarray(cluster_index)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        if labels.size != N:
            raise ValueError(
                f"cluster_index has {labels.size} entries; expected {N} "
                "(one per observation)."
            )
        _, codes = np.unique(labels, return_inverse=True)
        codes = np.asarray(codes, dtype=np.int64)
        G = int(codes.max() + 1) if codes.size > 0 else 0
        cluster_source = "argument"
    elif restriction.clusters is not None:
        codes, G = restriction._resolve_cluster_codes(N)
        cluster_source = "restriction"
    else:
        codes = np.arange(N, dtype=np.int64)
        G = N
        cluster_source = None

    method = "iid" if cluster_source is None else "cluster_wild"
    cluster_info: Mapping[str, Any] | None = (
        None
        if cluster_source is None
        else {"num_clusters": G, "source": cluster_source}
    )

    # 3. g_bar and centred contributions.  g_bar = (1/sqrt(N)) sum g_i;
    # individual contributions are centred by subtracting mean(g_i)
    # = g_bar / sqrt(N).
    sqrt_N = float(N) ** 0.5
    g_mean = g_matrix.mean(axis=0)  # shape (ell,)
    g_bar = sqrt_N * g_mean  # matches MomentRestriction.g_bar convention
    g_centered = g_matrix - g_mean  # shape (N, ell)

    # 4. Data-side ingredients at theta_0.  Use cached Jacobian when
    # eval_point is result._theta; recompute fresh otherwise (mirrors
    # ``k_statistic``'s caching logic).
    if eval_point is result._theta:
        D = result.canonical_jacobian()
    else:
        basis = restriction.tangent_basis(eval_point)
        D = restriction.jacobian_matrix(eval_point, basis=basis)
    D = np.asarray(D, dtype=float)

    omega = np.asarray(restriction.omega_hat(eval_point), dtype=float)
    omega_inv, _ = ridge_inverse(omega, target_condition=ridge_condition)
    DtW = D.T @ omega_inv  # (p, ell)
    DtWD = DtW @ D
    DtWD_inv, _ = ridge_inverse(DtWD, target_condition=ridge_condition)
    P = DtW.T @ DtWD_inv @ DtW  # (ell, ell); projection under Omega^{-1}

    # 5. Observed statistics at theta_0.
    K_observed = float(g_bar @ P @ g_bar)
    J_observed = float(g_bar @ omega_inv @ g_bar)
    S_observed = max(J_observed - K_observed, 0.0)
    p = D.shape[1]
    df_K = p
    df_S = max(ell - p, 0)

    # 6. Bootstrap loop, vectorised over replicates.
    rng_ = np.random.default_rng(rng)
    # Cluster signs: shape (n_replicates, G); broadcast to (n_replicates, N).
    cluster_signs = (2 * rng_.integers(0, 2, size=(n_replicates, G)) - 1).astype(float)
    weights = cluster_signs[:, codes]  # (n_replicates, N)
    # gbar_star[b, k] = (1/sqrt(N)) sum_i weights[b, i] * g_centered[i, k]
    gbar_star = (weights @ g_centered) / sqrt_N  # (n_replicates, ell)
    K_bootstrap = np.einsum("bi,ij,bj->b", gbar_star, P, gbar_star)
    J_bootstrap = np.einsum("bi,ij,bj->b", gbar_star, omega_inv, gbar_star)
    S_bootstrap = np.maximum(J_bootstrap - K_bootstrap, 0.0)

    # 7. p-values.  Bootstrap: small-sample-corrected percentile
    # ``(1 + #{K* >= K_obs}) / (1 + n_replicates)``.  Asymptotic:
    # chi^2 survival from the same observed statistic.
    p_K_boot = float((1 + np.sum(K_bootstrap >= K_observed)) / (1 + n_replicates))
    p_S_boot = float((1 + np.sum(S_bootstrap >= S_observed)) / (1 + n_replicates))
    p_K_asy = (
        float(1.0 - chi2_dist.cdf(K_observed, df=df_K)) if df_K > 0 else float("nan")
    )
    p_S_asy = (
        float(1.0 - chi2_dist.cdf(S_observed, df=df_S)) if df_S > 0 else float("nan")
    )

    return KStatBootstrapResult(
        K_observed=K_observed,
        S_observed=S_observed,
        J_observed=J_observed,
        K_bootstrap=K_bootstrap,
        S_bootstrap=S_bootstrap,
        J_bootstrap=J_bootstrap,
        df_K=df_K,
        df_S=df_S,
        p_K_bootstrap=p_K_boot,
        p_S_bootstrap=p_S_boot,
        p_K_asymptotic=p_K_asy,
        p_S_asymptotic=p_S_asy,
        n_replicates=n_replicates,
        method=method,
        cluster_info=cluster_info,
    )


__all__ = [
    "MomentWildBootstrap",
    "BootstrapTask",
    "BootstrapResult",
    "KStatBootstrapResult",
    "geodesic_mahalanobis_distance",
    "k_statistic_bootstrap_for_result",
    "rademacher_weights",
    "rademacher_signs",
    "mammen_weights",
    "exponential_weights",
]
