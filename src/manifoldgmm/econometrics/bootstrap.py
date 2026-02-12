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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # Optional dependency for richer pickling support
    import cloudpickle
except ImportError:  # pragma: no cover - optional
    cloudpickle = None

from ..geometry import ManifoldPoint
from .gmm import GMM, FixedWeighting, GMMResult
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
    weight_scheme : str
        Name of the weight distribution (``"rademacher"``, ``"mammen"``, or
        ``"exponential"``).
    optimizer_class : type or None
        Optimizer class to use (default: pymanopt TrustRegions).
    optimizer_kwargs : dict
        Extra keyword arguments forwarded to the optimizer constructor.
    task_id : int
        Integer identifying this replicate.
    """

    restriction: MomentRestriction
    weighting_matrix: Any
    initial_point: Any
    seed: int
    weight_scheme: str
    optimizer_class: type | None
    optimizer_kwargs: dict[str, Any]
    task_id: int

    def run(self) -> BootstrapResult:
        """Execute the bootstrap replicate (worker entry point).

        Steps
        -----
        1. Generate observation weights from ``seed`` using the named scheme.
        2. Create a weighted clone via ``restriction.with_weights(w)``.
        3. Construct a ``GMM`` estimator with ``FixedWeighting(W)`` and run it.
        4. Return a lightweight ``BootstrapResult``.
        """

        rng = np.random.default_rng(self.seed)

        generator = _WEIGHT_GENERATORS.get(self.weight_scheme)
        if generator is None:
            raise ValueError(
                f"Unknown weight scheme {self.weight_scheme!r}; "
                f"choose from {sorted(_WEIGHT_GENERATORS)}"
            )

        n = self.restriction.num_observations
        if n is None:
            raise RuntimeError(
                "Cannot determine observation count from restriction; "
                "evaluate the restriction at a point first."
            )

        weights = generator(n, rng)
        weighted_restriction = self.restriction.with_weights(weights)

        optimizer_kwargs = dict(self.optimizer_kwargs)
        optimizer_kwargs.setdefault("verbosity", 0)

        gmm = GMM(
            weighted_restriction,
            weighting=FixedWeighting(self.weighting_matrix),
            optimizer=self.optimizer_class,
            initial_point=self.initial_point,
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
        except (pickle.PicklingError, TypeError):
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

    coords = np.array(
        [float(inner(base_value, e_j, tangent_vector)) for e_j in basis]
    )
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
            np.asarray(t) - np.asarray(b) for b, t in zip(base, target)
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
    weight_scheme : str, default ``"rademacher"``
        Weight distribution: ``"rademacher"``, ``"mammen"``, or
        ``"exponential"``.
    base_seed : int, default 0
        Base random seed; replicate *b* uses seed ``base_seed + b``.
    optimizer_class : type or None
        Optimizer class forwarded to each task.
    optimizer_kwargs : dict or None
        Extra optimizer keyword arguments.
    """

    def __init__(
        self,
        gmm_result: GMMResult,
        n_bootstrap: int = 199,
        *,
        weight_scheme: str = "rademacher",
        base_seed: int = 0,
        optimizer_class: type | None = None,
        optimizer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if weight_scheme not in _WEIGHT_GENERATORS:
            raise ValueError(
                f"Unknown weight_scheme {weight_scheme!r}; "
                f"choose from {sorted(_WEIGHT_GENERATORS)}"
            )

        self._gmm_result = gmm_result
        self._n_bootstrap = n_bootstrap
        self._weight_scheme = weight_scheme
        self._base_seed = base_seed
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = dict(optimizer_kwargs or {})

        # Extract a fixed weighting matrix W evaluated at theta_hat
        weighting = gmm_result.weighting
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
        theta_hat = self._gmm_result.theta_point
        # Use the ambient value as initial point for each replicate
        initial_point = theta_hat.value

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
        dispatch the tasks from :meth:`tasks` to a cluster scheduler.

        Returns
        -------
        list of BootstrapResult
        """

        task_list = self.tasks()
        results = [task.run() for task in task_list]
        self.collect(results)
        return results

    # ------------------------------------------------------------------
    # Geodesic distances
    # ------------------------------------------------------------------

    def geodesic_distances(
        self, *, covariance: np.ndarray | None = None
    ) -> np.ndarray:
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
                result, br.theta_star, covariance=covariance,
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


__all__ = [
    "MomentWildBootstrap",
    "BootstrapTask",
    "BootstrapResult",
    "geodesic_mahalanobis_distance",
    "rademacher_weights",
    "mammen_weights",
    "exponential_weights",
]
