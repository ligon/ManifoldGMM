"""High-level GMM estimator built on top of :class:`MomentRestriction`."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import jax.numpy as jnp
import numpy as np
from datamat import DataMat

try:  # Optional dependency for richer pickling support
    import cloudpickle
except ImportError:  # pragma: no cover - optional
    cloudpickle = None

from pymanopt import Problem
from pymanopt.function import jax as pymanopt_jax_function
from pymanopt.function import numpy as pymanopt_numpy_function
from pymanopt.optimizers import TrustRegions
from pymanopt.optimizers.optimizer import Optimizer

from ..geometry import Manifold, ManifoldPoint
from .moment_restriction import MomentRestriction


class WeightingStrategy(Protocol):
    """Protocol for objects returning a weighting matrix W(θ)."""

    def matrix(self, theta: Any) -> Any:
        """Return the ℓ×ℓ weighting matrix evaluated at ``theta``."""

    def info(self) -> Mapping[str, Any]:  # pragma: no cover - default impl used
        """Metadata describing the weighting strategy."""


class FixedWeighting:
    """Always return the same weighting matrix regardless of θ."""

    def __init__(self, matrix: Any, *, label: str | None = None) -> None:
        self._matrix = matrix
        self._label = label or "fixed"

    def matrix(self, theta: Any) -> Any:  # noqa: D401 - simple delegation
        return self._matrix

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


class CallableWeighting:
    """Wrap a callable ``theta -> W`` as a :class:`WeightingStrategy`."""

    def __init__(self, fn: Callable[[Any], Any], *, label: str | None = None) -> None:
        self._fn = fn
        self._label = label or "callable"

    def matrix(self, theta: Any) -> Any:  # noqa: D401 - simple delegation
        try:
            return self._fn(theta)
        except TypeError:
            if isinstance(theta, ManifoldPoint):
                return self._fn(theta.value)
            raise

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


class CUEWeighting:
    """Continuously updated weighting based on Ω̂(θ)⁻¹.

    Parameters
    ----------
    restriction : MomentRestriction
        The moment restriction providing omega_hat(theta).
    ridge : float, default 0.0
        Minimum ridge regularization to add: W = (Ω + ridge·I)⁻¹.
    target_condition : float or None, default None
        If set, adaptively choose ridge at each evaluation to keep
        cond(Ω + ridge·I) ≤ target_condition. This handles cases where
        Ω(θ) becomes ill-conditioned as θ moves through the parameter space.
        The effective ridge is max(ridge, adaptive_ridge).
    """

    def __init__(
        self,
        restriction: MomentRestriction,
        ridge: float = 0.0,
        target_condition: float | None = None,
    ) -> None:
        self._restriction = restriction
        self._ridge = ridge
        self._target_condition = target_condition
        # Track diagnostics for inference validity
        self._last_ridge: float = ridge
        self._last_condition: float = 1.0
        self._last_lambda_min: float = 1.0  # smallest eigenvalue of Ω (before ridge)

    def matrix(self, theta: Any) -> Any:
        omega = self._restriction.omega_hat(theta)
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        omega_array = xp.asarray(omega)

        if self._target_condition is not None:
            is_jax = hasattr(xp, "where")

            if is_jax:
                # JAX path: compute eigenvalues (needed for tracing, can't branch)
                # eigvalsh is O(n³) same as cond, and we need eigenvalues for ridge
                eigvals = linalg.eigvalsh(omega_array)
                lambda_max = eigvals[-1]
                lambda_min = eigvals[0]
                current_cond = lambda_max / (xp.abs(lambda_min) + 1e-15)

                adaptive_ridge = lambda_max / self._target_condition - lambda_min
                adaptive_ridge = xp.maximum(adaptive_ridge, self._ridge)
                ridge = xp.where(
                    current_cond > self._target_condition,
                    adaptive_ridge,
                    xp.asarray(self._ridge, dtype=omega_array.dtype),
                )

                # Store diagnostics (skip during tracing)
                try:
                    self._last_condition = float(current_cond)
                    self._last_ridge = float(ridge)
                    self._last_lambda_min = float(xp.abs(lambda_min))
                except (TypeError, ValueError):
                    pass

                # Always add ridge for JAX (value may be 0)
                omega_array = omega_array + ridge * xp.eye(
                    omega_array.shape[0], dtype=omega_array.dtype
                )
            else:
                # NumPy path: use cond() first, only compute eigenvalues if needed
                current_cond = float(linalg.cond(omega_array))
                self._last_condition = current_cond

                if current_cond > self._target_condition:
                    eigvals = linalg.eigvalsh(omega_array)
                    lambda_max, lambda_min = float(eigvals[-1]), float(eigvals[0])
                    self._last_lambda_min = abs(lambda_min)
                    ridge = max(
                        lambda_max / self._target_condition - lambda_min,
                        self._ridge,
                    )
                    self._last_ridge = ridge
                    omega_array = omega_array + ridge * xp.eye(
                        omega_array.shape[0], dtype=omega_array.dtype
                    )
                elif self._ridge > 0.0:
                    self._last_ridge = self._ridge
                    # Compute lambda_min for diagnostic even when using fixed ridge
                    eigvals = linalg.eigvalsh(omega_array)
                    self._last_lambda_min = abs(float(eigvals[0]))
                    omega_array = omega_array + self._ridge * xp.eye(
                        omega_array.shape[0], dtype=omega_array.dtype
                    )
                else:
                    self._last_ridge = 0.0
                    self._last_lambda_min = 1.0  # Not computed when no ridge

        elif self._ridge > 0.0:
            # Fixed ridge (no adaptive) - compute lambda_min for diagnostic
            linalg = getattr(self._restriction, "_linalg", np.linalg)
            try:
                eigvals = linalg.eigvalsh(omega_array)
                self._last_lambda_min = abs(float(eigvals[0]))
                self._last_condition = float(eigvals[-1]) / (
                    self._last_lambda_min + 1e-15
                )
            except (TypeError, ValueError):
                pass  # Skip during JAX tracing
            self._last_ridge = self._ridge
            omega_array = omega_array + self._ridge * xp.eye(
                omega_array.shape[0], dtype=omega_array.dtype
            )

        return linalg.inv(omega_array)

    def info(self) -> Mapping[str, Any]:
        # Compute ridge_ratio: how much ridge dominates smallest eigenvalue
        # ridge_ratio > 0.1 suggests potential distortion of test statistics
        # ridge_ratio > 1.0 means ridge completely dominates λ_min
        # When lambda_min is near-zero (singular), ridge completely dominates
        if self._last_lambda_min < 1e-14:
            # Essentially singular - ridge dominates completely
            ridge_ratio = float("inf") if self._last_ridge > 0 else 0.0
        elif self._last_lambda_min > 0:
            ridge_ratio = self._last_ridge / self._last_lambda_min
        else:
            ridge_ratio = 0.0

        # Flag inference concerns
        inference_warning = None
        if ridge_ratio == float("inf") or ridge_ratio > 1.0:
            inference_warning = (
                f"Ridge ({self._last_ridge:.2e}) exceeds λ_min ({self._last_lambda_min:.2e}). "
                "Test statistics (J, Wald) may be substantially distorted."
            )
        elif ridge_ratio > 0.1:
            inference_warning = (
                f"Ridge is {ridge_ratio:.1%} of λ_min. "
                "Test statistics may have mild size distortion."
            )

        return {
            "type": "cue",
            "ridge": self._ridge,
            "target_condition": self._target_condition,
            "last_ridge": self._last_ridge,
            "last_condition": self._last_condition,
            "last_lambda_min": self._last_lambda_min,
            "ridge_ratio": ridge_ratio,
            "inference_warning": inference_warning,
        }


class IdentityWeighting(FixedWeighting):
    """Identity matrix weighting used for first-stage two-step GMM."""

    def __init__(self, dimension: int) -> None:
        super().__init__(np.eye(dimension, dtype=float), label="identity")


@dataclass
class WaldTestResult:
    """Result of a Wald test for H0: h(theta) = 0.

    Attributes:
        statistic: The Wald statistic W, asymptotically chi-squared distributed.
        degrees_of_freedom: The number of constraints q.
        p_value: The probability of observing a statistic > W under H0.
    """

    statistic: float
    degrees_of_freedom: int
    p_value: float


@dataclass
class KStatisticResult:
    """Result of the Kleibergen (2005) K-statistic decomposition.

    At a hypothesised value ``theta_0``, the efficient J-statistic
    decomposes as ``J(theta_0) = K(theta_0) + S(theta_0)``, where K
    and S are asymptotically independent under
    ``H0: theta = theta_0``.

    Attributes:
        K: K-statistic (score / LM component), chi2(df_K) under H0.
        S: S-statistic (overidentification complement), chi2(df_S) under H0.
        J: Efficient J-statistic (= K + S).
        df_K: Degrees of freedom for K (= manifold dimension p).
        df_S: Degrees of freedom for S (= ell - p).
        p_K: p-value for K under chi2(df_K).
        p_S: p-value for S under chi2(df_S).
    """

    K: float
    S: float
    J: float
    df_K: int
    df_S: int
    p_K: float
    p_S: float


@dataclass
class GMMResult:
    """Container returned by :meth:`GMM.estimate`."""

    _theta: ManifoldPoint
    criterion_value: float
    degrees_of_freedom: int
    weighting_info: Mapping[str, Any]
    weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    optimizer_report: Mapping[str, Any]
    restriction: MomentRestriction
    g_bar: Any
    two_step: bool
    _theta_labeled: Any | None = field(default=None, init=False, repr=False)
    # Lazy cache of the canonical Jacobian D bar g_N(theta_hat) in the
    # canonical tangent basis at theta_hat.  Computed on first access by
    # ``canonical_jacobian``; reused by ``tangent_covariance``,
    # ``k_statistic`` (when theta_0 is None or equals theta_hat), and any
    # external code calling ``canonical_jacobian`` directly.
    _cached_jacobian: np.ndarray | None = field(default=None, init=False, repr=False)
    _cached_jacobian_basis: list[Any] | None = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def to_pickle(
        self, path: str | Path, *, protocol: int = pickle.HIGHEST_PROTOCOL
    ) -> None:
        """Serialise the result to ``path`` using :mod:`pickle`."""

        file_path = Path(path)
        try:
            with file_path.open("wb") as handle:
                pickle.dump(self, handle, protocol=protocol)
            return
        except (pickle.PicklingError, TypeError, AttributeError):
            if cloudpickle is None:
                raise
        with file_path.open("wb") as handle:
            cloudpickle.dump(self, handle)

    @staticmethod
    def from_pickle(path: str | Path) -> GMMResult:
        """Load a pickled :class:`GMMResult` from ``path``."""

        file_path = Path(path)
        with file_path.open("rb") as handle:
            try:
                obj = pickle.load(handle)
            except Exception:
                if cloudpickle is None:
                    raise
                handle.seek(0)
                obj = cloudpickle.load(handle)
        if not isinstance(obj, GMMResult):  # pragma: no cover - safety check
            raise TypeError("Pickle does not contain a GMMResult")
        return obj

    # ------------------------------------------------------------------
    # Cached Jacobian at theta_hat
    # ------------------------------------------------------------------
    def canonical_jacobian(self, *, basis: list[Any] | None = None) -> np.ndarray:
        r"""Return the canonical Jacobian :math:`D\bar g_N(\hat\theta)`.

        The canonical basis at :math:`\hat\theta` is fixed once
        ``GMMResult`` is constructed, so the matrix is identical across
        ``tangent_covariance``, ``wald_test`` (via ``tangent_covariance``),
        and ``k_statistic`` (when ``theta_0`` is ``None``).  This method
        memoises the dense matrix on first access; subsequent callers
        reuse the cached array.

        Parameters
        ----------
        basis:
            Optional tangent basis.  When supplied, the cached value is
            used only if it matches the cached basis by object identity;
            otherwise the Jacobian is recomputed (and the cache is left
            untouched, since custom bases are typically a one-off).
            When ``None``, the canonical basis from
            :meth:`MomentRestriction.tangent_basis` is used and the
            result is cached.

        Returns
        -------
        numpy.ndarray
            Dense ``(ell, p)`` Jacobian in the chosen basis.

        Notes
        -----
        For large ``N`` (e.g., :math:`N \sim 10^5`) the Jacobian
        computation can dominate the cost of ``tangent_covariance``,
        ``wald_test``, and ``k_statistic``.  See #4 for context; this
        cache is the small-footprint half of that fix, paired with the
        ``jax.vmap`` batched assembly in
        :meth:`MomentRestriction.jacobian_matrix`.
        """

        if basis is not None:
            # Custom basis: reuse the cache only if the caller hands us the
            # very list we cached earlier (object identity).  Otherwise we
            # compute fresh without disturbing the canonical cache.
            if (
                self._cached_jacobian is not None
                and basis is self._cached_jacobian_basis
            ):
                return self._cached_jacobian
            return self.restriction.jacobian_matrix(self._theta, basis=basis)

        if self._cached_jacobian is None:
            basis_vectors = self.restriction.tangent_basis(self._theta)
            jac = self.restriction.jacobian_matrix(self._theta, basis=basis_vectors)
            # ``object.__setattr__`` because the dataclass is mutable by
            # default but the fields with init=False default to None and
            # we explicitly want to write through.
            self._cached_jacobian = jac
            self._cached_jacobian_basis = basis_vectors
        return self._cached_jacobian

    def tangent_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        """Return the sandwich covariance in the canonical tangent coordinates."""

        theta_hat = self._theta
        restriction = self.restriction
        if basis is None:
            jac_matrix = self.canonical_jacobian()
            # The basis used in the cache is what ``canonical_jacobian``
            # constructed; resurface it for ``manifold_covariance`` and
            # other downstream consumers that index by basis.
            basis_vectors = self._cached_jacobian_basis
            assert basis_vectors is not None  # set by canonical_jacobian
        else:
            basis_vectors = basis
            jac_matrix = self.canonical_jacobian(basis=basis_vectors)
        weights = weighting or self.weighting

        if weights is None:
            raise ValueError("No weighting strategy available to compute covariance")

        if hasattr(weights, "matrix") and callable(weights.matrix):
            W = weights.matrix(theta_hat)
        elif callable(weights):
            W = weights(theta_hat)
        else:
            W = weights

        W_array = np.asarray(W, dtype=float)
        jac_array = np.asarray(jac_matrix, dtype=float)
        omega_array = np.asarray(restriction.omega_hat(theta_hat), dtype=float)

        from ..utils.numeric import ridge_inverse

        XtWX = jac_array.T @ W_array @ jac_array
        inv_XtWX, ridge = ridge_inverse(XtWX, target_condition=ridge_condition)
        middle = jac_array.T @ W_array @ omega_array @ W_array @ jac_array
        covariance = inv_XtWX @ middle @ inv_XtWX

        if ridge != 0.0:
            covariance = np.asarray(covariance)  # ensure materialised array

        basis_labels = [f"basis[{index}]" for index in range(covariance.shape[0])]
        return DataMat(covariance, index=basis_labels, columns=basis_labels)

    def manifold_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        """Push forward the tangent covariance to ambient coordinates."""

        restriction = self.restriction
        base_point = self.theta_point
        basis_vectors = (
            basis if basis is not None else restriction.tangent_basis(base_point)
        )
        cov_tangent = self.tangent_covariance(
            weighting=weighting, ridge_condition=ridge_condition, basis=basis_vectors
        )

        columns: list[np.ndarray] = []
        for direction in basis_vectors:
            flat = restriction._array_adapter(direction)
            columns.append(np.asarray(flat, dtype=float).reshape(-1))

        if not columns:
            zero = np.zeros((0, 0), dtype=float)
            return DataMat(zero)

        chart_jacobian = np.column_stack(columns)
        covariance = (
            chart_jacobian @ cov_tangent.to_numpy(dtype=float) @ chart_jacobian.T
        )

        labels = list(restriction.parameter_labels or ())
        if len(labels) != covariance.shape[0]:
            labels = [f"theta[{index}]" for index in range(covariance.shape[0])]

        return DataMat(covariance, index=labels, columns=labels)

    @property
    def theta(self) -> ManifoldPoint:
        """Estimated parameter as a :class:`ManifoldPoint`."""

        if self._theta.formatted is self._theta.value:
            _ = self.theta_labeled
        return self._theta

    @property
    def theta_point(self) -> ManifoldPoint:
        """Explicit alias for the manifold-valued estimate."""

        return self.theta

    @property
    def theta_labeled(self) -> Any:
        """Labelled parameter estimate for user-facing consumption."""

        if self._theta_labeled is None:
            formatted = self._theta.formatted
            if formatted is self._theta.value:
                formatted = self.restriction.format_parameter(self._theta.value)
                self._theta = ManifoldPoint(
                    self._theta.manifold,
                    self._theta.value,
                    formatted=formatted,
                )
            self._theta_labeled = formatted
        return self._theta_labeled

    @property
    def theta_array(self) -> Any:
        """Raw parameter estimate suitable for numerical processing."""

        return self.theta.value

    def ambient_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> DataMat:
        """Backward-compatible alias for :meth:`manifold_covariance`."""

        return self.manifold_covariance(
            weighting=weighting, ridge_condition=ridge_condition, basis=basis
        )

    def as_dict(self) -> Mapping[str, Any]:
        """Return the result as a dictionary for quick inspection."""

        return {
            "theta": self.theta_labeled,
            "criterion_value": self.criterion_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "weighting": dict(self.weighting_info),
            "optimizer_report": dict(self.optimizer_report),
            "two_step": self.two_step,
        }

    def check_inference_validity(self, warn: bool = True) -> Mapping[str, Any]:
        """Check whether ridge regularization may distort test statistics.

        When CUE weighting uses ridge regularization, the weighting matrix
        W = (Ω + λI)⁻¹ ≠ Ω⁻¹, which can distort the asymptotic distribution
        of test statistics (J-statistic, Wald tests).

        Parameters
        ----------
        warn : bool, default True
            If True and ridge_ratio > 0.1, print a warning.

        Returns
        -------
        dict with keys:
            ridge_ratio : float
                Ratio of ridge to smallest eigenvalue of Ω.
                - < 0.01: negligible effect on inference
                - 0.01-0.1: minor effect, standard inference likely OK
                - 0.1-1.0: moderate effect, consider bootstrap
                - > 1.0: substantial effect, standard inference unreliable
            lambda_min : float
                Smallest eigenvalue of Ω (before ridge).
            ridge : float
                Ridge value used.
            inference_warning : str or None
                Warning message if ridge_ratio > 0.1.
        """
        import warnings

        info = self.weighting_info
        result = {
            "ridge_ratio": info.get("ridge_ratio", 0.0),
            "lambda_min": info.get("last_lambda_min", None),
            "ridge": info.get("last_ridge", 0.0),
            "inference_warning": info.get("inference_warning", None),
        }

        if warn and result["inference_warning"]:
            warnings.warn(result["inference_warning"], UserWarning, stacklevel=2)

        return result

    def wald_test(
        self,
        constraint: Callable[[ManifoldPoint], Any],
        q: int | None = None,
    ) -> WaldTestResult:
        """
        Perform a Wald test for H0: h(theta) = 0.

        Parameters
        ----------
        constraint:
            Function mapping a ManifoldPoint to a vector of size q.
            Returns either a JAX array or NumPy array.
        q:
            Number of constraints (dimension of h(theta)). If None, inferred from output.

        Returns
        -------
        WaldTestResult
            Object containing the Wald statistic, degrees of freedom, and p-value.
        """
        from scipy.stats import chi2

        from ..autodiff import jacobian_operator

        theta_hat = self._theta

        # 1. Evaluate constraint at estimate
        h_val = constraint(theta_hat)
        h_val = np.asarray(h_val, dtype=float).flatten()

        if q is None:
            q = h_val.size

        if q == 0:
            return WaldTestResult(0.0, 0, 1.0)

        # 2. Compute Jacobian of h w.r.t. tangent vector xi at xi=0
        # We leverage JacobianOperator which abstracts JAX/autodiff path
        manifold = theta_hat.manifold
        basis = self.restriction.tangent_basis(theta_hat)
        dim = len(basis)

        try:
            op = jacobian_operator(constraint, theta_hat)
            H_cols = []
            for b in basis:
                # op.matvec computes the directional derivative Dh(theta)[b]
                col = op.matvec(b)
                H_cols.append(np.asarray(col, dtype=float).flatten())
            H = np.column_stack(H_cols)
        except Exception:
            # Fallback to finite differences
            epsilon = 1e-5

            def _scale_structure(struct: Any, factor: float) -> Any:
                if isinstance(struct, tuple | list):
                    return type(struct)(_scale_structure(c, factor) for c in struct)
                return np.asarray(struct) * factor

            def _add_structure(lhs: Any, rhs: Any) -> Any:
                if isinstance(lhs, tuple | list):
                    return type(lhs)(
                        _add_structure(lhs_part, rhs_part)
                        for lhs_part, rhs_part in zip(lhs, rhs, strict=False)
                    )
                return np.asarray(lhs) + np.asarray(rhs)

            def composed_map_numpy(xi: np.ndarray) -> Any:
                tangent_vector = None
                for i, b in enumerate(basis):
                    term = _scale_structure(b, float(xi[i]))
                    if tangent_vector is None:
                        tangent_vector = term
                    else:
                        tangent_vector = _add_structure(tangent_vector, term)

                assert manifold.data is not None  # narrowed for mypy
                retraction_fn = getattr(manifold.data, "retraction", None)
                if retraction_fn is None:
                    retraction_fn = manifold.data.retract

                new_value = retraction_fn(theta_hat.value, tangent_vector)
                new_point = ManifoldPoint(manifold, new_value)
                return constraint(new_point)

            # Central difference
            H_cols = []
            for i in range(dim):
                xi_plus = np.zeros(dim)
                xi_plus[i] = epsilon
                val_plus = np.asarray(
                    composed_map_numpy(xi_plus), dtype=float
                ).flatten()

                xi_minus = np.zeros(dim)
                xi_minus[i] = -epsilon
                val_minus = np.asarray(
                    composed_map_numpy(xi_minus), dtype=float
                ).flatten()

                col = (val_plus - val_minus) / (2 * epsilon)
                H_cols.append(col)

            H = np.column_stack(H_cols)

        # H should be (q, dim). If q=1, jax.jacobian might return (dim,) or (1, dim) depending on output
        if H.ndim == 1 and q == 1:
            H = H.reshape(1, -1)

        # 3. Get Covariance
        Sigma = self.tangent_covariance().to_numpy()

        # 4. Compute W = h' (H Sigma H')^-1 h
        denom = H @ Sigma @ H.T

        try:
            # Use solve for better numerical stability than inv
            if q == 1:
                W = (h_val**2) / denom.item()
            else:
                W = h_val @ np.linalg.solve(denom, h_val)
        except np.linalg.LinAlgError:
            W = np.nan

        # 5. p-value
        if np.ndim(W) == 0:
            W_scalar = float(W)
        else:
            W_scalar = float(np.asarray(W).item())

        p_value = 1.0 - chi2.cdf(W_scalar, df=q)

        return WaldTestResult(W_scalar, int(q), float(p_value))

    def k_statistic(
        self,
        *,
        theta_0: ManifoldPoint | Any | None = None,
        ridge_condition: float = 1e8,
    ) -> KStatisticResult:
        r"""Kleibergen (2005) K-statistic decomposition.

        Decomposes the efficient J-statistic at ``theta_0`` as
        :math:`J(\theta_0) = K(\theta_0) + S(\theta_0)` where:

        - :math:`K` is a score / LM-type statistic, :math:`\chi^2(p)` under
          :math:`H_0\colon \theta = \theta_0` regardless of identification
          strength.
        - :math:`S` captures the overidentifying restrictions,
          :math:`\chi^2(\ell - p)` under :math:`H_0`.

        Parameters
        ----------
        theta_0 : ManifoldPoint or array-like, optional
            The parameter value at which to evaluate the decomposition.
            This is the hypothesised value under :math:`H_0`.  When
            ``None`` (default), the estimator :math:`\hat\theta` is used;
            note that K evaluated at the estimator is typically near zero
            because the first-order condition zeroes the score.
        ridge_condition : float, default 1e8
            Target condition number for matrix inversions via
            :func:`~manifoldgmm.utils.numeric.ridge_inverse`.

        Returns
        -------
        KStatisticResult

        References
        ----------
        Kleibergen, F. (2005). "Testing Parameters in GMM Without
        Assuming that They Are Identified." *Econometrica*, 73(4),
        1103--1123.
        """
        from scipy.stats import chi2 as chi2_dist

        from ..utils.numeric import ridge_inverse

        restriction = self.restriction

        # Resolve evaluation point
        if theta_0 is not None:
            if not isinstance(theta_0, ManifoldPoint):
                theta_0 = ManifoldPoint(self._theta.manifold, theta_0)
            eval_point = theta_0
        else:
            eval_point = self._theta

        # 1. Ingredients: g_bar, Omega, D (all in ManifoldGMM sqrt(N) scaling)
        g_bar_vec = np.asarray(restriction.g_bar(eval_point), dtype=float).reshape(-1)
        omega = np.asarray(restriction.omega_hat(eval_point), dtype=float)
        # Reuse the cached Jacobian when evaluating at the estimator;
        # otherwise compute fresh at the hypothesised value.
        if eval_point is self._theta:
            D = self.canonical_jacobian()
            basis = self._cached_jacobian_basis
            assert basis is not None
        else:
            basis = restriction.tangent_basis(eval_point)
            D = restriction.jacobian_matrix(eval_point, basis=basis)

        ell = g_bar_vec.shape[0]
        p = D.shape[1]

        # 2. Omega^{-1} (efficient weighting)
        omega_inv, _ = ridge_inverse(omega, target_condition=ridge_condition)

        # 3. Efficient J = g_bar' Omega^{-1} g_bar
        J_eff = float(g_bar_vec @ omega_inv @ g_bar_vec)

        # 4. K = g_bar' Omega^{-1} D (D' Omega^{-1} D)^{-1} D' Omega^{-1} g_bar
        #    Using the CUE-score vector:  s = (D'W D)^{-1} D'W g_bar
        #    so that  K = s' (D'W D) s
        DtW = D.T @ omega_inv  # (p, ell)
        DtWD = DtW @ D  # (p, p)
        DtWD_inv, _ = ridge_inverse(DtWD, target_condition=ridge_condition)
        DtW_gbar = DtW @ g_bar_vec  # (p,)
        score_vec = DtWD_inv @ DtW_gbar  # (p,)
        K = float(score_vec @ DtWD @ score_vec)

        # 5. S = J - K
        S = max(J_eff - K, 0.0)

        # 6. Degrees of freedom and p-values
        df_K = p
        df_S = max(ell - p, 0)

        p_K = float(1.0 - chi2_dist.cdf(K, df=df_K)) if df_K > 0 else float("nan")
        p_S = float(1.0 - chi2_dist.cdf(S, df=df_S)) if df_S > 0 else float("nan")

        return KStatisticResult(
            K=K,
            S=S,
            J=J_eff,
            df_K=df_K,
            df_S=df_S,
            p_K=p_K,
            p_S=p_S,
        )

    def in_asymptotic_region(
        self, point: ManifoldPoint | Any, alpha: float = 0.05
    ) -> bool:
        r"""Test whether ``point`` lies inside the asymptotic confidence region.

        The :math:`(1-\alpha)` confidence region is

        .. math::

            \bigl\{\theta : d^2(\hat\theta, \theta) \le \chi^2_{p,\,1-\alpha}\bigr\}

        where :math:`d^2` is the geodesic Mahalanobis distance (see
        :func:`~manifoldgmm.econometrics.bootstrap.geodesic_mahalanobis_distance`)
        and :math:`p` is the manifold dimension.

        Parameters
        ----------
        point : ManifoldPoint or array-like
            Candidate parameter value.
        alpha : float, default 0.05
            Significance level.

        Returns
        -------
        bool
            ``True`` if ``point`` lies inside the region.
        """

        from scipy.stats import chi2

        from .bootstrap import geodesic_mahalanobis_distance

        d2 = geodesic_mahalanobis_distance(self, point)

        # Manifold dimension
        manifold = self.restriction.manifold
        if manifold is not None and manifold.data is not None:
            p = getattr(manifold.data, "dim", None)
            if callable(p):
                p = p()
        else:
            p = None

        if p is None:
            # Fall back to tangent basis length
            p = len(self.restriction.tangent_basis(self.theta_point))

        cv = float(chi2.ppf(1.0 - alpha, df=p))
        return d2 <= cv


class GMM:
    """High-level GMM estimator operating on a :class:`MomentRestriction`."""

    def __init__(
        self,
        restriction: MomentRestriction,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        optimizer: type[Optimizer] | Optimizer | None = None,
        initial_point: Any | None = None,
        cue_ridge: float = 0.0,
        cue_target_condition: float | None = None,
    ) -> None:
        self._restriction = restriction
        self._cue_ridge = cue_ridge
        self._cue_target_condition = cue_target_condition
        self._weighting = self._coerce_weighting(weighting)
        self._optimizer = optimizer
        self._initial_point = initial_point

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    @property
    def moment_restriction(self) -> MomentRestriction:
        return self._restriction

    def g_bar(self, theta: Any) -> Any:
        return self._restriction.g_bar(theta)

    def gN(self, theta: Any) -> Any:
        return self._restriction.gN(theta)

    def omega_hat(self, theta: Any) -> Any:
        return self._restriction.omega_hat(theta)

    def criterion(self, theta: Any) -> float:
        weighting = self._weighting
        return float(self._backend_dot(theta, weighting))

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------
    def estimate(
        self,
        *,
        initial_point: Any | None = None,
        two_step: bool = False,
        weighting_iterations: int | Literal["converge"] = 1,
        weighting_tol: float = 1e-6,
        max_weighting_iterations: int = 25,
        optimizer_kwargs: Mapping[str, Any] | None = None,
        verbose: bool | int | None = None,
    ) -> GMMResult:
        """Run one-step, two-step, or iterated GMM.

        Parameters
        ----------
        initial_point:
            Starting parameter on the manifold.  Falls back to the GMM
            instance's initial point, then to a manifold-aware random draw.
        two_step:
            When ``True`` and ``weighting_iterations == 1`` (default),
            performs the classical two-step procedure (identity weighting
            then ``FixedWeighting(Ω̂(θ̂₁)⁻¹)``).
            Implicit when ``weighting_iterations`` is set to ``> 1`` or
            ``"converge"``.
        weighting_iterations:
            Controls how many reweighting stages follow the initial stage.
            ``1`` (default) reproduces today's behaviour exactly (one-step
            when ``two_step=False``; two-step when ``two_step=True``).  An
            integer ``k > 1`` runs ``k`` reweighting stages after an
            identity-weighted first stage, exposing the *iterated* GMM
            estimator (Hansen, Heaton and Yaron 1996).  ``"converge"``
            iterates until the manifold distance between consecutive
            estimates falls below ``weighting_tol`` or
            ``max_weighting_iterations`` is reached.  Note: this is
            distinct from CUE (see :class:`CUEWeighting`) -- iterated GMM
            holds the weighting fixed within each stage and does not carry
            a ``∂Ω/∂θ`` term in the first-order
            condition.
        weighting_tol:
            Tolerance on the manifold distance between consecutive
            estimates used as the convergence criterion when
            ``weighting_iterations="converge"``.
        max_weighting_iterations:
            Hard cap on the number of reweighting stages when iterating to
            convergence.
        optimizer_kwargs:
            Keyword arguments forwarded to the optimizer constructor or to
            an already-instantiated optimizer.
        verbose:
            Convenience flag for setting optimizer ``verbosity``.

        Returns
        -------
        GMMResult
            Result container whose ``weighting_info`` carries iteration
            diagnostics (``iterations``, ``theta_path`` of manifold
            distances, ``converged``, ``tol``).
        """

        theta_start = (
            initial_point if initial_point is not None else self._initial_point
        )
        if theta_start is None:
            theta_start = self._default_initial_point()
        if theta_start is None:
            raise ValueError("Provide an initial_point to start the optimisation.")

        optimizer_kwargs = dict(optimizer_kwargs or {})
        if verbose is not None and "verbosity" not in optimizer_kwargs:
            if isinstance(verbose, bool):
                optimizer_kwargs["verbosity"] = 2 if verbose else 0
            else:
                optimizer_kwargs["verbosity"] = int(verbose)

        # Normalise the iteration spec.  ``weighting_iterations == 1`` keeps
        # the historical two_step semantics (so callers that pass
        # ``two_step=True`` still get exactly today's behaviour); any value
        # greater than 1 (or ``"converge"``) implies iterated GMM with an
        # identity-weighted first stage, regardless of the ``two_step`` flag.
        converge_mode = weighting_iterations == "converge"
        if converge_mode:
            iteration_cap = int(max_weighting_iterations)
            if iteration_cap < 1:
                raise ValueError(
                    "max_weighting_iterations must be at least 1 when "
                    "weighting_iterations='converge'."
                )
        else:
            if not isinstance(weighting_iterations, int):
                raise TypeError(
                    "weighting_iterations must be an int or 'converge'; "
                    f"got {type(weighting_iterations).__name__}."
                )
            if weighting_iterations < 0:
                raise ValueError(
                    "weighting_iterations must be non-negative; "
                    f"got {weighting_iterations}."
                )
            iteration_cap = int(weighting_iterations)

        iterated = converge_mode or iteration_cap > 1
        # Stage 1 uses identity weighting whenever a reweighting is going to
        # happen (matches the conventional iterated-GMM setup).
        first_stage_reweighted = two_step or iterated

        weighting_stage1 = self._weighting
        if first_stage_reweighted:
            num_moments = self._ensure_metadata(theta_start)
            weighting_stage1 = IdentityWeighting(num_moments)

        stage = self._run_stage(theta_start, weighting_stage1, optimizer_kwargs)
        final_weighting: WeightingStrategy = weighting_stage1

        theta_path: list[float] = []
        iterations_run = 0
        converged = True

        # Reweighting loop.  ``iteration_cap`` is the planned number of
        # reweighting stages; ``converge_mode`` lets us stop earlier (or run
        # up to ``iteration_cap``) based on the manifold distance.
        target = iteration_cap if not converge_mode else int(max_weighting_iterations)
        if not first_stage_reweighted:
            target = 0
        # In two_step (default ``weighting_iterations=1``) the historical
        # label is "two_step"; iterated runs use "iterated".
        reweight_label = (
            "two_step" if (target <= 1 and not converge_mode) else "iterated"
        )

        if converge_mode:
            converged = False  # set True when distance < tol

        if target >= 1:
            _, linalg = self._backend_modules()
            for _ in range(target):
                omega = self._to_backend_matrix(
                    self._restriction.omega_hat(stage.theta)
                )
                updated_weighting = FixedWeighting(
                    linalg.inv(omega), label=reweight_label
                )
                prev_theta_point = stage.theta
                stage = self._run_stage(
                    prev_theta_point,
                    updated_weighting,
                    optimizer_kwargs,
                )
                final_weighting = updated_weighting
                iterations_run += 1

                distance = self._theta_distance(prev_theta_point, stage.theta)
                theta_path.append(distance)

                if converge_mode and distance < weighting_tol:
                    converged = True
                    break

        final_stage = stage

        df = self._degrees_of_freedom(final_stage.g_bar, final_stage.theta)
        weighting_info = dict(final_weighting.info())
        weighting_info.setdefault("two_step", first_stage_reweighted)
        weighting_info["iterations"] = iterations_run
        weighting_info["theta_path"] = list(theta_path)
        weighting_info["converged"] = converged
        weighting_info["tol"] = float(weighting_tol)

        return GMMResult(
            _theta=final_stage.theta,
            criterion_value=float(
                self._backend_dot(final_stage.theta, final_weighting)
            ),
            degrees_of_freedom=df,
            weighting_info=weighting_info,
            weighting=final_weighting,
            optimizer_report=final_stage.optimizer_report,
            restriction=self._restriction,
            g_bar=final_stage.g_bar,
            two_step=first_stage_reweighted,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_weighting(
        self, weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    ) -> WeightingStrategy:
        if weighting is None:
            return CUEWeighting(
                self._restriction,
                ridge=self._cue_ridge,
                target_condition=self._cue_target_condition,
            )
        if hasattr(weighting, "matrix") and callable(weighting.matrix):
            return cast(WeightingStrategy, weighting)
        if callable(weighting):
            return CallableWeighting(weighting)
        return FixedWeighting(weighting)

    # Kwargs that belong on TrustRegions.run() rather than __init__().
    # (pymanopt's TrustRegions.run accepts mininner, maxinner, Delta_bar,
    # Delta0; __init__ takes miniter, kappa, theta, rho_prime, use_rand,
    # rho_regularization, plus base Optimizer kwargs.)
    _OPTIMIZER_RUN_KWARGS = frozenset(
        {"mininner", "maxinner", "Delta_bar", "Delta0"}
    )

    @classmethod
    def _split_optimizer_kwargs(
        cls, optimizer_kwargs: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition optimizer_kwargs into (init_kwargs, run_kwargs)."""
        init_kwargs: dict[str, Any] = {}
        run_kwargs: dict[str, Any] = {}
        for key, value in optimizer_kwargs.items():
            if key in cls._OPTIMIZER_RUN_KWARGS:
                run_kwargs[key] = value
            else:
                init_kwargs[key] = value
        return init_kwargs, run_kwargs

    def _resolve_optimizer(self, optimizer_kwargs: Mapping[str, Any]) -> Optimizer:
        base = self._optimizer
        if base is None:
            return TrustRegions(**optimizer_kwargs)
        if isinstance(base, Optimizer):
            if optimizer_kwargs:
                allowed = {"verbosity", "log_verbosity"}
                unexpected = set(optimizer_kwargs) - allowed
                if unexpected:
                    raise ValueError(
                        "optimizer_kwargs are incompatible with a pre-configured "
                        f"optimizer (unexpected keys: {sorted(unexpected)!r})"
                    )
                for key, value in optimizer_kwargs.items():
                    setattr(base, key, value)
            return base
        return base(**optimizer_kwargs)

    def _run_stage(
        self,
        initial_point: Any,
        weighting: WeightingStrategy,
        optimizer_kwargs: Mapping[str, Any],
    ) -> _StageResult:
        cost = self._build_cost(weighting)
        manifold_wrapper = self._restriction.manifold
        if manifold_wrapper is None or manifold_wrapper.data is None:
            raise ValueError("MomentRestriction must define a manifold to run GMM.")
        problem = Problem(cost=cost, manifold=manifold_wrapper.data)
        init_kwargs, run_kwargs = self._split_optimizer_kwargs(
            dict(optimizer_kwargs)
        )
        optimizer = self._resolve_optimizer(init_kwargs)
        start_value = (
            initial_point.value
            if isinstance(initial_point, ManifoldPoint)
            else initial_point
        )
        result = optimizer.run(problem, initial_point=start_value, **run_kwargs)
        theta_value = result.point
        theta_point = ManifoldPoint(
            manifold_wrapper,
            theta_value,
        )
        g_bar_hat = self._restriction.g_bar(theta_point)
        optimizer_report = {
            "iterations": getattr(result, "iterations", None),
            "converged": getattr(result, "converged", None),
            "stopping_reason": getattr(result, "stopping_reason", None),
        }
        return _StageResult(
            theta=theta_point,
            g_bar=g_bar_hat,
            weighting=weighting,
            optimizer_report=optimizer_report,
        )

    def _theta_distance(
        self, prev_point: ManifoldPoint, cur_point: ManifoldPoint
    ) -> float:
        """Distance between two estimates for the iterated-GMM convergence test.

        Prefers the manifold's intrinsic ``dist(x, y)``; falls back to the
        ambient L2 norm of flattened coordinates when no distance is
        available (or when it raises -- e.g., custom manifolds without a
        Riemannian metric).
        """

        manifold_wrapper = self._restriction.manifold
        manifold_data = (
            getattr(manifold_wrapper, "data", None)
            if manifold_wrapper is not None
            else None
        )
        if manifold_data is not None:
            dist_fn = getattr(manifold_data, "dist", None)
            if callable(dist_fn):
                try:
                    return float(dist_fn(prev_point.value, cur_point.value))
                except Exception:  # pragma: no cover - manifold-specific fallback
                    pass

        flat_prev = np.asarray(
            self._restriction._array_adapter(prev_point.value), dtype=float
        ).reshape(-1)
        flat_cur = np.asarray(
            self._restriction._array_adapter(cur_point.value), dtype=float
        ).reshape(-1)
        return float(np.linalg.norm(flat_cur - flat_prev))

    def _default_initial_point(self) -> Any | None:
        restriction = self._restriction
        manifold_wrapper = restriction.manifold
        if manifold_wrapper is not None:
            try:
                return manifold_wrapper.random_point()
            except AttributeError:
                pass

        param_shape = restriction.parameter_shape
        param_dim = restriction.parameter_dimension
        if param_shape is not None:
            rng = np.random.default_rng()
            noise = rng.normal(scale=1e-3, size=int(np.prod(param_shape)))
            return noise.reshape(param_shape)
        if param_dim is not None:
            rng = np.random.default_rng()
            return rng.normal(scale=1e-3, size=param_dim)
        return None

    def _build_cost(self, weighting: WeightingStrategy) -> Callable[[Any], Any]:
        restriction = self._restriction
        manifold_wrapper: Manifold | None = restriction.manifold
        if manifold_wrapper is None or manifold_wrapper.data is None:
            raise ValueError(
                "MomentRestriction must carry a manifold for optimisation."
            )

        def _assemble_theta(blocks: tuple[Any, ...]) -> Any:
            if len(blocks) == 1:
                return blocks[0]
            return blocks

        if restriction._is_jax_backend:

            @pymanopt_jax_function(manifold_wrapper.data)
            def cost(*blocks: Any) -> Any:
                theta = _assemble_theta(blocks)
                return self._backend_dot(theta, weighting)

        else:

            @pymanopt_numpy_function(manifold_wrapper.data)
            def cost(*blocks: Any) -> Any:
                theta = _assemble_theta(blocks)
                return float(self._backend_dot(theta, weighting))

        return cost

    def _backend_dot(self, theta: Any, weighting: WeightingStrategy) -> Any:
        xp, _ = self._backend_modules()
        g_vec = self._to_backend_vector(self._restriction.g_bar(theta))
        W = self._to_backend_matrix(weighting.matrix(theta))
        return xp.dot(g_vec, xp.matmul(W, g_vec))

    def _to_backend_vector(self, value: Any) -> Any:
        xp, _ = self._backend_modules()
        array: Any
        if isinstance(value, jnp.ndarray):
            array = value
        else:
            array = np.asarray(value, dtype=float)
            if xp is not np:
                array = xp.asarray(array)
        array = array.reshape(-1)
        return array

    def _to_backend_matrix(self, value: Any) -> Any:
        xp, _ = self._backend_modules()
        array: Any
        if isinstance(value, jnp.ndarray):
            array = value
        else:
            array = np.asarray(value, dtype=float)
            if xp is not np:
                array = xp.asarray(array)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def _ensure_metadata(self, theta: Any) -> int:
        g_vec = self._to_backend_vector(self._restriction.g_bar(theta))
        return int(np.asarray(g_vec).size)

    def _degrees_of_freedom(self, g_bar_value: Any, theta: Any) -> int:
        num_moments = np.asarray(g_bar_value).reshape(-1).size
        param_dim = self._restriction.parameter_dimension
        if param_dim is None:
            manifold_wrapper = self._restriction.manifold
            manifold_dim = None
            if manifold_wrapper is not None and manifold_wrapper.data is not None:
                manifold_dim = getattr(manifold_wrapper.data, "dim", None)
                if callable(manifold_dim):
                    manifold_dim = manifold_dim()
            if manifold_dim is None:
                theta_sample = theta if theta is not None else self._initial_point
                if theta_sample is None:
                    raise RuntimeError(
                        "MomentRestriction has unknown parameter dimension; provide an initial point."
                    )
                if isinstance(theta_sample, tuple | list):
                    manifold_dim = sum(
                        int(np.asarray(block).size) for block in theta_sample
                    )
                else:
                    manifold_dim = int(np.asarray(theta_sample).size)
            param_dim = int(manifold_dim)
        return max(num_moments - param_dim, 0)

    def _backend_modules(self) -> tuple[Any, Any]:
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        return xp, linalg


@dataclass
class _StageResult:
    theta: ManifoldPoint
    g_bar: Any
    weighting: WeightingStrategy
    optimizer_report: Mapping[str, Any]
