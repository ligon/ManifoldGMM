"""High-level GMM estimator built on top of :class:`MomentRestriction`."""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

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
    """Continuously updated weighting based on Ω̂(θ)⁻¹."""

    def __init__(self, restriction: MomentRestriction, ridge: float = 0.0) -> None:
        self._restriction = restriction
        self._ridge = ridge

    def matrix(self, theta: Any) -> Any:
        omega = self._restriction.omega_hat(theta)
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        omega_array = xp.asarray(omega)

        if self._ridge > 0.0:
            omega_array = omega_array + self._ridge * xp.eye(
                omega_array.shape[0], dtype=omega_array.dtype
            )

        return linalg.inv(omega_array)

    def info(self) -> Mapping[str, Any]:
        return {"type": "cue", "ridge": self._ridge}


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
        except (pickle.PicklingError, TypeError):
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
        basis_vectors = (
            basis if basis is not None else restriction.tangent_basis(theta_hat)
        )
        jac_matrix = restriction.jacobian_matrix(theta_hat, basis=basis_vectors)
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
        
        # Scale by 1/N to get estimator variance
        if restriction.num_observations is not None:
            covariance = covariance / restriction.num_observations
            
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
                    return type(lhs)(_add_structure(l, r) for l, r in zip(lhs, rhs))
                return np.asarray(lhs) + np.asarray(rhs)

            def composed_map_numpy(xi: np.ndarray) -> Any:
                tangent_vector = None
                for i, b in enumerate(basis):
                    term = _scale_structure(b, float(xi[i]))
                    if tangent_vector is None:
                        tangent_vector = term
                    else:
                        tangent_vector = _add_structure(tangent_vector, term)

                retraction_fn = getattr(manifold.data, "retraction", None)
                if retraction_fn is None:
                    retraction_fn = getattr(manifold.data, "retract")

                new_value = retraction_fn(theta_hat.value, tangent_vector)
                new_point = ManifoldPoint(manifold, new_value)
                return constraint(new_point)

            # Central difference
            H_cols = []
            for i in range(dim):
                xi_plus = np.zeros(dim)
                xi_plus[i] = epsilon
                val_plus = np.asarray(composed_map_numpy(xi_plus), dtype=float).flatten()

                xi_minus = np.zeros(dim)
                xi_minus[i] = -epsilon
                val_minus = np.asarray(composed_map_numpy(xi_minus), dtype=float).flatten()

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
    ) -> None:
        self._restriction = restriction
        self._cue_ridge = cue_ridge
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
        optimizer_kwargs: Mapping[str, Any] | None = None,
        verbose: bool | int | None = None,
    ) -> GMMResult:
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

        # Stage 1
        weighting_stage1 = self._weighting
        if two_step:
            num_moments = self._ensure_metadata(theta_start)
            weighting_stage1 = IdentityWeighting(num_moments)

        stage1 = self._run_stage(theta_start, weighting_stage1, optimizer_kwargs)

        final_stage = stage1
        final_weighting = weighting_stage1

        if two_step:
            _, linalg = self._backend_modules()
            omega = self._to_backend_matrix(self._restriction.omega_hat(stage1.theta))
            updated_weighting = FixedWeighting(linalg.inv(omega), label="two_step")
            final_stage = self._run_stage(
                stage1.theta,
                updated_weighting,
                optimizer_kwargs,
            )
            final_weighting = updated_weighting

        df = self._degrees_of_freedom(final_stage.g_bar, final_stage.theta)
        weighting_info = dict(final_weighting.info())
        weighting_info.setdefault("two_step", two_step)

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
            two_step=two_step,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _coerce_weighting(
        self, weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    ) -> WeightingStrategy:
        if weighting is None:
            return CUEWeighting(self._restriction, ridge=self._cue_ridge)
        if hasattr(weighting, "matrix") and callable(weighting.matrix):
            return cast(WeightingStrategy, weighting)
        if callable(weighting):
            return CallableWeighting(weighting)
        return FixedWeighting(weighting)

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
        optimizer = self._resolve_optimizer(optimizer_kwargs)
        start_value = (
            initial_point.value
            if isinstance(initial_point, ManifoldPoint)
            else initial_point
        )
        result = optimizer.run(problem, initial_point=start_value)
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
