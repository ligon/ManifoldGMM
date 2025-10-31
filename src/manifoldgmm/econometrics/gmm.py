"""High-level GMM estimator built on top of :class:`MomentRestriction`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np

try:  # Optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - JAX not installed
    jnp = None  # type: ignore[assignment]

from pymanopt import Problem
from pymanopt.function import jax as pymanopt_jax_function
from pymanopt.function import numpy as pymanopt_numpy_function
from pymanopt.optimizers import TrustRegions
from pymanopt.optimizers.optimizer import Optimizer

from ..geometry import Manifold
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
        return self._fn(theta)

    def info(self) -> Mapping[str, Any]:
        return {"type": self._label}


class CUEWeighting:
    """Continuously updated weighting based on Ω̂(θ)⁻¹."""

    def __init__(self, restriction: MomentRestriction) -> None:
        self._restriction = restriction

    def matrix(self, theta: Any) -> Any:
        omega = self._restriction.omega_hat(theta)
        xp = getattr(self._restriction, "_xp", np)
        linalg = getattr(self._restriction, "_linalg", np.linalg)
        omega_array = xp.asarray(omega)
        return linalg.inv(omega_array)

    def info(self) -> Mapping[str, Any]:
        return {"type": "cue"}


class IdentityWeighting(FixedWeighting):
    """Identity matrix weighting used for first-stage two-step GMM."""

    def __init__(self, dimension: int) -> None:
        super().__init__(np.eye(dimension, dtype=float), label="identity")


@dataclass
class GMMResult:
    """Container returned by :meth:`GMM.estimate`."""

    theta: Any
    criterion_value: float
    degrees_of_freedom: int
    weighting_info: Mapping[str, Any]
    weighting: WeightingStrategy | Callable[[Any], Any] | Any | None
    optimizer_report: Mapping[str, Any]
    restriction: MomentRestriction
    g_bar: Any
    two_step: bool

    def tangent_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> np.ndarray:
        """Return the sandwich covariance in the canonical tangent coordinates."""

        theta_hat = self.theta
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
        if ridge != 0.0:
            covariance = np.asarray(covariance)  # ensure materialised array
        return covariance

    def manifold_covariance(
        self,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        ridge_condition: float = 1e8,
        basis: list[Any] | None = None,
    ) -> np.ndarray:
        """Push forward the tangent covariance to ambient coordinates."""

        restriction = self.restriction
        basis_vectors = (
            basis if basis is not None else restriction.tangent_basis(self.theta)
        )
        cov_tangent = self.tangent_covariance(
            weighting=weighting, ridge_condition=ridge_condition, basis=basis_vectors
        )

        columns: list[np.ndarray] = []
        for direction in basis_vectors:
            flat = restriction._array_adapter(direction)
            columns.append(np.asarray(flat, dtype=float).reshape(-1))

        if not columns:
            return np.zeros((0, 0), dtype=float)

        chart_jacobian = np.column_stack(columns)
        return chart_jacobian @ cov_tangent @ chart_jacobian.T

    def as_dict(self) -> Mapping[str, Any]:
        """Return the result as a dictionary for quick inspection."""

        return {
            "theta": self.theta,
            "criterion_value": self.criterion_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "weighting": dict(self.weighting_info),
            "optimizer_report": dict(self.optimizer_report),
            "two_step": self.two_step,
        }


class GMM:
    """High-level GMM estimator operating on a :class:`MomentRestriction`."""

    def __init__(
        self,
        restriction: MomentRestriction,
        *,
        weighting: WeightingStrategy | Callable[[Any], Any] | Any | None = None,
        optimizer: type[Optimizer] | Optimizer | None = None,
        initial_point: Any | None = None,
    ) -> None:
        self._restriction = restriction
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
    ) -> GMMResult:
        theta_start = (
            initial_point if initial_point is not None else self._initial_point
        )
        if theta_start is None:
            raise ValueError("Provide an initial_point to start the optimisation.")

        optimizer_kwargs = dict(optimizer_kwargs or {})

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
            theta=final_stage.theta,
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
            return CUEWeighting(self._restriction)
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
                raise ValueError(
                    "optimizer_kwargs are incompatible with a pre-configured optimizer"
                )
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
        result = optimizer.run(problem, initial_point=initial_point)
        theta_hat = result.point
        g_bar_hat = self._restriction.g_bar(theta_hat)
        optimizer_report = {
            "iterations": getattr(result, "iterations", None),
            "converged": getattr(result, "converged", None),
            "stopping_reason": getattr(result, "stopping_reason", None),
        }
        return _StageResult(
            theta=theta_hat,
            g_bar=g_bar_hat,
            weighting=weighting,
            optimizer_report=optimizer_report,
        )

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
        if jnp is not None and isinstance(value, jnp.ndarray):
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
        if jnp is not None and isinstance(value, jnp.ndarray):
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
    theta: Any
    g_bar: Any
    weighting: WeightingStrategy
    optimizer_report: Mapping[str, Any]
