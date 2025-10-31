from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:  # Optional dependency
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - JAX not installed
    jnp = None  ## type: ignore[assignment]

from datamat import DataMat, DataVec


def _flatten_numpy(value: Any) -> np.ndarray:
    """
    Convert a DataMat/DataVec/numpy/jax array to a flat numpy vector.
    """

    if isinstance(value, DataMat):
        return np.asarray(value.to_numpy(dtype=float)).ravel()
    if isinstance(value, DataVec):
        return np.asarray(value.to_numpy(dtype=float)).ravel()
    return np.asarray(value, dtype=float).ravel()


def assert_moment_translation_equivalent(
    gi_datamat: Callable[[Any, DataVec], Any],
    gi_jax: Callable[[Any, Any], Any],
    *,
    theta_datamat: Any,
    theta_jax: Any,
    observation_datamat: DataVec,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> None:
    """
    Assert that ``gi`` and its JAX twin return the same stacked moments.

    Parameters
    ----------
    gi_datamat:
        Observation-level moment function expressed with DataMat/DataVec
        operators.
    gi_jax:
        JAX-compatible twin returning the same stacked moments.
    theta_datamat:
        Parameter object to feed into ``gi_datamat``.
    theta_jax:
        Parameter object (tuple/array) to feed into ``gi_jax``.
    observation_datamat:
        Single observation represented as a :class:`DataVec`.
    rtol, atol:
        Relative and absolute tolerances for the comparison.
    """

    if jnp is None:  # pragma: no cover - depends on optional JAX
        raise RuntimeError(
            "assert_moment_translation_equivalent requires JAX. "
            "Install the 'jax' extra or invoke poetry install --with dev."
        )

    datamat_values = gi_datamat(theta_datamat, observation_datamat)
    observation_array = jnp.asarray(observation_datamat.to_numpy(dtype=float))
    jax_values = gi_jax(theta_jax, observation_array)

    dm_np = _flatten_numpy(datamat_values)
    jax_np = _flatten_numpy(jax_values)

    np.testing.assert_allclose(jax_np, dm_np, rtol=rtol, atol=atol)
