from __future__ import annotations

import jax.numpy as jnp
import pandas as pd
from datamat import DataVec

from ..utils.datamat_to_jax import assert_moment_translation_equivalent


def _make_theta() -> DataVec:
    return DataVec(
        [0.3, -0.2],
        index=pd.Index(["x1", "x2"], name="var"),
    )


def gi_datamat(theta: DataVec, x: DataVec) -> DataVec:
    resid = x - theta
    return DataVec(
        [resid.iloc[0], resid.iloc[1], resid.iloc[0] * resid.iloc[1]],
        index=pd.Index(["m1", "m2", "m12"], name="moment"),
    )


def gi_jax(theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    resid = x - theta
    return jnp.array([resid[0], resid[1], resid[0] * resid[1]])


def test_moment_translation_helper_round_trip() -> None:
    theta_dm = _make_theta()
    obs_dm = DataVec(
        [0.15, -0.05],
        index=theta_dm.index.copy(),
    )

    theta_jax = jnp.asarray(theta_dm.to_numpy(dtype=float))

    assert_moment_translation_equivalent(
        gi_datamat=gi_datamat,
        gi_jax=gi_jax,
        theta_datamat=theta_dm,
        theta_jax=theta_jax,
        observation_datamat=obs_dm,
    )
