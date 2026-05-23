"""Regression test: importing ``manifoldgmm`` enables JAX float64.

The GMM objective ``g' W g`` and its Hessian are conditioning-sensitive;
running JAX in its default float32 mode silently degrades the inner
truncated-CG and the Wald-test covariances.  Keep this guarantee local
to our package so future upstream changes in pymanopt cannot quietly
turn it off.
"""

import subprocess
import sys


def test_import_manifoldgmm_enables_x64() -> None:
    code = (
        "import manifoldgmm; "
        "import jax; "
        "import jax.numpy as jnp; "
        "assert jax.config.read('jax_enable_x64'), 'x64 disabled'; "
        "assert jnp.zeros(1).dtype == jnp.float64, jnp.zeros(1).dtype"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
