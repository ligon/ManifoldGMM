"""Process-wide JAX configuration applied at package import time.

Importing this module enables 64-bit precision in JAX.  The GMM objective
``g' W g`` and its Hessian are conditioning-sensitive; running JAX in its
default float32 mode silently degrades the inner truncated-CG and the
Wald-test covariances.  pymanopt's JAX backend currently enables x64 on
import as well, but we set it here so the requirement is owned by this
package and survives upstream changes.
"""

import jax

jax.config.update("jax_enable_x64", True)
