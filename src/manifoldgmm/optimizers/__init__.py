"""Pymanopt optimizer wrappers with extra telemetry for diagnostics.

The default :class:`pymanopt.optimizers.trust_regions.TrustRegions` exposes
only a small slice of its internal state through ``OptimizerResult``: the
final ``point``, ``cost``, ``gradient_norm``, ``iterations``, and the
stopping-criterion string.  Per-iteration history -- the inner-CG stop
codes that drive ``MAX_INNER_ITER`` warnings, the gradient-norm trajectory
that distinguishes stalling from steady progress -- is *printed* at
high verbosity but never written into the ``log`` dict.

:class:`LoggingTrustRegions` is a thin subclass that records both as a
side effect, attaching them to ``result.log`` so downstream code (notably
``GMMResult.optimizer_health``) can compute diagnostics like
``inner_cap_hit_frac`` and ``tail_grad_slope`` without re-running the
optimizer.
"""

from .logging_trust_regions import LoggingTrustRegions

__all__ = ["LoggingTrustRegions"]
