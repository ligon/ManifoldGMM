"""Vendored / customised optimizer subclasses used by manifoldgmm.

These exist to surface state pymanopt's stock optimizers know about but do
not persist into :class:`~pymanopt.optimizers.optimizer.OptimizerResult`.
Each subclass is a drop-in replacement for its upstream parent: same
constructor signature, same ``run`` interface, additive behaviour only.

See pymanopt/pymanopt#302 for the upstream patches that would let these
subclasses be retired.
"""

from .logging_trust_regions import LoggingTrustRegions

__all__ = ["LoggingTrustRegions"]
