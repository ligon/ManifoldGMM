"""Canonical warning classes for ManifoldGMM.

The package CLAUDE.md lists a small set of canonical warning categories
(``NumericalWarning``, ``GaugeWarning``, ``ConvergenceWarning``) that
downstream code is expected to use rather than bare ``UserWarning``.
This module is the home for those classes.  Each subclasses
``UserWarning`` so the standard :mod:`warnings` machinery (filters,
``warnings.simplefilter``, pytest's ``filterwarnings`` mark, etc.) works
unchanged, while letting callers target a specific category when they
want to silence (or escalate) just that family of diagnostics.

Adding more classes here is cheap; resist the urge to add one per
warning site.  A category buys filterability for an *audience*: users
who care about convergence behaviour but not about numerical-conditioning
nags, or vice versa.  If a single site is the only consumer, prefer the
existing :class:`UserWarning`.
"""

from __future__ import annotations


class ConvergenceWarning(UserWarning):
    """The optimisation completed but a convergence-health check failed.

    Emitted at the end of :meth:`manifoldgmm.GMM.estimate` when the
    diagnostics in :attr:`manifoldgmm.GMMResult.diagnostics.optimizer_health`
    flag a stall pattern: the outer loop ran out of budget (non-tolerance
    stopping criterion), the inner truncated-CG repeatedly hit its
    iteration cap, and the gradient norm stopped descending.  The
    ``GMMResult`` is still returned -- the warning is advisory.

    Filter via the standard :mod:`warnings` interface, e.g.::

        import warnings
        from manifoldgmm import ConvergenceWarning

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    """


class NumericalWarning(UserWarning):
    """A numerical operation completed in a degraded regime.

    Emitted from low-level numerical helpers when an iterative procedure
    bails out before reaching its target tolerance, returning a
    best-effort result rather than continuing indefinitely.  Issue #18
    is the motivating case: :func:`manifoldgmm.utils.numeric.ridge_inverse`
    used to spin in a ``while True`` ridge-bump loop on severely
    ill-conditioned matrices (``cond >> target_condition``); the loop
    now caps and emits this warning instead.

    Filter via the standard :mod:`warnings` interface, e.g.::

        import warnings
        from manifoldgmm import NumericalWarning

        warnings.filterwarnings("ignore", category=NumericalWarning)
    """


__all__ = ["ConvergenceWarning", "NumericalWarning"]
