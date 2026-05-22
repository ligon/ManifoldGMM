"""TrustRegions subclass that surfaces inner-CG state to the per-iteration log.

Pymanopt's :class:`~pymanopt.optimizers.trust_regions.TrustRegions` prints
the inner truncated-CG (tCG) stop reason at ``verbosity >= 2`` but never
persists it via ``_add_log_entry``.  The result is that the well-known
stuck-optimizer pathology -- outer iterations accept steps while every
inner tCG hits ``MAX_INNER_ITER`` -- is invisible to programmatic
diagnostics on :class:`~pymanopt.optimizers.optimizer.OptimizerResult`.

This subclass overrides two hooks on the base optimizer:

- :meth:`_truncated_conjugate_gradient`: stash ``(num_inner, stop_code)``
  for the surrounding outer iteration to read.
- :meth:`_check_stopping_criterion`: at the existing per-iteration call
  site, log the stashed inner state alongside the outer gradient norm via
  the existing ``_add_log_entry`` machinery.

No method bodies are duplicated; the upstream control flow is unchanged.
When ``log_verbosity == 0`` the overrides are no-ops -- pymanopt's
``_add_log_entry`` short-circuits in that mode -- so the cost over plain
``TrustRegions`` is two integer attribute assignments per outer iteration.

See:

- ManifoldGMM issue #10 for the surrounding observability proposal.
- pymanopt/pymanopt#302 for the upstream patch that would let us retire
  this subclass.
"""

from __future__ import annotations

import numpy as np
from pymanopt.optimizers import TrustRegions


class LoggingTrustRegions(TrustRegions):
    """Drop-in replacement for ``TrustRegions`` that logs inner-CG state.

    At ``log_verbosity >= 1`` the per-iteration log
    (``result.log["iterations"]``) gains four extra keys alongside the
    base optimizer's ``time``/``iteration``/``point``/``cost`` lists:

    ``num_inner``
        The integer iteration count returned by the inner truncated-CG.
    ``maxinner``
        The cap on inner iterations in effect for this outer step
        (constant across a run; logged per-row so the cap-hit predicate
        ``num_inner == maxinner`` is self-contained).
    ``inner_stop_code``
        One of the integer stop-code constants defined on
        :class:`TrustRegions`
        (``NEGATIVE_CURVATURE``, ``EXCEEDED_TR``,
        ``REACHED_TARGET_LINEAR``, ``REACHED_TARGET_SUPERLINEAR``,
        ``MAX_INNER_ITER``, ``MODEL_INCREASED``).  Note: pymanopt
        pre-assumes ``MAX_INNER_ITER`` and only overwrites on early
        termination, so the stop code alone can false-positive when
        ``num_inner == 0`` (gradient zero, inner loop body never ran);
        use ``num_inner == maxinner`` for the cap-hit predicate.
    ``gradient_norm``
        The outer Riemannian gradient norm at this iteration.

    ``point`` and ``cost`` are logged as ``None`` -- they live in scope
    only inside the outer ``run`` loop, which we do not duplicate.  The
    downstream metrics on :class:`~manifoldgmm.econometrics.gmm.GMMResult`
    do not consume them.
    """

    def _truncated_conjugate_gradient(
        self,
        problem,
        x,
        fgradx,
        eta,
        Delta,
        theta,
        kappa,
        mininner,
        maxinner,
    ):
        eta, Heta, num_inner, stop_code = super()._truncated_conjugate_gradient(
            problem, x, fgradx, eta, Delta, theta, kappa, mininner, maxinner
        )
        # Stashed for the immediately-following ``_check_stopping_criterion``
        # call to read.  ``maxinner`` is constant across a run but we stash
        # it alongside the per-iteration ``num_inner`` so the per-row record
        # is self-contained (a consumer reading a single row of the log can
        # decide cap-hit without joining against a separate scalar).
        self._last_inner_num_inner = int(num_inner)
        self._last_inner_stop_code = int(stop_code)
        self._last_inner_maxinner = int(maxinner)
        return eta, Heta, num_inner, stop_code

    def _check_stopping_criterion(
        self,
        *,
        start_time,
        iteration: int = -1,
        gradient_norm: float = np.inf,
        step_size: float = np.inf,
        cost_evaluations: int = -1,
    ):
        reason = super()._check_stopping_criterion(
            start_time=start_time,
            iteration=iteration,
            gradient_norm=gradient_norm,
            step_size=step_size,
            cost_evaluations=cost_evaluations,
        )
        # Append a per-outer-iteration log entry with the inner-CG state.
        # ``_add_log_entry`` short-circuits at ``log_verbosity < 1``, so
        # the overhead in the common (non-logging) case is one attribute
        # access and the call itself.
        #
        # ``iteration > 0`` skips the pre-loop call (before any inner CG
        # has run).  ``self._log is not None`` is the standard guard used
        # by the base class.
        if iteration > 0 and self._log is not None:
            self._add_log_entry(
                iteration=iteration,
                point=None,
                cost=None,
                gradient_norm=float(gradient_norm),
                num_inner=getattr(self, "_last_inner_num_inner", None),
                inner_stop_code=getattr(self, "_last_inner_stop_code", None),
                maxinner=getattr(self, "_last_inner_maxinner", None),
            )
        return reason
