"""LoggingTrustRegions: TrustRegions with inner-CG and gradient-norm telemetry.

See module docstring of :mod:`manifoldgmm.optimizers` for context.
"""

from __future__ import annotations

import collections
from typing import Any

from pymanopt.optimizers import TrustRegions
from pymanopt.optimizers.optimizer import OptimizerResult

# Defaults for the opt-in adaptive-maxinner policy (#10 PR 3).  Constants
# rather than buried magic numbers so tests can patch and downstream can
# read the chosen tuning.
_DEFAULT_ADAPTIVE_THRESHOLD: float = 0.6
_DEFAULT_ADAPTIVE_WINDOW: int = 5
_DEFAULT_ADAPTIVE_CEILING_FACTOR: int = 8


class LoggingTrustRegions(TrustRegions):
    """Drop-in TrustRegions replacement that captures per-iter telemetry.

    Two pieces of diagnostic state are written into ``result.log`` (which
    rides through to ``GMMResult.optimizer_report["log"]``):

    - ``inner_stop_counts``: a ``dict[str, int]`` keyed by the
      inner-CG stop-reason string (``"maximum inner iterations"``,
      ``"exceeded trust region"``, etc.), counting how many outer
      iterations terminated their inner solve via each reason.  Suitable
      for computing ``inner_cap_hit_frac``.
    - ``gradient_norms``: a ``list[float]`` of Riemannian gradient norms,
      one per call to the gradient function during optimisation.  Under
      ``TrustRegions``' acceptance logic this corresponds to one entry at
      initialisation plus one per *accepted* outer iteration; rejected
      steps reuse the previous gradient.

    When ``adaptive_maxinner=True`` (opt-in; see :meth:`__init__`), two
    additional log keys are populated:

    - ``maxinner_history``: a ``list[int]`` of the ``maxinner`` value
      passed to each inner truncated-CG call.  Constant when the policy
      stays quiet; ratchets up when the rolling cap-hit window crosses
      the threshold.
    - ``numit_history``: a ``list[int]`` of the actual inner iteration
      counts.  Lets downstream code compute the *strict*
      ``numit == maxinner`` cap-hit predicate, which avoids pymanopt's
      false-positive where ``stop_code == MAX_INNER_ITER`` is
      pre-assumed and stays put when the inner loop body never runs
      (e.g., gradient already zero, ``numit == 0``).

    Telemetry overhead is one ``Counter`` increment per outer iteration
    and one ``float`` cast plus list append per gradient call -- both
    negligible relative to the inner-CG / Hessian work pymanopt performs.

    Implementation notes
    --------------------
    The class avoids vendoring the ~270-line outer-loop in
    :meth:`TrustRegions.run`.  Instead it:

    1. Overrides :meth:`_truncated_conjugate_gradient` to tally
       ``stop_inner`` codes (one increment per outer iter), and, when
       adaptive mode is active, substitutes its own ``maxinner`` for
       the local value pymanopt's ``run`` passed in.
    2. In :meth:`run`, sets the backing attribute
       ``problem._riemannian_gradient`` to a wrapping closure that
       records the post-call norm.  Pymanopt's property lazy-resolves
       through this attribute, so the substitution is transparent to
       the outer loop's local ``gradient = problem.riemannian_gradient``
       binding.  Restored in ``finally`` so callers that reuse the
       same ``Problem`` see no permanent mutation.

    If pymanopt later adds ``_add_log_entry`` calls inside
    :meth:`TrustRegions.run`, the telemetry parts of this subclass
    become redundant and can be deleted in favour of using
    ``TrustRegions`` directly with ``log_verbosity=1``.  The adaptive-
    maxinner policy would still need a subclass because pymanopt's
    ``run`` reads ``maxinner`` from a local, not an attribute.
    """

    def __init__(
        self,
        *args: Any,
        adaptive_maxinner: bool = False,
        adaptive_threshold: float = _DEFAULT_ADAPTIVE_THRESHOLD,
        adaptive_window: int = _DEFAULT_ADAPTIVE_WINDOW,
        adaptive_ceiling: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Construct a logging TrustRegions optimizer.

        Parameters
        ----------
        adaptive_maxinner:
            Opt-in to the adaptive-maxinner policy (#10 PR 3).  When
            enabled, the optimizer doubles ``maxinner`` whenever a
            rolling window of recent outer iterations crossed the
            ``adaptive_threshold`` fraction of cap-hits.  The current
            ``maxinner`` value is exposed per-iteration via
            ``result.log["maxinner_history"]``.
        adaptive_threshold:
            Fraction of the rolling window that must hit the cap to
            trigger a doubling.  Default ``0.6``.  The cap-hit predicate
            is the strict ``numit >= current_maxinner``, not the
            stop-code string-match -- avoids pymanopt's false-positive
            when ``numit == 0`` (see class docstring).
        adaptive_window:
            Size of the rolling window used to compute the cap-hit
            fraction.  Default ``5``.
        adaptive_ceiling:
            Upper bound on ``maxinner``.  ``None`` (default) resolves at
            ``run()`` to ``8 * starting_maxinner`` (i.e., ``8 *
            manifold.dim`` when ``maxinner`` is not explicitly passed),
            which allows three doublings before saturating.  Pass an
            explicit integer to override; pass ``float('inf')`` to
            disable the bound entirely (use with caution -- pathological
            problems can push ``maxinner`` indefinitely).
        *args, **kwargs:
            Forwarded to :class:`TrustRegions.__init__`.
        """

        super().__init__(*args, **kwargs)
        self._inner_stop_counts: collections.Counter[int] = collections.Counter()
        self._iter_gradient_norms: list[float] = []
        # Adaptive policy state.
        self._adaptive_maxinner: bool = bool(adaptive_maxinner)
        self._adaptive_threshold: float = float(adaptive_threshold)
        self._adaptive_window: int = int(adaptive_window)
        # Note ``None`` here distinguishes "use the 8x default" from
        # ``float('inf')`` ("explicitly no ceiling").  Resolved at run().
        self._adaptive_ceiling: int | float | None = adaptive_ceiling
        self._current_maxinner: int | None = None
        self._maxinner_history: list[int] = []
        self._numit_history: list[int] = []

    def _reset_telemetry(self) -> None:
        self._inner_stop_counts = collections.Counter()
        self._iter_gradient_norms = []
        self._current_maxinner = None
        self._maxinner_history = []
        self._numit_history = []

    def _truncated_conjugate_gradient(
        self,
        problem: Any,
        x: Any,
        fgradx: Any,
        eta: Any,
        Delta: Any,
        theta: Any,
        kappa: Any,
        mininner: int,
        maxinner: int,
    ) -> tuple[Any, Any, int, int]:
        # Adaptive override: ignore the local ``maxinner`` from pymanopt's
        # ``run`` (it's a frozen-at-entry local) and use the value the
        # adaptive policy has been ratcheting.  ``_current_maxinner`` is
        # initialised in ``run`` to mirror pymanopt's default so the
        # first call is identical to the non-adaptive case.
        if self._adaptive_maxinner and self._current_maxinner is not None:
            effective = self._current_maxinner
        else:
            effective = int(maxinner)

        self._maxinner_history.append(effective)

        result = super()._truncated_conjugate_gradient(
            problem, x, fgradx, eta, Delta, theta, kappa, mininner, effective
        )
        eta_out, Heta_out, numit, stop_inner = result
        self._inner_stop_counts[stop_inner] += 1
        self._numit_history.append(int(numit))

        if self._adaptive_maxinner:
            self._maybe_double_maxinner(effective)

        return result

    def _maybe_double_maxinner(self, current: int) -> None:
        """Update ``_current_maxinner`` based on the rolling cap-hit window.

        Strict cap-hit predicate: ``numit >= maxinner_used_that_iter``.
        Uses :attr:`_numit_history` and :attr:`_maxinner_history` (both
        ordered, same length).  Closes #10 PR 3.
        """

        recent_numit = self._numit_history[-self._adaptive_window :]
        recent_max = self._maxinner_history[-self._adaptive_window :]
        if len(recent_numit) < self._adaptive_window:
            return
        cap_hits = sum(
            1 for n, m in zip(recent_numit, recent_max, strict=False) if n >= m
        )
        cap_frac = cap_hits / len(recent_numit)
        if cap_frac < self._adaptive_threshold:
            return

        proposed = current * 2
        ceiling = self._adaptive_ceiling
        # ``None`` means "use 8x starting" but starting is whatever
        # run() initialised ``_current_maxinner`` to, which is what
        # ``current`` already reflects when we haven't doubled yet.
        # We need the *initial* maxinner, which we cache as
        # ``self._adaptive_starting_maxinner`` in run().
        if ceiling is None:
            ceiling = (
                _DEFAULT_ADAPTIVE_CEILING_FACTOR * self._adaptive_starting_maxinner
            )
        if proposed > ceiling:
            proposed = int(ceiling) if ceiling != float("inf") else proposed
        if proposed > current:
            self._current_maxinner = int(proposed)

    def run(
        self, problem: Any, *, initial_point: Any = None, **kwargs: Any
    ) -> OptimizerResult:
        self._reset_telemetry()

        manifold = problem.manifold

        # Adaptive maxinner setup: mirror pymanopt's ``maxinner`` resolution
        # so our state starts at the same value the non-adaptive run would
        # use.  Pymanopt's default is ``manifold.dim``; an explicit kwarg
        # overrides.
        if self._adaptive_maxinner:
            passed_maxinner = kwargs.get("maxinner")
            if passed_maxinner is None:
                # Mirror pymanopt's ``if maxinner is None: maxinner = manifold.dim``.
                passed_maxinner = manifold.dim
            self._adaptive_starting_maxinner = int(passed_maxinner)
            self._current_maxinner = int(passed_maxinner)

        # Resolve the gradient function through the property first so the
        # lazy fallback (euclidean->riemannian) runs once and writes to
        # ``_riemannian_gradient``.  After that we wrap it in place.
        original_gradient = problem.riemannian_gradient

        def recording_gradient(point: Any) -> Any:
            g = original_gradient(point)
            self._iter_gradient_norms.append(float(manifold.norm(point, g)))
            return g

        original_backing = getattr(problem, "_riemannian_gradient", None)
        problem._riemannian_gradient = recording_gradient
        try:
            result = super().run(problem, initial_point=initial_point, **kwargs)
        finally:
            # Restore so the same ``Problem`` instance can be reused for
            # later fits without leaking our wrapper.
            problem._riemannian_gradient = original_backing

        # Attach diagnostics to the log dict.  ``result.log`` is always
        # set by ``Optimizer._initialize_log``; ``"iterations"`` may be
        # ``None`` at ``log_verbosity=0`` but the top-level dict exists.
        if result.log is None:
            result.log = {}
        result.log["inner_stop_counts"] = {
            self.TCG_STOP_REASONS[code]: count
            for code, count in self._inner_stop_counts.items()
        }
        result.log["gradient_norms"] = list(self._iter_gradient_norms)
        if self._adaptive_maxinner:
            result.log["maxinner_history"] = list(self._maxinner_history)
            result.log["numit_history"] = list(self._numit_history)
        return result


__all__ = ["LoggingTrustRegions"]
