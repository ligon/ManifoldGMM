"""LoggingTrustRegions: TrustRegions with inner-CG and gradient-norm telemetry.

See module docstring of :mod:`manifoldgmm.optimizers` for context.
"""

from __future__ import annotations

import collections
from typing import Any

from pymanopt.optimizers import TrustRegions
from pymanopt.optimizers.optimizer import OptimizerResult


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

    Telemetry overhead is one ``Counter`` increment per outer iteration
    and one ``float`` cast plus list append per gradient call -- both
    negligible relative to the inner-CG / Hessian work pymanopt performs.

    Implementation notes
    --------------------
    The class avoids vendoring the ~270-line outer-loop in
    :meth:`TrustRegions.run`.  Instead it:

    1. Overrides :meth:`_truncated_conjugate_gradient` to tally
       ``stop_inner`` codes (one increment per outer iter).
    2. In :meth:`run`, sets the backing attribute
       ``problem._riemannian_gradient`` to a wrapping closure that
       records the post-call norm.  Pymanopt's property lazy-resolves
       through this attribute, so the substitution is transparent to
       the outer loop's local ``gradient = problem.riemannian_gradient``
       binding.  Restored in ``finally`` so callers that reuse the
       same ``Problem`` see no permanent mutation.

    If pymanopt later adds ``_add_log_entry`` calls inside
    :meth:`TrustRegions.run`, this subclass becomes redundant and can be
    deleted in favour of using ``TrustRegions`` directly with
    ``log_verbosity=1``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._inner_stop_counts: collections.Counter[int] = collections.Counter()
        self._iter_gradient_norms: list[float] = []

    def _reset_telemetry(self) -> None:
        self._inner_stop_counts = collections.Counter()
        self._iter_gradient_norms = []

    def _truncated_conjugate_gradient(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, Any, int, int]:
        result = super()._truncated_conjugate_gradient(*args, **kwargs)
        eta, Heta, numit, stop_inner = result
        self._inner_stop_counts[stop_inner] += 1
        return result

    def run(
        self, problem: Any, *, initial_point: Any = None, **kwargs: Any
    ) -> OptimizerResult:
        self._reset_telemetry()

        manifold = problem.manifold
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
        return result


__all__ = ["LoggingTrustRegions"]
