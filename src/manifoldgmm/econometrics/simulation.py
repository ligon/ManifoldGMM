"""Generic Monte Carlo simulation runner with reproducible parallel streams.

This module provides a thin dispatcher that runs a user-supplied replication
function many times, each with an independent random stream derived from
:class:`numpy.random.SeedSequence`, and optionally parallelises the work
with :mod:`joblib`.

Key function
------------
monte_carlo
    Run *n_reps* independent replications and collect the results as a list
    of dictionaries (trivially convertible to a :class:`pandas.DataFrame`).

Design notes
------------
- **Reproducibility.** Each replication receives a :class:`numpy.random.Generator`
  seeded from ``SeedSequence(seed).spawn(n_reps)``, which guarantees
  statistically independent streams regardless of execution order or
  parallelism level.  This is superior to the common ``seed + i`` pattern,
  which can produce correlated streams when seeds are close together
  (L'Ecuyer, 2017).
- **Fault tolerance.** If a single replication raises, its exception is
  captured as ``{"error": str(e)}`` rather than aborting the entire run.
  This is important for manifold optimisation where individual solves may
  fail to converge.
- **Parallelism.** ``n_jobs=1`` runs a plain serial loop (convenient for
  debugging and profiling); ``n_jobs>1`` or ``n_jobs=-1`` dispatches via
  ``joblib.Parallel``.

References
----------
L'Ecuyer, P., Simard, R., Chen, E.J. & Kelton, W.D. (2002). An
object-oriented random-number package with many long streams and substreams.
*Operations Research*, 50(6), 1073--1075.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import numpy as np


def monte_carlo(
    replication_fn: Callable[[int, np.random.Generator], dict],
    n_reps: int,
    *,
    seed: int = 0,
    n_jobs: int = 1,
    progress: bool = True,
) -> list[dict]:
    r"""Run a Monte Carlo simulation with reproducible parallel streams.

    Parameters
    ----------
    replication_fn : callable
        Signature ``(rep_index: int, rng: numpy.random.Generator) -> dict``.
        Each call should return a dictionary of scalar results for that
        replication.
    n_reps : int
        Number of independent replications.
    seed : int, default 0
        Base seed for :class:`numpy.random.SeedSequence`.
    n_jobs : int, default 1
        Number of parallel workers.  ``1`` runs serially; ``-1`` uses all
        available cores; ``-2`` reserves one core for the OS.
    progress : bool, default True
        If ``True``, print a progress indicator to stderr.

    Returns
    -------
    list of dict
        One dictionary per replication.  Failed replications contain an
        ``"error"`` key with the exception message.

    Examples
    --------
    >>> def coin_flip(rep, rng):
    ...     return {"heads": int(rng.integers(0, 2))}
    >>> results = monte_carlo(coin_flip, 10, seed=42, progress=False)
    >>> len(results)
    10
    """

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_reps)

    def _run_one(idx: int, child_seed: np.random.SeedSequence) -> dict:
        rng = np.random.default_rng(child_seed)
        try:
            return replication_fn(idx, rng)
        except Exception as exc:
            return {"error": str(exc)}

    if n_jobs == 1:
        # Serial loop
        records: list[dict] = []
        for i, cs in enumerate(child_seeds):
            records.append(_run_one(i, cs))
            if progress and (i + 1) % max(1, n_reps // 10) == 0:
                print(
                    f"  [{i + 1}/{n_reps}]",
                    file=sys.stderr,
                    flush=True,
                )
        return records

    # Parallel via joblib
    from joblib import Parallel, delayed

    verbose = 5 if progress else 0
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_run_one)(i, cs) for i, cs in enumerate(child_seeds)
    )
    return list(results)


__all__ = ["monte_carlo"]
