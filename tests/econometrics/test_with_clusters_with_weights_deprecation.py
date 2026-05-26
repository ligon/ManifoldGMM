"""Issue #47: ``MomentRestriction.with_clusters`` / ``with_weights`` deprecation.

Asserts that public calls to ``with_clusters`` and ``with_weights`` emit
:class:`DeprecationWarning` pointing users at the v2 DGP-side sampling-
design state (``EmpiricalDGP(sampling=...)``).  Library-internal callers
use the private ``_set_clusters`` / ``_set_weights`` mutators (which do
*not* warn) so the v1 moment-error bootstrap does not generate noise on
its own internal cloning.

Companion to the larger issue-#47 cleanup; the deprecated methods are
scheduled for removal in v0.5.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from manifoldgmm.econometrics.moment_restriction import MomentRestriction


def _make_restriction() -> MomentRestriction:
    data = jnp.asarray(np.arange(12).reshape(6, 2).astype(float))

    def g(theta, x):
        return x - theta[None, :]

    return MomentRestriction(g=g, data=data, backend="numpy")


def test_with_clusters_emits_deprecation_warning() -> None:
    """``with_clusters`` raises a ``DeprecationWarning`` pointing at v2."""

    restriction = _make_restriction()
    cluster_ids = np.array([0, 0, 0, 1, 1, 1])
    with pytest.warns(DeprecationWarning, match="ClusteredSampling"):
        restriction.with_clusters(cluster_ids)


def test_with_weights_emits_deprecation_warning() -> None:
    """``with_weights`` raises a ``DeprecationWarning`` pointing at v2."""

    restriction = _make_restriction()
    weights = np.ones(6)
    with pytest.warns(DeprecationWarning, match="IIDSampling"):
        restriction.with_weights(weights)


def test_with_clusters_still_works_under_warning() -> None:
    """The deprecated path still functions (warns + applies)."""

    restriction = _make_restriction()
    cluster_ids = np.array([0, 0, 0, 1, 1, 1])
    with pytest.warns(DeprecationWarning):
        clone = restriction.with_clusters(cluster_ids)
    np.testing.assert_array_equal(np.asarray(clone.clusters), cluster_ids)


def test_with_weights_still_works_under_warning() -> None:
    """The deprecated path still functions (warns + applies)."""

    restriction = _make_restriction()
    weights = np.array([1.5, 0.5, 2.0, 0.0, 1.0, 1.0])
    with pytest.warns(DeprecationWarning):
        clone = restriction.with_weights(weights)
    np.testing.assert_array_equal(np.asarray(clone.weights), weights)


def test_set_clusters_internal_does_not_warn() -> None:
    """The private ``_set_clusters`` mutator is the no-warn library internal."""

    restriction = _make_restriction()
    cluster_ids = np.array([0, 0, 0, 1, 1, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        clone = restriction._set_clusters(cluster_ids)  # would re-raise if warned
    np.testing.assert_array_equal(np.asarray(clone.clusters), cluster_ids)


def test_set_weights_internal_does_not_warn() -> None:
    """The private ``_set_weights`` mutator is the no-warn library internal."""

    restriction = _make_restriction()
    weights = np.array([1.5, 0.5, 2.0, 0.0, 1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        clone = restriction._set_weights(weights)  # would re-raise if warned
    np.testing.assert_array_equal(np.asarray(clone.weights), weights)


def test_warning_points_at_v2_dgp_path() -> None:
    """The deprecation message names the v2 EmpiricalDGP construct."""

    restriction = _make_restriction()
    with pytest.warns(DeprecationWarning) as record:
        restriction.with_clusters(np.zeros(6, dtype=int))
    assert any("EmpiricalDGP" in str(w.message) for w in record)
    assert any("issue #47" in str(w.message) for w in record)
