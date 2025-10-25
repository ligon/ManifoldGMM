from __future__ import annotations

import numpy as np
import pytest

from manifoldgmm.geometry import Manifold, ManifoldPoint

from pymanopt.manifolds.euclidean import Euclidean
from pymanopt.manifolds.psd import PSDFixedRank
from pymanopt.manifolds.stiefel import Stiefel


def _stiefel_project(point: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(point)
    return q[:, : point.shape[1]]


def _psd_project(point: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(point, full_matrices=False)
    rank = point.shape[1]
    return u[:, :rank] @ np.diag(s[:rank])


@pytest.fixture(params=["euclidean", "stiefel", "psd"])
def manifold_fixture(request):
    if request.param == "euclidean":
        manifold = Euclidean(5)
        project = None
        shape = (5,)
    elif request.param == "stiefel":
        n, k = 6, 3
        manifold = Stiefel(n, k)
        project = _stiefel_project
        shape = (n, k)
    elif request.param == "psd":
        m, r = 5, 2
        manifold = PSDFixedRank(m, r)
        project = _psd_project
        shape = (m, r)
    else:  # pragma: no cover
        raise RuntimeError("Unsupported manifold fixture")
    return manifold, project, shape


def test_construct_from_on_manifold_point(manifold_fixture):
    manifold, _, _ = manifold_fixture
    wrapper = Manifold.from_pymanopt(manifold)
    value = manifold.random_point()
    point = ManifoldPoint(wrapper, value)
    assert np.allclose(point.value, value)


def test_construct_from_off_manifold_point_projects(manifold_fixture):
    manifold, project_point, _ = manifold_fixture
    if project_point is None:
        pytest.skip("Projection is trivial for Euclidean manifold")
    wrapper = Manifold.from_pymanopt(manifold, project_point=project_point)
    on_manifold = manifold.random_point()
    off_manifold = on_manifold + 0.05 * np.random.randn(*on_manifold.shape)
    point = ManifoldPoint(wrapper, off_manifold)
    if isinstance(manifold, Stiefel):
        gram = point.value.T @ point.value
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6)
    elif isinstance(manifold, PSDFixedRank):
        rank = np.linalg.matrix_rank(point.value, tol=1e-8)
        assert rank == point.value.shape[1]


def test_shape_and_dtype_preserved(manifold_fixture):
    manifold, project_point, expected_shape = manifold_fixture
    wrapper = Manifold.from_pymanopt(manifold, project_point=project_point)
    value = manifold.random_point()
    point = ManifoldPoint(wrapper, value)
    assert point.value.shape == expected_shape
    assert point.value.dtype == np.float64


def test_project_tangent_idempotent(manifold_fixture):
    manifold, project_point, _ = manifold_fixture
    wrapper = Manifold.from_pymanopt(manifold, project_point=project_point)
    value = manifold.random_point()
    point = ManifoldPoint(wrapper, value)
    ambient = np.random.randn(*point.value.shape)
    tangent_once = point.project_tangent(ambient)
    tangent_twice = point.project_tangent(tangent_once)
    assert np.allclose(tangent_once, tangent_twice)


def test_stiefel_tangent_condition():
    manifold = Stiefel(5, 3)
    wrapper = Manifold.from_pymanopt(manifold, project_point=_stiefel_project)
    value = manifold.random_point()
    point = ManifoldPoint(wrapper, value)
    ambient = np.random.randn(*value.shape)
    tangent = point.project_tangent(ambient)
    skew = point.value.T @ tangent + tangent.T @ point.value
    assert np.allclose(skew, np.zeros_like(skew), atol=1e-8)


def test_with_value_creates_new_point():
    manifold = Euclidean(4)
    wrapper = Manifold.from_pymanopt(manifold)
    initial = ManifoldPoint(wrapper, np.ones(4))
    updated = initial.with_value(np.zeros(4))
    assert np.allclose(initial.value, np.ones(4))
    assert np.allclose(updated.value, np.zeros(4))
