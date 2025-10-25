from __future__ import annotations

import numpy as np
import pytest
from collections.abc import Sequence

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from jax import linearize, vjp, jvp

from manifoldgmm.autodiff import jacobian_from_pymanopt
from manifoldgmm.geometry import ManifoldPoint

from pymanopt import function as pymanopt_function
from pymanopt.manifolds.euclidean import Euclidean
from pymanopt.manifolds.product import Product
from pymanopt.manifolds.psd import PSDFixedRank


def _psd_manifold():
    return PSDFixedRank(n=3, k=2)


def _product_manifold():
    return Product([PSDFixedRank(n=3, k=2), Euclidean(2)])


def _is_sequence(obj):
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))


def _map_structure(func, tree):
    if _is_sequence(tree):
        return type(tree)(_map_structure(func, element) for element in tree)
    return func(tree)


def _assert_allclose_structure(actual, expected, *, atol=1e-8, rtol=1e-7):
    if _is_sequence(actual) and _is_sequence(expected):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_allclose_structure(a, b, atol=atol, rtol=rtol)
        return
    assert np.allclose(np.asarray(actual), np.asarray(expected), atol=atol, rtol=rtol)


def _to_jnp(tree):
    return _map_structure(jnp.asarray, tree)


def test_psd_tangent_projection_matches_pymanopt():
    manifold = _psd_manifold()
    theta = ManifoldPoint.from_pymanopt(manifold, manifold.random_point())
    ambient = np.random.randn(*theta.value.shape)
    projected = theta.project_tangent(ambient)
    expected = manifold.projection(theta.value, ambient)
    assert np.allclose(projected, expected)


def test_psd_jacobian_operator_matches_linearize():
    manifold = _psd_manifold()
    theta = ManifoldPoint.from_pymanopt(manifold, manifold.random_point())

    def vector_function_array(y):
        y = jnp.asarray(y)
        gram = y.T @ y
        trace_term = jnp.trace(gram)
        det_term = jnp.linalg.det(gram + jnp.eye(gram.shape[0]))
        return jnp.stack([trace_term, det_term])

    jac = jacobian_from_pymanopt(vector_function_array, manifold, theta.value)

    base = jnp.asarray(theta.value)
    _, jvp_fn = linearize(vector_function_array, base)
    _, vjp_fn = vjp(vector_function_array, base)

    tangent = theta.project_tangent(np.random.randn(*theta.value.shape))
    tangent_jnp = jnp.asarray(tangent)
    expected_matvec = jvp_fn(tangent_jnp)
    result_matvec = jnp.asarray(jac.matvec(tangent))
    assert jnp.allclose(result_matvec, expected_matvec)

    covector = jnp.array([1.5, -0.25])
    expected_tangent = vjp_fn(covector)[0]
    expected_tangent = theta.project_tangent(np.asarray(expected_tangent))
    result_tangent = jac.T_matvec(covector)
    assert jnp.allclose(result_tangent, expected_tangent)


def test_product_tangent_projection_matches_pymanopt():
    manifold = _product_manifold()
    theta = ManifoldPoint.from_pymanopt(manifold, manifold.random_point())
    ambient_components = [
        np.random.randn(*theta.value[0].shape),
        np.random.randn(*theta.value[1].shape),
    ]
    if isinstance(theta.value, tuple):
        ambient = tuple(ambient_components)
    else:
        ambient = ambient_components
    projected = theta.project_tangent(ambient)
    expected = manifold.projection(theta.value, ambient)
    _assert_allclose_structure(projected, expected)


def test_product_jacobian_operator_matches_linearize():
    manifold = _product_manifold()
    theta = ManifoldPoint.from_pymanopt(manifold, manifold.random_point())

    def vector_function_tree(data):
        y, u = data
        y = jnp.asarray(y)
        u = jnp.asarray(u)
        gram = y.T @ y
        trace_term = jnp.trace(gram)
        det_term = jnp.linalg.det(gram + jnp.eye(gram.shape[0]))
        norm_u = jnp.sum(u**2)
        return jnp.stack([trace_term, det_term, norm_u])

    jac = jacobian_from_pymanopt(vector_function_tree, manifold, theta.value)

    ambient_components = [
        np.random.randn(*theta.value[0].shape),
        np.random.randn(*theta.value[1].shape),
    ]
    if isinstance(theta.value, tuple):
        ambient = tuple(ambient_components)
    else:
        ambient = ambient_components

    tangent = theta.project_tangent(ambient)
    base = [jnp.asarray(component) for component in theta.value]
    tangent_jnp = [jnp.asarray(component) for component in tangent]

    _, expected_matvec = jvp(
        vector_function_tree,
        (base,),
        (tangent_jnp,),
    )
    _, vjp_fn = vjp(vector_function_tree, base)
    result_matvec = jnp.asarray(jac.matvec(tangent))
    assert jnp.allclose(result_matvec, expected_matvec)

    covector = jnp.array([0.7, -1.2, 0.5])
    expected_tangent = vjp_fn(covector)[0]
    expected_tangent = theta.project_tangent(
        [np.asarray(component) for component in expected_tangent]
    )
    result_tangent = jac.T_matvec(covector)
    _assert_allclose_structure(result_tangent, expected_tangent)
def test_psd_jacobian_matches_pymanopt_gradient():
    manifold = _psd_manifold()
    theta = manifold.random_point()

    def scalar_function(y):
        y = jnp.asarray(y)
        gram = y.T @ y
        return jnp.trace(gram @ gram)

    cost = pymanopt_function.jax(manifold)(scalar_function)
    gradient_operator = cost.get_gradient_operator()
    gradient = gradient_operator(theta)

    jac = jacobian_from_pymanopt(
        lambda y: jnp.array([scalar_function(y)]),
        manifold,
        theta,
    )

    covector = jnp.array([1.0])
    t_grad = jac.T_matvec(covector)
    assert np.allclose(np.asarray(t_grad), np.asarray(gradient))

    ambient_direction = np.random.randn(*theta.shape)
    tangent = manifold.projection(theta, ambient_direction)
    directional_derivative = jac.matvec(tangent)[0]
    expected_derivative = manifold.inner_product(theta, gradient, tangent)
    assert float(directional_derivative) == pytest.approx(
        expected_derivative, rel=1e-7, abs=1e-9
    )


def test_psd_jacobian_matches_gradient_loop():
    manifold = _psd_manifold()
    theta = manifold.random_point()

    def vector_function(y):
        y = jnp.asarray(y)
        gram = y.T @ y
        return jnp.stack(
            [jnp.trace(gram), jnp.trace(gram @ gram), jnp.linalg.det(gram + jnp.eye(gram.shape[0]))]
        )

    components = vector_function(theta)
    gradients = []
    for index in range(components.shape[0]):
        def scalar_component(y):
            return vector_function(y)[index]

        cost = pymanopt_function.jax(manifold)(scalar_component)
        gradients.append(cost.get_gradient_operator()(theta))

    jac = jacobian_from_pymanopt(vector_function, manifold, theta)

    for index, grad in enumerate(gradients):
        covector = np.zeros(components.shape[0])
        covector[index] = 1.0
        t_grad = jac.T_matvec(covector)
        assert np.allclose(np.asarray(t_grad), np.asarray(grad))

    ambient_direction = np.random.randn(*theta.shape)
    tangent = manifold.projection(theta, ambient_direction)
    jac_matvec = jac.matvec(tangent)
    for index, grad in enumerate(gradients):
        expected = manifold.inner_product(theta, grad, tangent)
        assert float(jac_matvec[index]) == pytest.approx(expected, rel=1e-7, abs=1e-9)
