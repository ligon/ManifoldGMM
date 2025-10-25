"""Automatic differentiation utilities."""

from typing import Any, Callable

from ..geometry import ManifoldPoint, PointProjectionFn
from .jax_backend import jacobian_operator

VectorFunctionAmbient = Callable[[Any], Any]


def jacobian_from_pymanopt(
    function: VectorFunctionAmbient,
    manifold: Any,
    value: Any,
    *,
    project_point: PointProjectionFn | None = None,
):
    """
    Build a Jacobian operator for a vector-valued function defined on a
    ``pymanopt`` manifold.

    Parameters
    ----------
    function:
        Callable accepting the ambient representation used by the manifold and
        returning ℝ^m (or a PyTree thereof).
    manifold:
        Instance of :class:`pymanopt.manifolds.manifold.Manifold`.
    value:
        Ambient coordinates for the evaluation point. This will be projected
        onto the manifold before differentiation.
    project_point:
        Optional callable that maps ``value`` back onto the manifold before
        creating the internal :class:`ManifoldPoint`.

    Returns
    -------
    :class:`~manifoldgmm.autodiff.jax_backend.JacobianOperator`
        Linear operator exposing =matvec= and =T_matvec= closures compatible
        with the manifold's tangent structure.
    """

    point = ManifoldPoint.from_pymanopt(
        manifold,
        value,
        project_point=project_point,
    )

    def wrapped(point_obj: ManifoldPoint) -> Any:
        return function(point_obj.as_pymanopt_data())

    return jacobian_operator(wrapped, point)


__all__ = ["jacobian_operator", "jacobian_from_pymanopt"]
