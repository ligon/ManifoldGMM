import jax.numpy as jnp
import numpy as np
from manifoldgmm import GMM, Manifold, MomentRestriction, ManifoldPoint
from pymanopt.manifolds import Euclidean

def gi_euc(theta, observation):
    A_vals = theta
    A = jnp.zeros((3, 3))
    idx = jnp.triu_indices(3)
    A = A.at[idx].set(A_vals)
    A = A + A.T - jnp.diag(jnp.diag(A))
    xxT = jnp.outer(observation, observation)
    diff = xxT - A
    return diff[idx]

def constraint_euc(theta_point):
    return theta_point.value[3]

data = np.random.randn(20, 3)
data_jax = jnp.array(data)
manifold_euc = Manifold.from_pymanopt(Euclidean(6))

print("Starting GMM estimate...")
res_euc = GMM(
    MomentRestriction(gi_jax=gi_euc, data=data_jax, manifold=manifold_euc, backend="jax"),
    initial_point=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
).estimate(verbose=0)
print("GMM estimate done.")

print("Starting Wald test...")
# Try calling tangent_basis separately
print("Calling tangent_basis...")
basis = res_euc.restriction.tangent_basis(res_euc.theta)
print(f"Tangent basis size: {len(basis)}")

print("Calling wald_test...")
wald = res_euc.wald_test(constraint_euc, q=1)
print(f"Wald test done. Stat: {wald.statistic}")
