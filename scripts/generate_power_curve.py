import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import Sphere, Euclidean

# 1. Setup
ROT90 = jnp.array([[0.0, -1.0], [1.0, 0.0]], dtype=jnp.float64)

def gi_jax(theta, observation):
    theta_perp = ROT90 @ theta
    return jnp.array([jnp.dot(theta_perp, observation)], dtype=jnp.float64)

def gi_euc(theta, observation):
    return observation - theta

def constraint(theta_point):
    return theta_point.value[1]

# 2. Parameters
rng = np.random.default_rng(42)
n_obs = 100
alpha = 0.05
angles = np.linspace(0, 0.20, 11) # Finer grid
n_reps = 200 # More reps

manifold = Manifold.from_pymanopt(Sphere(2))
manifold_euc = Manifold.from_pymanopt(Euclidean(2))

power_manifold = []
power_euclidean = []

print("Generating power curve (this may take a few minutes)...")

for phi in angles:
    mu_true = np.array([np.cos(phi), np.sin(phi)])
    rej_man = 0
    rej_euc = 0
    
    for _ in range(n_reps):
        data_raw = rng.normal(loc=mu_true, scale=0.5, size=(n_obs, 2))
        data_raw /= np.linalg.norm(data_raw, axis=1, keepdims=True)
        data_jax = jnp.array(data_raw)
        
        # Manifold GMM
        res_man = GMM(
            MomentRestriction(gi_jax=gi_jax, data=data_jax, manifold=manifold, backend="jax"),
            initial_point=jnp.array([1.0, 0.0])
        ).estimate(verbose=0)
        if res_man.wald_test(constraint, q=1).p_value < alpha:
            rej_man += 1
            
        # Euclidean GMM
        res_euc = GMM(
            MomentRestriction(gi_jax=gi_euc, data=data_jax, manifold=manifold_euc, backend="jax"),
            initial_point=jnp.array([1.0, 0.0])
        ).estimate(verbose=0)
        if res_euc.wald_test(constraint, q=1).p_value < alpha:
            rej_euc += 1
            
    power_manifold.append(rej_man / n_reps)
    power_euclidean.append(rej_euc / n_reps)
    print(f"Angle {phi:.3f}: Manifold={power_manifold[-1]:.3f}, Euclidean={power_euclidean[-1]:.3f}")

# 3. Plotting
plt.figure(figsize=(8, 5))
plt.plot(angles, power_manifold, 'o-', label='Manifold GMM')
plt.plot(angles, power_euclidean, 's--', label='Euclidean GMM')
plt.axhline(alpha, color='r', linestyle=':', label='Size (0.05)')
plt.xlabel('Deviation Angle (radians)')
plt.ylabel('Power (Rejection Rate)')
plt.title('Power Curve: Manifold vs Euclidean Wald Test')
plt.legend()
plt.grid(True)
plt.savefig('docs/examples/power_curve.png')
print("Power curve saved to docs/examples/power_curve.png")
