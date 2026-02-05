#!/usr/bin/env python3
"""Generate power curve comparing Manifold vs Euclidean Wald tests on PSD matrices.

This script demonstrates that the Euclidean approach has severely inflated size
(rejects even when null is true), while the Manifold approach has correct size
and proper power.
"""
import argparse
import numpy as np
import jax.numpy as jnp
from manifoldgmm import GMM, Manifold, MomentRestriction
from pymanopt.manifolds import PSDFixedRank, Euclidean

# Suppress JAX warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def gi_man(Y, x):
    """Moment function for manifold (factor) parameterization."""
    A = Y @ Y.T
    diff = jnp.outer(x, x) - A
    return diff[jnp.triu_indices(3)]


def gi_euc(theta, x):
    """Moment function for Euclidean (vech) parameterization."""
    A = jnp.zeros((3, 3))
    idx = jnp.triu_indices(3)
    A = A.at[idx].set(theta)
    A = A + A.T - jnp.diag(jnp.diag(A))
    diff = jnp.outer(x, x) - A
    return diff[idx]


def constraint_man(theta_point):
    """H0: A[1,1] = 0 for manifold parameterization."""
    Y = theta_point.value
    return (Y @ Y.T)[1, 1]


def constraint_euc(theta_point):
    """H0: A[1,1] = 0 for Euclidean parameterization."""
    return theta_point.value[3]


def run_simulation(
    effect_sizes: list[float],
    n_obs: int = 30,
    n_reps: int = 100,
    noise_scale: float = 0.1,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run Monte Carlo simulation for power curve.

    Parameters
    ----------
    effect_sizes : list of float
        Values of v[1] to test. v[1]=0 corresponds to null being true.
    n_obs : int
        Sample size per replication.
    n_reps : int
        Number of Monte Carlo replications per effect size.
    noise_scale : float
        Standard deviation of additive noise.
    alpha : float
        Significance level.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys 'effect_sizes', 'power_manifold', 'power_euclidean'
    """
    manifold_man = Manifold.from_pymanopt(PSDFixedRank(3, 1))
    manifold_euc = Manifold.from_pymanopt(Euclidean(6))

    rng = np.random.default_rng(seed)
    power_manifold = []
    power_euclidean = []

    for v1 in effect_sizes:
        v_true = np.array([[1.0], [v1], [0.0]])
        rej_man, rej_euc = 0, 0
        valid_man, valid_euc = 0, 0

        for rep in range(n_reps):
            # Generate data: x = z * v' + noise
            z = rng.normal(size=(n_obs, 1))
            data = z @ v_true.T + rng.normal(scale=noise_scale, size=(n_obs, 3))
            data_jax = jnp.array(data)

            # Manifold GMM
            try:
                res_man = GMM(
                    MomentRestriction(
                        gi_jax=gi_man, data=data_jax, manifold=manifold_man, backend="jax"
                    ),
                    initial_point=np.array([[1.0], [0.0], [0.0]]),
                    weighting=np.eye(6),
                ).estimate(verbose=0)
                pval = res_man.wald_test(constraint_man, q=1).p_value
                if not np.isnan(pval):
                    valid_man += 1
                    if pval < alpha:
                        rej_man += 1
            except Exception:
                pass

            # Euclidean GMM
            try:
                res_euc = GMM(
                    MomentRestriction(
                        gi_jax=gi_euc, data=data_jax, manifold=manifold_euc, backend="jax"
                    ),
                    initial_point=np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
                    weighting=np.eye(6),
                ).estimate(verbose=0)
                pval = res_euc.wald_test(constraint_euc, q=1).p_value
                if not np.isnan(pval):
                    valid_euc += 1
                    if pval < alpha:
                        rej_euc += 1
            except Exception:
                pass

        pm = rej_man / max(valid_man, 1)
        pe = rej_euc / max(valid_euc, 1)
        power_manifold.append(pm)
        power_euclidean.append(pe)

        if verbose:
            print(f"v1={v1:.2f}: Manifold={pm:.2f}, Euclidean={pe:.2f}")

    return {
        "effect_sizes": effect_sizes,
        "power_manifold": power_manifold,
        "power_euclidean": power_euclidean,
    }


def plot_power_curve(results: dict, output_path: str | None = None):
    """Plot power curves for Manifold vs Euclidean.

    Parameters
    ----------
    results : dict
        Output from run_simulation.
    output_path : str or None
        If provided, save figure to this path. Otherwise show interactively.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    effect = results["effect_sizes"]
    ax.plot(effect, results["power_manifold"], "o-", label="Manifold (PSD rank-1)",
            linewidth=2, markersize=8, color="#2E86AB")
    ax.plot(effect, results["power_euclidean"], "s--", label="Euclidean (unconstrained)",
            linewidth=2, markersize=8, color="#E94F37")

    # Reference lines
    ax.axhline(y=0.05, color="gray", linestyle=":", linewidth=1, label="α = 0.05")
    ax.axvline(x=0.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_xlabel("Effect size (v₁)", fontsize=12)
    ax.set_ylabel("Rejection rate", fontsize=12)
    ax.set_title("Wald Test Power: Manifold vs Euclidean\n(H₀: Σ₂₂ = 0)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min(effect) - 0.02, max(effect) + 0.02)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate(
        "Euclidean has\ninflated size!",
        xy=(0.0, 1.0),
        xytext=(0.08, 0.85),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        color="#E94F37",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate PSD Wald test power curves")
    parser.add_argument("--n-obs", type=int, default=30, help="Sample size")
    parser.add_argument("--n-reps", type=int, default=100, help="Monte Carlo replications")
    parser.add_argument("--noise", type=float, default=0.1, help="Noise scale")
    parser.add_argument("--output", type=str, default=None, help="Output file for plot")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()

    effect_sizes = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("Running Monte Carlo simulation...")
    print(f"  n_obs={args.n_obs}, n_reps={args.n_reps}, noise={args.noise}")
    print()

    results = run_simulation(
        effect_sizes=effect_sizes,
        n_obs=args.n_obs,
        n_reps=args.n_reps,
        noise_scale=args.noise,
    )

    print("\n=== Results ===")
    print("Effect | Manifold | Euclidean")
    print("-------|----------|----------")
    for i, e in enumerate(results["effect_sizes"]):
        pm = results["power_manifold"][i]
        pe = results["power_euclidean"][i]
        print(f" {e:5.2f} |   {pm:5.2f}  |   {pe:5.2f}")

    if not args.no_plot:
        plot_power_curve(results, args.output)


if __name__ == "__main__":
    main()
