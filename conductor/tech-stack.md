# Technology Stack: ManifoldGMM

## Core Language & Runtime
- **Python (>= 3.11):** Leverages modern Python features like advanced type hinting and performance improvements.

## Primary Frameworks & Libraries
- **pymanopt (GitHub version):** Provides the foundational Riemannian manifold definitions, projections, and optimization algorithms (e.g., TrustRegions).
- **JAX (>= 0.4.25):** Powering the autodiff backend for Jacobian operators and vectorized moment calculations, supporting high-performance research.
- **datamat (0.2.0a1):** Used for labeled array algebra, facilitating the mapping between observational data and econometric moment restrictions.

## Development & Quality Assurance
- **Poetry:** Manages project dependencies, virtual environments, and packaging.
- **Pytest:** The primary testing framework, integrated with `make` targets for quick-check and slow-tests.
- **Ruff:** Used for fast, comprehensive linting and import sorting.
- **Black:** Enforces a consistent code style across the repository.
- **Mypy:** Ensures type safety through static analysis, specifically for public APIs.

## Documentation & Research
- **Org-mode:** The primary format for all design docs, standards, and example walkthroughs, enabling seamless integration of LaTeX math and executable code blocks.
- **GNU Make:** Orchestrates quality checks, test execution, and environment setup via a `Makefile`.
