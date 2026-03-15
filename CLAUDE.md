# ManifoldGMM

## AGENTS.org
Read `AGENTS.org` before making changes — it is the authoritative guide for AI assistants working on this project.

## Naming & Notation Standards
`docs/standards/naming_notation.org` defines canonical math-to-code mappings. These are strict:
- theta in M -> `ManifoldPoint`; T_theta M -> `TangentSpace`; Retr -> `retract`; Pi -> `proj`
- Never invent synonyms (e.g., don't use `project` when `proj` exists).
- `riem_` prefix only when both Riemannian and Euclidean variants coexist.
- Standards changes require a PR labeled `standards`.

## Quality Checks
Run before every handoff: `make check` (ruff, black, mypy, pytest). Or `make quick-check` to skip slow tests.

## Error Types
Use canonical exceptions: `ManifoldError`, `RetractionError`, `ProjectionError`, `JacobianShapeError`, `WeightingError`, `NumericalWarning`, `GaugeWarning`, `ConvergenceWarning`.

## Org-Mode Conventions
- ASCII-safe LaTeX only (no Unicode math characters).
- Display math: `\[ \]` or `\begin{equation}`.
- Emacs >= 28 required for export.
