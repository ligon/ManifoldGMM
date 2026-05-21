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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **ManifoldGMM** (1804 symbols, 3261 relationships, 155 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/ManifoldGMM/context` | Codebase overview, check index freshness |
| `gitnexus://repo/ManifoldGMM/clusters` | All functional areas |
| `gitnexus://repo/ManifoldGMM/processes` | All execution flows |
| `gitnexus://repo/ManifoldGMM/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
