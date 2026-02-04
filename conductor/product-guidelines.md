# Product Guidelines: ManifoldGMM

## Tone and Style
- **Academic and Formal:** Documentation and user-facing content must be precise, rigorous, and include relevant academic references.
- **Clarity of Purpose:** Every concept should be clearly defined, bridging the gap between mathematical abstraction and computational implementation.

## Documentation Standards
- **Org-mode Primary Format:** **CRITICAL:** All documentation, design notes, and example walkthroughs MUST be authored in Org-mode (`.org` files). This is essential for blending math, code, and design notes seamlessly and supporting high-quality LaTeX/PDF exports.
- **Consult Org-mode Skill:** For guidance on correct Org-mode syntax, structure, and capabilities, consult an available Org-mode skill or expert agent.
- **Mathematical Depth:** Docstrings for public APIs must include the formal mathematical definition, relevant LaTeX-formatted equations, and citations.
- **Usage-Oriented Examples:** Complement theoretical definitions with clear code examples demonstrating how to use the API for practical estimation tasks, preferably within Org-mode source blocks.

## Code and Naming Conventions
- **Strict Notation Adherence:** Use only the project's established naming and notation standards (e.g., `ManifoldPoint`, `TangentSpace`, `retract`, `proj`). Avoid introducing synonyms or non-standard abbreviations.
- **Type Hinting:** Mandatory type hints for all public functions and methods to ensure clarity and support static analysis.
- **Shape Annotations:** Include expected array shapes and manifold types in docstrings to assist users and maintainers.

## Visual Identity
- **Scientific and Clean:** Visualizations and plots should use standard scientific libraries (Matplotlib, Seaborn) and be formatted for high-quality, publication-ready output.
- **Clarity Over Complexity:** Prioritize clear, informative labels and clean layouts in all graphical representations.

## Maintenance and Enforcement
- **Iterative Standard Evolution:** Guidelines and standards should be reviewed and updated as the project expands to include new manifold types and econometric techniques.
- **Automated Quality Checks:** Utilize linters, custom scripts, and a robust test suite (including finite-difference checks) to automatically enforce project standards and ensure code quality.
