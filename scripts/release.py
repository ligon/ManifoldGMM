#!/usr/bin/env python3
"""
Lightweight release checklist runner.

This script automates the steps we expect before cutting a release:

- verify the git working tree is clean;
- run the quick-check and slow test batteries;
- build the source and wheel distributions with Poetry;
- optionally tag and push the repository; and
- optionally publish the artifacts via ``poetry publish``.

It does not push to GitHub or Zenodo automatically unless requested.  The
default behaviour is intentionally conservative so maintainers can review the
git diff and release notes before tagging.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


class ReleaseError(Exception):
    """Raised when a release prerequisite fails."""


def read_version() -> str:
    if not PYPROJECT.exists():
        raise ReleaseError("pyproject.toml not found")
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    try:
        return data["tool"]["poetry"]["version"]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ReleaseError("Unable to locate version in pyproject.toml") from exc


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {"cwd": REPO_ROOT, "text": True}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, check=check, **kwargs)  # type: ignore[arg-type]


def ensure_clean_worktree() -> None:
    result = run(["git", "status", "--porcelain"], capture=True)
    if result.stdout.strip():
        raise ReleaseError("Git working tree is dirty; commit or stash changes first.")


def tag_release(version: str, *, push: bool) -> None:
    tag_name = f"v{version}"
    print(f"Creating tag {tag_name!r}...")
    run(["git", "tag", "-a", tag_name, "-m", f"ManifoldGMM {tag_name}"])
    if push:
        print("Pushing main branch…")
        run(["git", "push"])
        print("Pushing tags…")
        run(["git", "push", "--tags"])


def publish_release() -> None:
    print("Publishing distributions via `poetry publish`…")
    run(["poetry", "publish"])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run the ManifoldGMM release checklist.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running quick-check and slow-tests (not recommended).",
    )
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip `poetry build`.",
    )
    parser.add_argument(
        "--tag",
        action="store_true",
        help="Create an annotated git tag `v<version>` after checks succeed.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push main and tags to origin (implies --tag).",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run `poetry publish` after building artifacts.",
    )
    args = parser.parse_args(argv)

    version = read_version()
    print(f"Preparing release for version {version}")

    try:
        ensure_clean_worktree()

        if not args.skip_tests:
            print("Running quick-check…")
            run(["make", "quick-check"])
            print("Running slow-tests…")
            run(["make", "slow-tests"])
        else:
            print("Skipping tests per --skip-tests flag.")

        if not args.no_build:
            print("Building distributions with Poetry…")
            run(["poetry", "build"])
        else:
            print("Skipping build per --no-build flag.")

        if args.tag or args.push:
            tag_release(version, push=args.push)
        else:
            print("Tagging skipped (pass --tag to create v<version>).")

        if args.publish:
            publish_release()
        else:
            print("Publish skipped (pass --publish to run `poetry publish`).")

        print(
            "\nRelease checklist complete. "
            "Remember to draft a GitHub release and trigger the Zenodo upload."
        )
        return 0
    except (subprocess.CalledProcessError, ReleaseError) as exc:
        print(f"\nRelease aborted: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main(sys.argv[1:]))
