#!/usr/bin/env python3
"""Detect mismatches between callable parameters and documented names."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

SECTION_HEADERS = {
    "parameters",
    "args",
    "arguments",
    "keyword args",
    "keyword arguments",
    "inputs",
}
OTHER_HEADERS = {
    "returns",
    "yields",
    "raises",
    "examples",
    "notes",
    "references",
    "see also",
    "warns",
    "warnings",
    "attributes",
    "methods",
    "other parameters",
    "output",
    "outputs",
    "result",
    "results",
    "todo",
    "summary",
}
EXCLUDE_PARAMS = {"self", "cls", "mcls"}


@dataclass
class DocstringIssue:
    path: Path
    qualname: str
    missing: tuple[str, ...]
    extra: tuple[str, ...]


def normalize(name: str) -> str:
    return name.lstrip("*")


def is_dataclass_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "dataclass"
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "dataclasses"
            and node.attr == "dataclass"
        )
    if isinstance(node, ast.Call):
        return is_dataclass_decorator(node.func)
    return False


def extract_doc_params(docstring: str) -> set[str]:
    params: set[str] = set()
    if not docstring:
        return params
    lines = docstring.expandtabs().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip().lower().rstrip(":")
        if stripped in SECTION_HEADERS:
            i += 1
            while i < n and set(lines[i].strip()) <= {"-", "=", "~", "`", "_"} and lines[i].strip():
                i += 1
            param_indent = None
            while i < n:
                raw = lines[i]
                stripped_line = raw.strip()
                if not stripped_line:
                    break
                header_key = stripped_line.lower().rstrip(":")
                if header_key in SECTION_HEADERS.union(OTHER_HEADERS):
                    break
                if ":" not in raw:
                    i += 1
                    continue
                indent = len(raw) - len(raw.lstrip(" "))
                if param_indent is None:
                    param_indent = indent
                if indent > param_indent:
                    i += 1
                    continue
                prefix = raw.split(":", 1)[0]
                candidate = prefix.strip()
                if not candidate:
                    i += 1
                    continue
                token = candidate.split()[0].rstrip(",")
                token = normalize(token)
                if not token or token in OTHER_HEADERS:
                    i += 1
                    continue
                params.add(token)
                i += 1
            continue
        i += 1
    return params


def function_params(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return set()
    params: list[str] = []
    for arg in getattr(node.args, "posonlyargs", []):
        params.append(arg.arg)
    for arg in node.args.args:
        params.append(arg.arg)
    if node.args.vararg:
        params.append(node.args.vararg.arg)
    for arg in node.args.kwonlyargs:
        params.append(arg.arg)
    if node.args.kwarg:
        params.append(node.args.kwarg.arg)
    return {normalize(p) for p in params if p not in EXCLUDE_PARAMS}


def dataclass_params(node: ast.ClassDef) -> set[str]:
    params: set[str] = set()
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            params.add(normalize(item.target.id))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    params.add(normalize(target.id))
    return params


def class_params(node: ast.ClassDef) -> set[str]:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return function_params(item)
    if any(is_dataclass_decorator(dec) for dec in node.decorator_list):
        return dataclass_params(node)
    return set()


def iter_nodes(module: ast.Module, path: Path) -> Iterator[tuple[Path, list[str], ast.AST]]:
    def walk(body: Sequence[ast.stmt], parents: list[str]) -> Iterator[tuple[Path, list[str], ast.AST]]:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                qual_parents = parents + [node.name]
                yield path, qual_parents, node
                if isinstance(node, ast.ClassDef):
                    yield from walk(node.body, qual_parents)
                else:
                    yield from walk(node.body, qual_parents)
    yield from walk(module.body, [])


def collect_issues(paths: Iterable[Path]) -> list[DocstringIssue]:
    issues: list[DocstringIssue] = []
    for file_path in sorted(paths):
        try:
            source = file_path.read_text()
        except OSError as exc:
            raise RuntimeError(f"Unable to read {file_path}: {exc}") from exc
        tree = ast.parse(source, filename=str(file_path))
        for _, qual_parts, node in iter_nodes(tree, file_path):
            doc = ast.get_docstring(node)
            doc_params = extract_doc_params(doc or "")
            if not doc_params:
                continue
            if isinstance(node, ast.ClassDef):
                actual_params = class_params(node)
                if not actual_params:
                    continue
            else:
                actual_params = function_params(node)
            if not actual_params:
                continue
            missing = sorted(actual_params - doc_params)
            extra = sorted(doc_params - actual_params)
            if missing or extra:
                qualname = ".".join(qual_parts)
                issues.append(
                    DocstringIssue(
                        path=file_path,
                        qualname=qualname,
                        missing=tuple(missing),
                        extra=tuple(extra),
                    )
                )
    return issues


def iter_python_files(targets: Sequence[Path]) -> Iterator[Path]:
    for target in targets:
        if target.is_dir():
            yield from sorted(target.rglob("*.py"))
        elif target.suffix == ".py":
            yield target


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("src")],
        help="Directories or files to scan (default: src)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    targets = list(iter_python_files(args.paths))
    if not targets:
        print("No Python files to inspect.", file=sys.stderr)
        return 1
    issues = collect_issues(targets)
    if not issues:
        print("Docstring parameters match signatures.")
        return 0
    print("Docstring parameter inconsistencies detected:")
    cwd = Path.cwd()
    for issue in issues:
        try:
            rel_path = issue.path.resolve().relative_to(cwd)
        except ValueError:
            rel_path = issue.path
        print(f"- {rel_path}:{issue.qualname}")
        if issue.missing:
            print(f"  Missing in docstring: {', '.join(issue.missing)}")
        if issue.extra:
            print(f"  Documented but absent: {', '.join(issue.extra)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
