"""
graphlet_analyzer.py – Control Flow Graph (CFG) graphlet pattern detector.

Parses Python source into an AST and identifies costly subgraph patterns
(graphlets) of 2-3 CFG nodes.  Returns a composite score used by Track C.
"""
import ast
from typing import Any


# ── Pattern cost weights ──────────────────────────────────────────────────────
_PATTERN_COSTS: dict[str, float] = {
    "NestedLoop": 3.0,
    "LoopWithCall": 1.5,
    "DeepBranch": 2.0,
    "RepeatedComprehension": 1.0,
}


def _is_loop(node: ast.AST) -> bool:
    """Return True if *node* is a loop statement."""
    return isinstance(node, (ast.For, ast.While))


def _body_calls(body: list[ast.stmt]) -> bool:
    """Return True if *body* contains at least one function call."""
    for stmt in body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call):
                return True
    return False


def _detect_nested_loops(tree: ast.AST) -> int:
    """Count For→For or While→While (and mixed) nested pairs."""
    count = 0
    for node in ast.walk(tree):
        if _is_loop(node):
            body: list[ast.stmt] = getattr(node, "body", [])
            for child in body:
                if _is_loop(child):
                    count += 1
    return count


def _detect_loop_with_call(tree: ast.AST) -> int:
    """Count loops whose body contains at least one Call node."""
    count = 0
    for node in ast.walk(tree):
        if _is_loop(node):
            body: list[ast.stmt] = getattr(node, "body", [])
            if _body_calls(body):
                count += 1
    return count


def _detect_deep_branch(tree: ast.AST, depth: int = 3) -> int:
    """Count If→If→If chains of *depth* or more levels."""
    def _if_depth(node: ast.AST) -> int:
        if not isinstance(node, ast.If):
            return 0
        inner = max((_if_depth(c) for c in ast.walk(node)
                     if c is not node and isinstance(c, ast.If)), default=0)
        return 1 + inner

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _if_depth(node) >= depth:
            count += 1
    return count


def _detect_repeated_comprehension(tree: ast.AST) -> int:
    """Count functions that contain 2+ ListComp nodes."""
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            comps = [c for c in ast.walk(node) if isinstance(c, ast.ListComp)]
            if len(comps) >= 2:
                count += 1
    return count


def analyze_graphlets(code: str) -> dict[str, Any]:
    """Parse *code* and detect costly CFG graphlet patterns.

    Parameters
    ----------
    code:
        Python source code as a plain string.

    Returns
    -------
    dict with keys:
        ``patterns_found`` – list of dicts, one per detected instance
        ``total_cost``     – cumulative weighted cost (float)
        ``score``          – efficiency score in [0.0, 1.0]
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"patterns_found": [], "total_cost": 0.0, "score": 1.0}

    detectors: dict[str, Any] = {
        "NestedLoop": _detect_nested_loops(tree),
        "LoopWithCall": _detect_loop_with_call(tree),
        "DeepBranch": _detect_deep_branch(tree),
        "RepeatedComprehension": _detect_repeated_comprehension(tree),
    }

    patterns_found: list[dict[str, Any]] = []
    total_cost = 0.0
    for name, count in detectors.items():
        if count > 0:
            cost = _PATTERN_COSTS[name] * count
            total_cost += cost
            patterns_found.append({
                "pattern": name,
                "count": count,
                "cost_weight": _PATTERN_COSTS[name],
                "total_cost": cost,
            })

    score = max(0.0, 1.0 - total_cost / 10.0)
    return {
        "patterns_found": patterns_found,
        "total_cost": round(total_cost, 4),
        "score": round(score, 4),
    }
