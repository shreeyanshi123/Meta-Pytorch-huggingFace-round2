"""Track A — Code quality evaluation via tests, linting, complexity, and module size."""

import ast
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeScore:
    """Composite code quality score with sub-metrics."""

    test_pass_rate: float
    lint_improvement: float
    complexity_reduction: float
    module_size_compliance: float
    total: float


class CodeQualityEvaluator:
    """Evaluates code quality using tests, linting, complexity, and module size."""

    def __init__(self):
        self.baseline_violations: int | None = None
        self.baseline_complexity: float | None = None

    def _run_tests(self, temp_dir: str) -> float:
        try:
            result = subprocess.run(
                ["pytest", temp_dir, "--collect-only", "-q"],
                capture_output=True, text=True,
            )
            if "no tests collected" in result.stdout:
                return 0.0

            result = subprocess.run(
                ["pytest", temp_dir],
                capture_output=True, text=True,
            )

            output = result.stdout
            if "failed" in output and "passed" in output:
                return 0.5
            elif "failed" in output:
                return 0.0
            return 1.0
        except Exception:
            return 0.0

    def _run_ruff(self, temp_dir: str) -> int:
        try:
            result = subprocess.run(
                ["ruff", "check", temp_dir],
                capture_output=True, text=True,
            )
            return len([line for line in result.stdout.split('\n') if ".py:" in line])
        except Exception:
            return 100

    def _compute_complexity(self, files: dict) -> float:
        total_branches = 0
        total_functions = 0

        for content in files.values():
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        for subnode in ast.walk(node):
                            if isinstance(subnode, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                                total_branches += 1
            except SyntaxError:
                pass

        if total_functions == 0:
            return 0.0
        return total_branches / total_functions

    def _compute_module_size(self, files: dict) -> float:
        if not files:
            return 0.0
        compliant_files = sum(1 for content in files.values() if len(content.split('\n')) <= 200)
        return compliant_files / len(files)

    def evaluate(self, files: dict) -> CodeScore:
        """Evaluate code quality and return a composite score.

        Weights:
            0.35 * test_pass_rate
            0.25 * lint_improvement
            0.20 * complexity_reduction
            0.20 * module_size_compliance
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            for filename, content in files.items():
                filepath = os.path.join(temp_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(content)

            test_pass_rate = self._run_tests(temp_dir)
            current_violations = self._run_ruff(temp_dir)
            current_complexity = self._compute_complexity(files)

            if self.baseline_violations is None:
                self.baseline_violations = current_violations
            if self.baseline_complexity is None:
                self.baseline_complexity = current_complexity

            if self.baseline_violations == 0:
                lint_improvement = 1.0
            else:
                lint_improvement = max(0.0, (self.baseline_violations - current_violations) / self.baseline_violations)

            if self.baseline_complexity == 0:
                complexity_reduction = 1.0
            else:
                complexity_reduction = max(0.0, (self.baseline_complexity - current_complexity) / self.baseline_complexity)

            module_size_compliance = self._compute_module_size(files)

            total = (
                0.35 * test_pass_rate
                + 0.25 * lint_improvement
                + 0.20 * complexity_reduction
                + 0.20 * module_size_compliance
            )

            logger.debug(
                "CodeScore: test=%.2f lint=%.2f complex=%.2f size=%.2f total=%.3f",
                test_pass_rate, lint_improvement, complexity_reduction,
                module_size_compliance, total,
            )

            return CodeScore(
                test_pass_rate=test_pass_rate,
                lint_improvement=lint_improvement,
                complexity_reduction=complexity_reduction,
                module_size_compliance=module_size_compliance,
                total=total,
            )
