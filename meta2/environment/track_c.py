"""
track_c.py – Green-Code Evaluator (Track C).

Measures energy efficiency of refactored code via:
  * AST-based graphlet analysis (see graphlet_analyzer.py)
  * CPU execution time via timeit (stdlib)
  * Peak memory usage via tracemalloc (stdlib)

No third-party dependencies are required.
"""
import os
import sys
import timeit
import tracemalloc
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any

from environment.graphlet_analyzer import analyze_graphlets


@dataclass
class GreenScore:
    """Composite green-code score for a refactoring episode."""

    graphlet_score: float      # AST-pattern efficiency [0, 1]
    cpu_improvement: float     # Relative CPU-time reduction [0, 1]
    memory_improvement: float  # Relative peak-memory reduction [0, 1]
    total: float               # Weighted aggregate [0, 1]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Return *value* clamped to [lo, hi]."""
    return max(lo, min(hi, value))


class GreenCodeEvaluator:
    """Track C evaluator: energy efficiency of generated code."""

    # ── Public API ──────────────────────────────────────────────────────────

    def measure_execution(self, files: dict[str, str]) -> dict[str, float]:
        """Run each Python file and measure CPU time + peak memory.

        Files are written to a temporary directory; each is timed with
        ``timeit`` (1 repeat, number=1) and memory-traced via ``tracemalloc``.

        Parameters
        ----------
        files:
            Mapping of ``filename → source_code``.

        Returns
        -------
        dict with keys:
            ``cpu_time_ms``    – total CPU execution time in milliseconds
            ``peak_memory_mb`` – peak resident memory in megabytes
        """
        cpu_total_ms = 0.0
        peak_mb = 0.0

        with tempfile.TemporaryDirectory() as tmpdir:
            for fname, content in files.items():
                fpath = os.path.join(tmpdir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w") as fh:
                    fh.write(content)

                # ── CPU timing via timeit ──────────────────────────────────
                setup_code = textwrap.dedent(f"""
                    import sys
                    sys.path.insert(0, {repr(tmpdir)})
                """)
                run_code = textwrap.dedent(f"""
                    with open({repr(fpath)}) as _f:
                        exec(compile(_f.read(), {repr(fpath)}, 'exec'), {{}})
                """)
                try:
                    elapsed = timeit.timeit(
                        stmt=run_code,
                        setup=setup_code,
                        number=1,
                        globals={},
                    )
                    cpu_total_ms += elapsed * 1000.0
                except Exception:
                    cpu_total_ms += 0.0

                # ── Peak memory via tracemalloc ────────────────────────────
                tracemalloc.start()
                try:
                    ns: dict[str, Any] = {}
                    exec(compile(content, fpath, "exec"), ns)  # noqa: S102
                except Exception:
                    pass
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_mb = max(peak_mb, peak / (1024 * 1024))

        return {
            "cpu_time_ms": round(cpu_total_ms, 3),
            "peak_memory_mb": round(peak_mb, 4),
        }

    def evaluate_graphlets(self, files: dict[str, str]) -> float:
        """Return average graphlet score across all *files*.

        Parameters
        ----------
        files:
            Mapping of ``filename → source_code``.

        Returns
        -------
        Average graphlet score in [0.0, 1.0].
        """
        if not files:
            return 1.0
        scores = [analyze_graphlets(code)["score"] for code in files.values()]
        return round(sum(scores) / len(scores), 4)

    def evaluate(
        self,
        orig_files: dict[str, str],
        updated_files: dict[str, str],
    ) -> GreenScore:
        """Compute a full GreenScore comparing *updated_files* vs *orig_files*.

        Parameters
        ----------
        orig_files:
            Original (pre-refactoring) source files.
        updated_files:
            Refactored source files produced by the agent.

        Returns
        -------
        :class:`GreenScore` with all fields in [0.0, 1.0].
        """
        graphlet_score = self.evaluate_graphlets(updated_files)

        orig_metrics = self.measure_execution(orig_files)
        new_metrics = self.measure_execution(updated_files)

        # ── CPU improvement ────────────────────────────────────────────────
        orig_cpu = orig_metrics["cpu_time_ms"] or 1e-6
        new_cpu = new_metrics["cpu_time_ms"]
        cpu_improvement = _clamp((orig_cpu - new_cpu) / orig_cpu)

        # ── Memory improvement ─────────────────────────────────────────────
        orig_mem = orig_metrics["peak_memory_mb"] or 1e-9
        new_mem = new_metrics["peak_memory_mb"]
        memory_improvement = _clamp((orig_mem - new_mem) / orig_mem)

        # ── Weighted aggregate ─────────────────────────────────────────────
        total = _clamp(
            0.40 * graphlet_score
            + 0.35 * cpu_improvement
            + 0.25 * memory_improvement
        )

        return GreenScore(
            graphlet_score=graphlet_score,
            cpu_improvement=cpu_improvement,
            memory_improvement=memory_improvement,
            total=total,
        )
