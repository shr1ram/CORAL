"""ALE-Bench grader: scores a C++ solution on an AtCoder Heuristic Contest problem.

Uses ALE-Bench's public test cases only. Private cases are never touched during
a run; evaluate the final best solution with ALE-Bench's private_eval afterwards.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from pathlib import Path

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    """Score = mean absolute score over the problem's public test cases."""

    _session = None

    def _get_session(self):
        # CORAL runs each eval in a fresh process, so the session is rebuilt
        # every time. ale_bench.start() compiles the problem's Rust tools in
        # Docker (~20 s); keep the compiled binaries under the private dir
        # (agents cannot read it) and copy them in instead of rebuilding.
        if Grader._session is None:
            import importlib

            import ale_bench

            # ale_bench re-exports the `start` function, which shadows the module name.
            ale_start = importlib.import_module("ale_bench.start")

            cache = Path(self.private_dir) / "ale_bench_tools" / self.args["problem_id"]
            build = ale_start.build_rust_tools

            def build_or_reuse(tools_dir: Path) -> None:
                if (cache / ".built").exists():
                    shutil.copytree(cache, tools_dir / "target" / "release", dirs_exist_ok=True)
                    return
                build(tools_dir)
                release = tools_dir / "target" / "release"
                cache.mkdir(parents=True, exist_ok=True)
                for binary in release.iterdir():
                    if binary.is_file() and os.access(binary, os.X_OK):
                        shutil.copy2(binary, cache / binary.name)
                (cache / ".built").touch()

            ale_start.build_rust_tools = build_or_reuse
            try:
                Grader._session = ale_bench.start(
                    problem_id=self.args["problem_id"],
                    lite_version=self.args.get("lite_version", False),
                    num_workers=self.args.get("num_workers", 4),
                )
            finally:
                ale_start.build_rust_tools = build
        return Grader._session

    def evaluate(self) -> ScoreBundle:
        solution_file = self.args.get("solution_file", "solution.cpp")
        path = os.path.join(self.codebase_path, solution_file)
        if not os.path.exists(path):
            return self.fail(f"{solution_file} not found")
        with open(path, encoding="utf-8") as f:
            code = f.read()

        try:
            result = self._get_session().public_eval(
                code, code_language=self.args.get("code_language", "cpp20")
            )
        except Exception as e:
            return self.fail(f"ALE-Bench evaluation failed: {e}")

        cases = result.case_results
        verdicts = Counter(c.judge_result.value for c in cases)
        mean_score = sum(c.absolute_score for c in cases) / len(cases)
        slowest = max(c.execution_time for c in cases)
        explanation = (
            f"Mean score {mean_score:.1f} over {len(cases)} public cases | "
            f"verdicts {dict(verdicts)} | slowest case {slowest:.2f}s"
        )

        failures = [c for c in cases if c.judge_result.value != "ACCEPTED"]
        feedback = explanation
        if failures:
            first = failures[0]
            detail = (first.error_str or first.message or "")[-1500:]
            feedback += f"\nFirst failing case: {first.judge_result.value}: {first.message}\n{detail}"
        scores = sorted(c.absolute_score for c in cases)
        feedback += (
            f"\nPer-case scores: min {scores[0]}, median {scores[len(scores) // 2]}, "
            f"max {scores[-1]}"
        )
        return self.score(mean_score, explanation, feedback=feedback)
