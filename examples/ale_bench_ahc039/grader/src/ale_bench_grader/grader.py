"""ALE-Bench grader: scores a C++ solution on an AtCoder Heuristic Contest problem.

Uses ALE-Bench's public test cases only. Private cases are never touched during
a run; evaluate the final best solution with ALE-Bench's private_eval afterwards.
"""

from __future__ import annotations

import os
from collections import Counter

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    """Score = mean absolute score over the problem's public test cases."""

    _session = None

    def _get_session(self):
        # One ALE-Bench session per grader worker: starting one takes ~25 s
        # (problem download + tool build), evaluating takes seconds.
        if Grader._session is None:
            import ale_bench

            Grader._session = ale_bench.start(
                problem_id=self.args["problem_id"],
                lite_version=self.args.get("lite_version", False),
                num_workers=self.args.get("num_workers", 4),
            )
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
