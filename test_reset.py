"""Tests for the Constrained Refactor Gauntlet environment.

Covers server reset/step, reward function correctness, and adversarial robustness.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker
from training.train_grpo import (
    compute_code_quality_fast,
    parse_completions,
    reward_function,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "environment/base_codebase")
STANDARDS_PATH = os.path.join(os.path.dirname(__file__), "environment/ENGINEERING_STANDARDS.md")


# ---------------------------------------------------------------------------
# Episode generation tests
# ---------------------------------------------------------------------------

class TestEpisodeGeneration:
    def test_generates_files(self):
        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()
        assert "files" in ep
        assert len(ep["files"]) > 0

    def test_generates_rules(self):
        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()
        assert "rules_active" in ep
        assert len(ep["rules_active"]) > 0

    def test_has_episode_id(self):
        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()
        assert "episode_id" in ep
        assert len(ep["episode_id"]) > 0


# ---------------------------------------------------------------------------
# Compliance checker tests
# ---------------------------------------------------------------------------

class TestComplianceChecker:
    def test_reset_deterministic(self):
        """Same seed should produce the same initial triggers."""
        checker1 = ComplianceChecker(STANDARDS_PATH)
        checker2 = ComplianceChecker(STANDARDS_PATH)

        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()

        r1 = checker1.reset(ep["files"], ep["rules_active"], seed=42)
        r2 = checker2.reset(ep["files"], ep["rules_active"], seed=42)

        assert r1.newly_triggered == r2.newly_triggered

    def test_different_seeds_different_triggers(self):
        """Different seeds should produce different initial triggers."""
        checker1 = ComplianceChecker(STANDARDS_PATH)
        checker2 = ComplianceChecker(STANDARDS_PATH)

        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()

        r1 = checker1.reset(ep["files"], ep["rules_active"], seed=42)
        r2 = checker2.reset(ep["files"], ep["rules_active"], seed=99)

        # Different seeds should (almost certainly) produce different triggers
        assert r1.newly_triggered != r2.newly_triggered

    def test_step_processes_actions(self):
        checker = ComplianceChecker(STANDARDS_PATH)
        gen = EpisodeGenerator(BASE_DIR)
        ep = gen.generate()
        checker.reset(ep["files"], ep["rules_active"], seed=42)

        report = checker.step({"tool": "edit_file", "args": {"filename": "api.py"}}, "Edited api.py")
        assert report is not None
        assert isinstance(report.still_outstanding, set)


# ---------------------------------------------------------------------------
# Reward function tests (adversarial robustness)
# ---------------------------------------------------------------------------

class TestRewardFunction:
    def _make_episode(self):
        gen = EpisodeGenerator(BASE_DIR)
        return gen.generate()

    def test_empty_completion_penalized(self):
        """Empty completion should get a negative reward."""
        ep = self._make_episode()
        rewards = reward_function(
            completions=[""],
            prompts=["test"],
            files=[json.dumps(ep["files"])],
            rules_active=[json.dumps(ep["rules_active"])],
        )
        assert rewards[0] < 0, f"Empty completion should be penalized, got {rewards[0]}"

    def test_no_xml_format_penalized(self):
        """Completion without XML format should be penalized."""
        ep = self._make_episode()
        bad = "Here is the refactored code:\ndef hello():\n    print('hi')"
        rewards = reward_function(
            completions=[bad],
            prompts=["test"],
            files=[json.dumps(ep["files"])],
            rules_active=[json.dumps(ep["rules_active"])],
        )
        assert rewards[0] <= 0, f"Non-XML format should be penalized, got {rewards[0]}"

    def test_proper_format_positive(self):
        """Properly formatted completion should get a positive reward."""
        ep = self._make_episode()
        fname = list(ep["files"].keys())[0]
        good = f'<file name="{fname}">"""Module."""\ndef hello():\n    """Say hello."""\n    print("hi")\n</file>'
        rewards = reward_function(
            completions=[good],
            prompts=["test"],
            files=[json.dumps(ep["files"])],
            rules_active=[json.dumps(ep["rules_active"])],
        )
        assert rewards[0] > 0, f"Good completion should be positive, got {rewards[0]}"

    def test_copy_paste_low_reward(self):
        """Copying the original code without changes should yield minimal reward."""
        ep = self._make_episode()
        fname = list(ep["files"].keys())[0]
        # Return original code unchanged
        copy = f'<file name="{fname}">{ep["files"][fname]}</file>'
        rewards = reward_function(
            completions=[copy],
            prompts=["test"],
            files=[json.dumps(ep["files"])],
            rules_active=[json.dumps(ep["rules_active"])],
        )
        # Copy-paste should produce low (but not necessarily negative) reward
        assert rewards[0] < 0.5, f"Copy-paste should be low reward, got {rewards[0]}"


# ---------------------------------------------------------------------------
# Code quality metrics tests
# ---------------------------------------------------------------------------

class TestCodeQuality:
    def test_lint_score_clean_code(self):
        from training.train_grpo import _compute_lint_score
        clean = "def hello():\n    print('hi')\n"
        score = _compute_lint_score(clean)
        assert score > 0.9

    def test_complexity_empty_files(self):
        from training.train_grpo import _compute_complexity
        assert _compute_complexity({}) == 0.0

    def test_module_size_compliance(self):
        from training.train_grpo import _compute_module_size_compliance
        small = {"a.py": "\n".join(["x"] * 100)}
        assert _compute_module_size_compliance(small) == 1.0
        large = {"a.py": "\n".join(["x"] * 300)}
        assert _compute_module_size_compliance(large) == 0.0
