"""Track B — Compliance checking against cascading engineering rules."""

import random
from typing import List, Dict, Set

from .rule_engine import RuleEngine, EpisodeState, ViolationReport

# Number of rules to trigger at episode start
INITIAL_TRIGGER_COUNT = 15


class ComplianceChecker:
    def __init__(self, standards_path: str):
        self.engine = RuleEngine(standards_path)
        self.state = EpisodeState()

    def reset(
        self,
        files: Dict[str, str],
        rules_active: List[int],
        seed: int | None = None,
    ) -> ViolationReport:
        """Reset the compliance state for a new episode.

        Args:
            files: Current codebase files.
            rules_active: IDs of rules enabled for this episode.
            seed: Optional RNG seed for deterministic initial triggers.
        """
        self.state = EpisodeState()

        rng = random.Random(seed)
        count = min(len(rules_active), INITIAL_TRIGGER_COUNT)
        initial_triggers = rng.sample(rules_active, count)

        for rid in initial_triggers:
            self.state.triggered_rules.add(rid)
            self.state.outstanding_obligations.add(rid)

        return ViolationReport(
            newly_triggered=set(initial_triggers),
            newly_resolved=set(),
            still_outstanding=set(initial_triggers),
            conflict_flags=[],
        )

    def step(self, action: dict, diff: str) -> ViolationReport:
        self.state.steps_taken += 1
        return self.engine.process_action(action, diff, self.state)

    def get_score(self) -> float:
        return self.engine.compute_compliance_score(self.state)

    def get_outstanding(self) -> List[int]:
        return list(self.state.outstanding_obligations)
