"""Episode generator for the Constrained Refactor Gauntlet.

Loads a base Python codebase and applies configurable corruptions
to produce training episodes with varying difficulty levels.
"""

import ast
import logging
import os
import random
import uuid
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (extracted from magic numbers)
# ---------------------------------------------------------------------------
CURRICULUM_HISTORY_SIZE = 150
ESCALATION_THRESHOLD = 0.7
ESCALATION_WINDOW_SIZE = 50
MAX_CURRICULUM_LEVEL = 4

RENAME_FUNC_PROBABILITY = 0.4
RENAME_VAR_PROBABILITY = 0.3
DEAD_CODE_INSERT_PROBABILITY = 0.3
FILE_BLOAT_PROBABILITY = 0.5
BLOAT_TARGET_LINES = 310

PROMPT_CONTEXT_LIMIT = 8000
MIN_CORRUPTIONS = 4
MAX_CORRUPTIONS = 8
BASE_RULES_COUNT = 20
RULES_PER_LEVEL = 40
MAX_RULES = 150


class CurriculumManager:
    """Tracks agent performance and escalates difficulty."""

    def __init__(self):
        self.history: List[float] = []
        self.level: int = 1

    def add_result(self, reward: float):
        self.history.append(reward)
        if len(self.history) > CURRICULUM_HISTORY_SIZE:
            self.history.pop(0)

        if self._should_escalate():
            if self.level < MAX_CURRICULUM_LEVEL:
                self.level += 1
                self.history = []
                logger.info("Curriculum escalated to level %d", self.level)

    def _should_escalate(self) -> bool:
        if len(self.history) < CURRICULUM_HISTORY_SIZE:
            return False

        windows = [
            self.history[0:ESCALATION_WINDOW_SIZE],
            self.history[ESCALATION_WINDOW_SIZE:2 * ESCALATION_WINDOW_SIZE],
            self.history[2 * ESCALATION_WINDOW_SIZE:3 * ESCALATION_WINDOW_SIZE],
        ]

        for w in windows:
            if sum(w) / len(w) <= ESCALATION_THRESHOLD:
                return False
        return True


class EpisodeGenerator:
    """Generates corrupted codebase episodes for RL training."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.curriculum = CurriculumManager()

    def _load_base_files(self) -> Dict[str, str]:
        files = {}
        for root, _, filenames in os.walk(self.base_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    path = os.path.join(root, filename)
                    with open(path, "r") as f:
                        rel_path = os.path.relpath(path, self.base_dir)
                        files[rel_path] = f.read()
        return files

    def _inject_circular_imports(self, files: Dict[str, str]):
        if "api.py" in files and "utils.py" in files:
            files["api.py"] = "from .utils import *\n" + files["api.py"]
            files["utils.py"] = "from .api import *\n" + files["utils.py"]

    def _rename_to_cryptic(self, files: Dict[str, str]):
        counter = [0]  # mutable counter for unique names

        class RenameTransformer(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if random.random() < RENAME_FUNC_PROBABILITY:
                    counter[0] += 1
                    node.name = f"_fn_{counter[0]}"
                self.generic_visit(node)
                return node

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store) and random.random() < RENAME_VAR_PROBABILITY:
                    counter[0] += 1
                    node.id = f"_v{counter[0]}"
                return node

        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                tree = RenameTransformer().visit(tree)
                ast.fix_missing_locations(tree)
                files[filename] = ast.unparse(tree)
            except SyntaxError:
                pass

    def _remove_type_hints(self, files: Dict[str, str]):
        class TypeHintRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                node.returns = None
                self.generic_visit(node)
                return node

            def visit_arg(self, node):
                node.annotation = None
                return node

            def visit_AnnAssign(self, node):
                val = node.value if node.value is not None else ast.Constant(value=None)
                return ast.Assign(targets=[node.target], value=val)

        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                tree = TypeHintRemover().visit(tree)
                ast.fix_missing_locations(tree)
                files[filename] = ast.unparse(tree)
            except SyntaxError:
                pass

    def _bloat_with_dead_code(self, files: Dict[str, str]):
        dead_code = "\n".join([
            "    # This is dead code",
            "    _tmp_x = 42",
            "    for _i in range(10):",
            "        _tmp_x += _i",
        ])
        for filename in list(files.keys()):
            if random.random() < FILE_BLOAT_PROBABILITY:
                lines = files[filename].split("\n")
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if line.strip().startswith("def ") and random.random() < DEAD_CODE_INSERT_PROBABILITY:
                        new_lines.extend(dead_code.split("\n"))
                # Duplicate complete blocks (groups of 10 lines) to reach target
                while len(new_lines) < BLOAT_TARGET_LINES:
                    idx = random.randint(0, max(1, len(new_lines) - 10))
                    block = new_lines[idx:idx + 10]
                    new_lines.extend(block)
                files[filename] = "\n".join(new_lines)

    def _hardcode_secrets(self, files: Dict[str, str]):
        for filename, content in files.items():
            modified = content.replace(
                'os.getenv("API_KEY", "default_secret_key")', '"password123"'
            )
            modified = modified.replace(
                'os.getenv("DB_URL", "sqlite:///./tasks.db")',
                '"root:supersecret@localhost:5432/db"',
            )
            files[filename] = modified

    def _break_import_paths(self, files: Dict[str, str]):
        for filename, content in files.items():
            if "api.py" in filename:
                files[filename] = content.replace("from .utils import", "from ..utils import")
            if "main.py" in filename:
                files[filename] = content.replace("from .api import", "from api_broken import")

    def _remove_docstrings(self, files: Dict[str, str]):
        for filename, content in files.items():
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                        if ast.get_docstring(node):
                            node.body.pop(0)
                files[filename] = ast.unparse(tree)
            except SyntaxError:
                pass

    def _add_god_function(self, files: Dict[str, str]):
        if "utils.py" in files:
            god_function = (
                "def god_function(data, date_str, current_date):\n"
                "    # MERGED FUNCTION\n"
                "    try:\n"
                "        dt = datetime.fromisoformat(date_str)\n"
                "    except ValueError as e:\n"
                "        logger.error(f'Error parsing date {date_str}: {e}')\n"
                "        raise ValueError('Invalid date format. Expected ISO 8601.')\n"
                "    delta = dt - current_date\n"
                "    if delta.days < 0:\n"
                "        prio = 'OVERDUE'\n"
                "    else:\n"
                "        prio = 'NORMAL'\n"
                "    if not data.get('title') or len(data.get('title', '')) > 100:\n"
                "        valid = False\n"
                "    else:\n"
                "        valid = True\n"
                "    return dt, prio, valid\n"
            )
            files["utils.py"] += "\n\n" + god_function

    def generate(self) -> dict:
        files = self._load_base_files()

        corruptions = [
            self._inject_circular_imports,
            self._rename_to_cryptic,
            self._remove_type_hints,
            self._bloat_with_dead_code,
            self._hardcode_secrets,
            self._break_import_paths,
            self._remove_docstrings,
            self._add_god_function,
        ]

        num_corruptions = random.randint(MIN_CORRUPTIONS, MAX_CORRUPTIONS)
        chosen_corruptions = random.sample(corruptions, num_corruptions)

        for corr in chosen_corruptions:
            corr(files)

        num_rules = min(MAX_RULES, BASE_RULES_COUNT + (self.curriculum.level - 1) * RULES_PER_LEVEL)
        rules_active = list(range(1, num_rules + 1))

        return {
            "files": files,
            "rules_active": rules_active,
            "curriculum_level": self.curriculum.level,
            "episode_id": str(uuid.uuid4()),
        }
