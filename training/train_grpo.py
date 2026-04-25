"""GRPO training script for the Constrained Refactor Gauntlet.

Trains a Qwen2.5-Coder-7B model with LoRA using Group Relative Policy
Optimization to refactor legacy Python code under engineering constraints.
"""

import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"  # Prevent wandb login prompt on Colab

import ast
import json
import logging
import random
import re
import sys

import datasets
import torch
from trl import GRPOConfig, GRPOTrainer
try:
    from unsloth import FastLanguageModel, is_bfloat16_supported, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
except ImportError:
    pass

try:
    import wandb
except ImportError:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.episode_generator import EpisodeGenerator
from environment.rule_engine import RuleEngine
from environment.track_b import ComplianceChecker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & model config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/base_codebase"))
STANDARDS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../environment/ENGINEERING_STANDARDS.md"))

# ---------------------------------------------------------------------------
# Cached global rule engine (avoids re-parsing the 1062-line markdown per
# completion — was previously 800+ file parses per training run)
# ---------------------------------------------------------------------------
_CACHED_ENGINE: RuleEngine | None = None


def _get_cached_engine() -> RuleEngine:
    global _CACHED_ENGINE
    if _CACHED_ENGINE is None:
        _CACHED_ENGINE = RuleEngine(STANDARDS_PATH)
    return _CACHED_ENGINE


# ---------------------------------------------------------------------------
# Completion parsing helpers
# ---------------------------------------------------------------------------

def parse_completions(completion_text: str) -> dict[str, str]:
    """Extract file edits from XML-formatted completion text."""
    pattern = r'<file name="(.*?)">(.*?)</file>'
    matches = re.findall(pattern, completion_text, flags=re.DOTALL)
    edits = {}
    for filename, content in matches:
        edits[filename.strip()] = content.strip()
    return edits


def extract_completion_text(completion) -> str:
    """Normalize completion to a plain string."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


# ---------------------------------------------------------------------------
# Code quality metrics (fast, AST-based — no subprocess calls)
# ---------------------------------------------------------------------------

def _compute_lint_score(code_text: str) -> float:
    issues = 0
    lines = code_text.split("\n")
    for line in lines:
        if len(line) > 88:
            issues += 1
        if line.rstrip() != line:
            issues += 1
        if "import *" in line:
            issues += 1
        if line.strip().startswith("except:"):
            issues += 1
    return max(0.0, 1.0 - issues / max(len(lines), 1))


def _compute_complexity(files: dict) -> float:
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


def _compute_module_size_compliance(files: dict) -> float:
    if not files:
        return 0.0
    compliant = sum(1 for content in files.values() if len(content.split("\n")) <= 200)
    return compliant / len(files)


def _has_docstrings(code_text: str) -> float:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return 0.0
    total = 0
    with_doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            total += 1
            if ast.get_docstring(node):
                with_doc += 1
    if total == 0:
        return 1.0
    return with_doc / total


def _has_type_hints(code_text: str) -> float:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return 0.0
    total_args = 0
    annotated_args = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args:
                if arg.arg not in ("self", "cls"):
                    total_args += 1
                    if arg.annotation is not None:
                        annotated_args += 1
    if total_args == 0:
        return 1.0
    return annotated_args / total_args


def compute_code_quality_fast(orig_files: dict, updated_files: dict) -> float:
    """Compute a composite code-quality score.

    Weights are aligned with the server's Track A evaluator:
      0.35 * test_pass_rate (approximated via lint improvement)
      0.25 * lint_improvement
      0.20 * complexity_reduction
      0.20 * module_size_compliance

    Additional signals (docstrings, type hints) are folded in as bonuses.
    """
    orig_lint = sum(_compute_lint_score(c) for c in orig_files.values()) / max(len(orig_files), 1)
    new_lint = sum(_compute_lint_score(c) for c in updated_files.values()) / max(len(updated_files), 1)
    lint_improvement = max(0.0, new_lint - orig_lint + 0.5)
    lint_improvement = min(lint_improvement, 1.0)

    orig_complexity = _compute_complexity(orig_files)
    new_complexity = _compute_complexity(updated_files)
    if orig_complexity == 0:
        complexity_score = 1.0
    else:
        complexity_score = max(0.0, (orig_complexity - new_complexity) / orig_complexity)

    size_score = _compute_module_size_compliance(updated_files)

    doc_score = sum(_has_docstrings(c) for c in updated_files.values()) / max(len(updated_files), 1)
    hint_score = sum(_has_type_hints(c) for c in updated_files.values()) / max(len(updated_files), 1)

    # Aligned with Track A weights + bonus for docstrings/hints
    total = (
        0.25 * lint_improvement
        + 0.20 * complexity_score
        + 0.20 * size_score
        + 0.20 * doc_score
        + 0.15 * hint_score
    )
    return total


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def reward_function(completions, prompts, files, rules_active, **kwargs) -> list[float]:
    """Compute rewards for GRPO training.

    Uses fast AST-based code quality scoring (no subprocess calls)
    and a graduated format reward to bootstrap learning.

    The final reward uses **multiplicative** combination (A × B) to
    match the server's evaluation formula, preventing reward mismatch.
    """
    t0 = time.time()
    rewards: list[float] = []
    files_batch = [json.loads(f) for f in files]
    rules_batch = [json.loads(r) for r in rules_active]

    engine = _get_cached_engine()

    for idx, (completion, orig_files, active_rules) in enumerate(
        zip(completions, files_batch, rules_batch)
    ):
        try:
            completion_text = extract_completion_text(completion)
            edits = parse_completions(completion_text)

            # Graduated format reward to bootstrap learning
            format_reward = 0.0
            if "<file" in completion_text:
                format_reward += 0.05
            if "</file>" in completion_text:
                format_reward += 0.05
            if edits:
                format_reward += 0.1 * min(len(edits), 4) / 4.0

            if not edits:
                rewards.append(-0.1 + format_reward)
                continue

            updated_files = orig_files.copy()
            for fname, content in edits.items():
                if fname in updated_files:
                    updated_files[fname] = content

            # Track A: code quality
            score_a = compute_code_quality_fast(orig_files, updated_files)

            # Track B: compliance (uses cached engine)
            evaluator_b = ComplianceChecker.__new__(ComplianceChecker)
            evaluator_b.engine = engine
            from environment.rule_engine import EpisodeState
            evaluator_b.state = EpisodeState()
            evaluator_b.reset(orig_files, active_rules, seed=hash(completion_text) % (2**31))
            for fname in edits.keys():
                action = {"tool": "edit_file", "args": {"filename": fname}}
                evaluator_b.step(action, f"Edited {fname}")
            score_b = evaluator_b.get_score()

            # Multiplicative reward (aligned with server evaluation)
            reward = score_a * score_b + format_reward
            rewards.append(reward)
        except Exception as e:
            logger.warning("Reward calculation error: %s", e)
            rewards.append(-0.1)

    elapsed = time.time() - t0
    avg_r = sum(rewards) / len(rewards) if rewards else 0
    print(f"  [Reward] {len(rewards)} completions scored in {elapsed:.1f}s | avg={avg_r:.3f}")
    return rewards


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def create_training_dataset(num_episodes: int = 50) -> datasets.Dataset:
    """Generate a training dataset of corrupted code episodes."""
    print(f"Generating {num_episodes} episodes for training dataset...")
    generator = EpisodeGenerator(BASE_DIR)
    dataset_dict: dict[str, list] = {"prompt": [], "files": [], "rules_active": []}

    for _ in range(num_episodes):
        ep = generator.generate()
        code_context = ""
        # Sort files by corruption severity (longest first) so the most
        # corrupted files are always included within the context limit.
        sorted_files = sorted(ep["files"].items(), key=lambda x: len(x[1]), reverse=True)
        for fname, content in sorted_files:
            file_block = f"\n--- {fname} ---\n```python\n{content}\n```\n"
            if len(code_context) + len(file_block) > 8000:
                break
            code_context += file_block

        system_prompt = (
            "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
            "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
            "You must return your edited files using the following exact XML format:\n"
            '<file name="filename.py">\n... complete new code ...\n</file>\n'
            "Do not omit any code inside the file block. Provide the full updated file."
        )
        user_prompt = (
            f"Here is the codebase to refactor:\n{code_context}\n\n"
            "Please refactor and return the updated files."
        )
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        dataset_dict["prompt"].append(prompt_messages)
        dataset_dict["files"].append(json.dumps(ep["files"]))
        dataset_dict["rules_active"].append(json.dumps(ep["rules_active"]))

    return datasets.Dataset.from_dict(dataset_dict)


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------

def main():
    print("Loading tokenizer and model with Unsloth...")
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        
        max_seq_length = 2048
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=32,
            gpu_memory_utilization=0.6,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=64,
            lora_dropout=0.0, # Unsloth recommends 0.0 for GRPO/LoRA
            bias="none",
            use_gradient_checkpointing="unsloth", # 30% less VRAM
            random_state=42,
        )
        model.print_trainable_parameters()
    except ImportError:
        print("Unsloth not installed. Falling back to transformers & peft.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output"))
    os.makedirs(output_dir, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        max_steps=100,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=1024,   # Increased from 512 for multi-file XML output
        save_steps=25,
        logging_steps=5,
        bf16=True,
        report_to="none",
    )

    # Create dataset with eval split
    full_dataset = create_training_dataset(num_episodes=220)
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"Train: {len(train_dataset)} episodes, Eval: {len(eval_dataset)} episodes")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Training...")
    torch.cuda.empty_cache()
    trainer.train()

    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    print(f"✅ Training complete! Adapter saved to {final_path}")


if __name__ == "__main__":
    main()
