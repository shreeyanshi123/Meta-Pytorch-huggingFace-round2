"""
evaluate_agent.py — Run the trained LoRA agent against the environment.

Demonstrates the trained agent's ability to refactor corrupted Python
code while respecting cascading engineering rules.

Usage:
    python evaluate_agent.py [--episodes N] [--adapter-path PATH]
"""

import argparse
import json
import os
import sys
import time
import statistics

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker
from training.train_grpo import (
    parse_completions,
    compute_code_quality_fast,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_ADAPTER = os.path.join(os.path.dirname(__file__), "grpo_output/final_adapter")
BASE_DIR = os.path.join(os.path.dirname(__file__), "environment/base_codebase")
STANDARDS_PATH = os.path.join(os.path.dirname(__file__), "environment/ENGINEERING_STANDARDS.md")

SYSTEM_PROMPT = (
    "You are an expert Python refactoring agent. Your task is to clean up the provided codebase, "
    "improve its quality (tests, linting, complexity), and fix compliance issues.\n"
    "You must return your edited files using the following exact XML format:\n"
    '<file name="filename.py">\n... complete new code ...\n</file>\n'
    "Do not omit any code inside the file block. Provide the full updated file."
)


def load_model(adapter_path: str):
    """Load the base model + trained LoRA adapter."""
    print(f"Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    if os.path.exists(adapter_path):
        print(f"Attaching adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print(f"⚠ Adapter not found at {adapter_path} — using base model only")
        model = base_model

    model.eval()
    return tokenizer, model


def run_episode(tokenizer, model, episode_num: int) -> dict:
    """Run a single evaluation episode and return metrics."""
    generator = EpisodeGenerator(BASE_DIR)
    ep = generator.generate()

    # Build prompt
    code_context = ""
    for fname, content in ep["files"].items():
        block = f"\n--- {fname} ---\n```python\n{content}\n```\n"
        if len(code_context) + len(block) > 6000:
            break
        code_context += block

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Refactor:\n{code_context}\n\nReturn updated files."},
    ]

    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    edits = parse_completions(generated)

    # Compute scores
    updated_files = ep["files"].copy()
    for fname, content in edits.items():
        if fname in updated_files:
            updated_files[fname] = content

    score_a = compute_code_quality_fast(ep["files"], updated_files)

    checker = ComplianceChecker(STANDARDS_PATH)
    seed = hash(ep["episode_id"]) % (2**31)
    checker.reset(ep["files"], ep["rules_active"], seed=seed)
    for fname in edits.keys():
        checker.step({"tool": "edit_file", "args": {"filename": fname}}, f"Edited {fname}")
    score_b = checker.get_score()

    reward = score_a * score_b

    result = {
        "episode": episode_num,
        "files_edited": len(edits),
        "code_quality": round(score_a, 3),
        "compliance": round(score_b, 3),
        "reward": round(reward, 3),
        "gen_time_s": round(gen_time, 1),
        "outstanding_rules": len(checker.get_outstanding()),
    }

    print(f"  Episode {episode_num}: reward={reward:.3f} "
          f"(A={score_a:.3f} × B={score_b:.3f}), "
          f"edits={len(edits)}, time={gen_time:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained refactoring agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER, help="LoRA adapter path")
    args = parser.parse_args()

    print("=" * 60)
    print("Constrained Refactor Gauntlet — Agent Evaluation")
    print("=" * 60)

    tokenizer, model = load_model(args.adapter_path)
    results = []

    print(f"\nRunning {args.episodes} evaluation episodes...\n")
    for i in range(1, args.episodes + 1):
        result = run_episode(tokenizer, model, i)
        results.append(result)

    # Summary statistics
    rewards = [r["reward"] for r in results]
    code_scores = [r["code_quality"] for r in results]
    compliance_scores = [r["compliance"] for r in results]

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes:         {len(results)}")
    print(f"  Avg Reward:       {statistics.mean(rewards):.3f} ± {statistics.stdev(rewards):.3f}" if len(rewards) > 1 else f"  Avg Reward:       {statistics.mean(rewards):.3f}")
    print(f"  Avg Code Quality: {statistics.mean(code_scores):.3f}")
    print(f"  Avg Compliance:   {statistics.mean(compliance_scores):.3f}")
    print(f"  Best Reward:      {max(rewards):.3f}")
    print(f"  Worst Reward:     {min(rewards):.3f}")
    print(f"  Avg Gen Time:     {statistics.mean([r['gen_time_s'] for r in results]):.1f}s")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": {
            "episodes": len(results),
            "avg_reward": round(statistics.mean(rewards), 3),
            "avg_code_quality": round(statistics.mean(code_scores), 3),
            "avg_compliance": round(statistics.mean(compliance_scores), 3),
        }, "episodes": results}, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
