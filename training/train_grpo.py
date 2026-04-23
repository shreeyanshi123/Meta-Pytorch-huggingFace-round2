import os
import sys
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# ── Add project root to path so environment/ can be imported ──────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.episode_generator import EpisodeGenerator
from environment.track_a import CodeQualityEvaluator
from environment.track_b import ComplianceChecker

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME     = "Qwen/Qwen2.5-Coder-0.5B-Instruct"   # tiny model → fast
BASE_DIR       = os.path.join(PROJECT_ROOT, "environment", "base_codebase")
STANDARDS_PATH = os.path.join(PROJECT_ROOT, "environment", "ENGINEERING_STANDARDS.md")


# ── In-process environment (no HTTP server needed) ────────────────────────
class RefactorEnv:
    """
    Runs a single refactoring episode fully in-process.
    No HTTP server required — environment classes are imported directly.
    """

    def __init__(self):
        self.episode_id  = None
        self.files       = {}
        self.steps_remaining = 70
        self.done        = False
        self.reward      = 0.0
        self._track_a    = None
        self._track_b    = None
        self._obs        = {}

    def reset(self) -> str:
        """Start a fresh episode. Returns a readable observation string."""
        self.done             = False
        self.reward           = 0.0
        self.steps_remaining  = 70

        gen = EpisodeGenerator(BASE_DIR)
        data = gen.generate()

        self.episode_id = data["episode_id"]
        self.files      = data["files"]

        self._track_a = CodeQualityEvaluator()
        self._track_a.evaluate(self.files)           # set baselines

        self._track_b = ComplianceChecker(STANDARDS_PATH)
        report = self._track_b.reset(self.files, data["rules_active"])

        self._obs = {
            "files": self.files,
            "active_rules_count": len(data["rules_active"]),
            "steps_remaining": self.steps_remaining,
            "violation_report": {
                "newly_triggered": list(report.newly_triggered),
                "newly_resolved":  list(report.newly_resolved),
                "still_outstanding": list(report.still_outstanding),
                "conflict_flags":  report.conflict_flags,
            },
            "curriculum_level": data["curriculum_level"],
        }
        return self._format_observation()

    # ── Tool actions ──────────────────────────────────────────────────────

    def read_file(self, filename: str) -> str:
        content = self.files.get(filename, f"File '{filename}' not found.")
        return f"=== {filename} ===\n{content}"

    def edit_file(self, filename: str, content: str) -> str:
        self.files[filename] = content
        self._track_a.evaluate(self.files)
        report = self._track_b.step(
            {"tool": "edit_file", "args": {"filename": filename, "content": content}},
            f"Edited {filename}"
        )
        return self._apply_step(report)

    def run_tests(self) -> str:
        self._track_a.evaluate(self.files)
        report = self._track_b.step({"tool": "run_tests", "args": {}}, "Ran tests")
        return self._apply_step(report)

    def check_compliance(self) -> str:
        report = self._track_b.step({"tool": "check_compliance", "args": {}}, "Checked compliance")
        return self._apply_step(report)

    def finish(self) -> str:
        """Terminate the episode and compute the final reward."""
        return self._finalize()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _apply_step(self, report) -> str:
        self.steps_remaining -= 1
        self._obs["steps_remaining"] = self.steps_remaining
        self._obs["violation_report"] = {
            "newly_triggered": list(report.newly_triggered),
            "newly_resolved":  list(report.newly_resolved),
            "still_outstanding": list(report.still_outstanding),
            "conflict_flags":  report.conflict_flags,
        }
        if self.steps_remaining <= 0:
            return self._finalize()
        return (
            f"Steps remaining: {self.steps_remaining}\n"
            f"Newly triggered rules: {list(report.newly_triggered)}\n"
            f"Newly resolved rules:  {list(report.newly_resolved)}\n"
            f"Still outstanding:     {list(report.still_outstanding)}\n"
            f"Conflict flags:        {report.conflict_flags}"
        )

    def _finalize(self) -> str:
        self.done = True
        code_score      = self._track_a.evaluate(self.files)
        compliance_score = self._track_b.get_score()
        self.reward = float(code_score.total * compliance_score)
        return (
            f"EPISODE COMPLETE\n"
            f"Total Reward:     {self.reward:.4f}\n"
            f"Code Score:       {code_score.total:.4f}\n"
            f"Compliance Score: {compliance_score:.4f}\n"
            f"Final Violations: {len(self._track_b.get_outstanding())}"
        )

    def _format_observation(self) -> str:
        obs  = self._obs
        vr   = obs.get("violation_report", {})
        files = list(obs.get("files", {}).keys())
        return (
            f"EPISODE START\n"
            f"Files: {files}\n"
            f"Active rules: {obs.get('active_rules_count', 20)}\n"
            f"Steps remaining: {obs.get('steps_remaining', 70)}\n"
            f"Curriculum level: {obs.get('curriculum_level', 1)}\n"
            f"Outstanding violations: {vr.get('still_outstanding', [])}\n"
            f"Use read_file, edit_file, run_tests, or check_compliance."
        )


# ── Reward function (TRL-compatible) ─────────────────────────────────────
def reward_func(completions: list[str], prompts: list, **kwargs) -> list[float]:
    """
    Reward breakdown (additive):
      format_bonus  (0–0.25) : valid JSON tool call with 'tool' + 'args'
      tool_bonus    (0–0.10) : bonus for high-value tool choices (edit_file)
      code_r        (0–0.40) : improvement in CodeScore vs fresh baseline
      comp_r        (0–0.25) : compliance fraction after dispatching action
    Final = clamp(sum, 0.01, 0.99)

    Key design decisions:
    - Baseline is locked BEFORE the action so improvements are measurable.
    - comp_r uses the fraction of rules resolved, not raw score (avoids
      the always-1.0 trap when triggered_rules is empty).
    - format_bonus is 0.25 so it alone creates variance early in training.
    """
    rewards = []
    for completion in completions:
        # TRL ≥ 0.12 passes completions as a list of message-dicts.
        # Older TRL passes a plain string. Handle both.
        if isinstance(completion, list):
            completion_text = next(
                (m["content"] for m in reversed(completion)
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                ""
            )
        else:
            completion_text = str(completion)

        # ── 1. Parse the completion ───────────────────────────────────────
        parsed_action = None
        format_bonus  = 0.0
        tool_bonus    = 0.0
        try:
            raw = completion_text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
            parsed_action = json.loads(raw)
            if isinstance(parsed_action, dict) and "tool" in parsed_action and "args" in parsed_action:
                format_bonus = 0.25           # well-formed tool call
                # Extra bonus for high-signal actions
                if parsed_action["tool"] in ("edit_file", "check_compliance"):
                    tool_bonus = 0.10
        except (json.JSONDecodeError, Exception):
            pass                              # stays 0

        # ── 2. Environment reward ─────────────────────────────────────────
        code_r = 0.0
        comp_r = 0.0
        try:
            env = RefactorEnv()
            env.reset()  # sets baselines inside _track_a on first evaluate()

            # Lock a FRESH baseline BEFORE the action so improvement is measurable.
            # (reset() already called evaluate() once to set baseline_violations /
            #  baseline_complexity inside _track_a — so a second call here gives
            #  the true pre-action score.)
            pre_score = env._track_a.evaluate(env.files)

            # Dispatch the agent's action into the live environment.
            _dispatch_tool(env, completion_text)

            if not env.done:
                env.finish()

            # ── code_r: relative improvement over baseline ─────────────
            if env._track_a and env.files:
                post_score = env._track_a.evaluate(env.files)
                # Reward improvement; neutral at 0.5 (no change), peaks at 1.0
                delta = post_score.total - pre_score.total   # in [-1, 1]
                code_r = float(max(0.0, 0.5 + delta))        # shift to [0, 1]

            # ── comp_r: fraction of triggered rules resolved ───────────
            if env._track_b:
                state = env._track_b.state
                triggered = len(state.triggered_rules)
                resolved  = len(state.resolved_rules)
                if triggered > 0:
                    comp_r = float(resolved / triggered)
                else:
                    # No rules triggered yet — give small baseline credit
                    comp_r = 0.1

        except Exception as e:
            print(f"[reward_func] env error: {e}")

        # ── 3. Combine & clamp ────────────────────────────────────────────
        raw_reward = format_bonus + tool_bonus + 0.40 * code_r + 0.25 * comp_r
        final_reward = float(max(0.01, min(0.99, raw_reward)))

        print(
            f"[reward] fmt={format_bonus:.2f} tool={tool_bonus:.2f} "
            f"code_r={code_r:.3f} comp_r={comp_r:.3f} → {final_reward:.4f}"
        )
        rewards.append(final_reward)

    return rewards


# ── Dataset ───────────────────────────────────────────────────────────────
def build_dataset(n_episodes: int = 5) -> Dataset:
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a code refactoring agent. "
                "Always respond with exactly ONE JSON tool call."
            ),
        },
        {
            "role": "user",
            "content": (
                "You have a broken Python codebase. Fix it using these tools.\n\n"
                "TOOL CALL FORMAT — output EXACTLY one JSON object:\n"
                '{"tool": "read_file",  "args": {"filename": "utils.py"}}\n'
                '{"tool": "edit_file",  "args": {"filename": "utils.py", "content": "..."}}\n'
                '{"tool": "run_tests",  "args": {}}\n'
                '{"tool": "check_compliance", "args": {}}\n\n'
                "RULES:\n"
                "- Output ONE tool call per response\n"
                "- Always read files before editing them\n"
                "- After each edit_file, check remaining violations\n"
                "- Your goal: fix all violations to maximise reward\n\n"
                "Start now. What is your first action?"
            ),
        },
    ]
    return Dataset.from_dict({"prompt": [prompt] * n_episodes})


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=5,
        num_generations=2,
        save_steps=25,
        logging_steps=1,
        bf16=False,
        fp16=False,
        max_completion_length=512,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=build_dataset(5),
        processing_class=tokenizer,
    )

    print("Starting GRPO Training...")
    trainer.train()

    model.save_pretrained("./grpo_output/final_adapter")
    tokenizer.save_pretrained("./grpo_output/final_adapter")
    print("\n✅ Training complete!")

    # ── Re-enable cache for inference ────────────────────────────────────
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    # ── Evaluation: full environment episodes ────────────────────────────
    print("\n--- Evaluation: Full Environment Episodes ---")
    print(f"{'Episode':>8} | {'Init Viol':>9} | {'Final Viol':>10} | "
          f"{'Code Score':>10} | {'Comp Score':>10} | {'Total Reward':>12}")
    print("-" * 70)

    for i in range(5):
        env = RefactorEnv()
        try:
            obs_str = env.reset()
        except Exception as e:
            print(f"{i+1:>8} | reset failed: {e}")
            continue

        # Count initial violations safely
        vr = env._obs.get("violation_report", {})
        init_viol = len(vr.get("still_outstanding", []))

        # Run up to 70 steps
        for _ in range(70):
            if env.done:
                break
            inputs = tokenizer(
                obs_str, return_tensors="pt",
                truncation=True, max_length=1024
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=True, temperature=0.3
                )
            action_text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            obs_str = _dispatch_tool(env, action_text)

        if not env.done:
            env.finish()

        # Read final violation count
        final_viol = len(env._track_b.get_outstanding()) if env._track_b else "N/A"
        code_score = env._track_a.evaluate(env.files) if env._track_a else None
        comp_score = env._track_b.get_score() if env._track_b else 0.0

        cs_str = f"{code_score.total:.4f}" if code_score else "N/A"
        print(f"{i+1:>8} | {init_viol:>9} | {str(final_viol):>10} | "
              f"{cs_str:>10} | {comp_score:>10.4f} | {env.reward:>12.4f}")


# ── Tool dispatcher ───────────────────────────────────────────────────────
def _dispatch_tool(env: RefactorEnv, action_text) -> str:
    """Parse one JSON tool call from model output and execute it."""
    # Defensive: handle list-of-dicts if called directly with raw TRL output
    if isinstance(action_text, list):
        action_text = next(
            (m["content"] for m in reversed(action_text)
             if isinstance(m, dict) and m.get("role") == "assistant"),
            ""
        )
    # Strip markdown code fences if the model wraps its output
    text = str(action_text).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    try:
        action = json.loads(text)
        tool   = action.get("tool", "check_compliance")
        args   = action.get("args", {})

        if tool == "read_file":
            return env.read_file(args.get("filename", "main.py"))
        elif tool == "edit_file":
            return env.edit_file(
                args.get("filename", "main.py"),
                args.get("content", "")
            )
        elif tool == "run_tests":
            return env.run_tests()
        elif tool == "finish":
            return env.finish()
        else:
            return env.check_compliance()
    except (json.JSONDecodeError, Exception):
        # Model produced non-JSON output — default to a safe action
        return env.check_compliance()


if __name__ == "__main__":
    main()