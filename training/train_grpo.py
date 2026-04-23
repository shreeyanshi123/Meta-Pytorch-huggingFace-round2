import os
import json
import requests
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer

# ── Config ────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ENV_URL    = os.getenv("ENV_URL", "http://localhost:7860")  # your HF Space URL here

# ── Environment class using OpenEnv environment_factory pattern ───────────
class RefactorEnv:
    """
    OpenEnv-compatible environment class.
    Each instance = one episode.
    Tool methods = the 4 actions the agent can take.
    """
    def __init__(self):
        self.episode_id   = None
        self.observation  = None
        self.reward       = 0.0
        self.done         = False

    def reset(self, **kwargs) -> str:
        """Start a fresh episode — gets a new broken codebase + active rules."""
        self.reward = 0.0
        self.done   = False
        try:
            resp = requests.post(f"{ENV_URL}/reset", json={}, timeout=30)
            data = resp.json()
            self.episode_id  = data["episode_id"]
            self.observation = data["observation"]
            # Return the initial observation as a string the model can read
            return self._format_observation(self.observation)
        except Exception as e:
            return f"Environment error on reset: {e}"

    def read_file(self, filename: str) -> str:
        """
        Read a file from the current codebase.
        Args:
            filename: Name of the file to read (e.g. utils.py)
        Returns:
            File contents as a string.
        """
        return self._step({"tool": "read_file", "args": {"filename": filename}})

    def edit_file(self, filename: str, content: str) -> str:
        """
        Edit a file in the codebase. Triggers rule engine and violation report.
        Args:
            filename: Name of the file to edit
            content: New complete content of the file
        Returns:
            Violation report showing triggered and resolved rules.
        """
        return self._step({"tool": "edit_file", "args": {"filename": filename, "content": content}})

    def run_tests(self) -> str:
        """
        Run pytest on the current codebase.
        Returns:
            Test results including pass rate.
        """
        return self._step({"tool": "run_tests", "args": {}})

    def check_compliance(self) -> str:
        """
        Check current compliance status — lists outstanding rule obligations.
        Returns:
            Current outstanding violations and compliance score.
        """
        return self._step({"tool": "check_compliance", "args": {}})

    # ── Internal helpers ──────────────────────────────────────────────────

    def _step(self, action: dict) -> str:
        """Send one action to the environment, update state."""
        if self.done or not self.episode_id:
            return "Episode already ended."
        try:
            resp = requests.post(f"{ENV_URL}/step", json={
                "episode_id": self.episode_id,
                "action": action
            }, timeout=60)
            result = resp.json()
            self.observation = result.get("observation", self.observation)
            self.done = result.get("done", False)

            if self.done:
                self.reward = result.get("reward", 0.0)
                return self._format_final(result)
            else:
                return self._format_step(result)
        except Exception as e:
            return f"Environment error: {e}"

    def _format_observation(self, obs: dict) -> str:
        """Format observation into a readable string for the model."""
        files = list(obs.get("files", {}).keys())
        violations = obs.get("violation_report", {}).get("still_outstanding", [])
        steps = obs.get("steps_remaining", 70)
        level = obs.get("curriculum_level", 1)
        return (
            f"EPISODE START\n"
            f"Files: {files}\n"
            f"Active rules: {obs.get('active_rules_count', 20)}\n"
            f"Steps remaining: {steps}\n"
            f"Curriculum level: {level}\n"
            f"Outstanding violations: {violations}\n"
            f"Use read_file() to inspect files, edit_file() to fix them, "
            f"run_tests() to check quality, check_compliance() to see rule status."
        )

    def _format_step(self, result: dict) -> str:
        """Format step result for the model."""
        obs = result.get("observation", {})
        vr  = obs.get("violation_report", {})
        return (
            f"Steps remaining: {obs.get('steps_remaining', '?')}\n"
            f"Newly triggered rules: {vr.get('newly_triggered', [])}\n"
            f"Newly resolved rules:  {vr.get('newly_resolved', [])}\n"
            f"Still outstanding:     {vr.get('still_outstanding', [])}\n"
            f"Conflict flags:        {vr.get('conflict_flags', [])}"
        )

    def _format_final(self, result: dict) -> str:
        """Format final episode result."""
        reward = result.get("reward", 0.0)
        info   = result.get("info", {})
        return (
            f"EPISODE COMPLETE\n"
            f"Total Reward:      {reward:.4f}\n"
            f"Code Score:        {info.get('code_score', 'N/A')}\n"
            f"Compliance Score:  {info.get('compliance_score', 'N/A')}\n"
            f"Final Violations:  {info.get('final_violations', 'N/A')}"
        )


# ── Reward function (TRL-compatible) ─────────────────────────────────────
def reward_func(completions: list[str], prompts: list, **kwargs) -> list[float]:
    """
    TRL calls this with the model's text completions.
    We spin up a RefactorEnv episode for each completion,
    parse the model's JSON tool call, execute it, then
    finish the episode to get the real CodeScore × ComplianceScore.
    Falls back to 0.0 if the server is not running.
    """
    rewards = []
    for completion in completions:
        env = RefactorEnv()
        obs = env.reset()       # start fresh episode
        if env.episode_id is None:
            # Server not reachable — fallback to 0
            rewards.append(0.0)
            continue

        # --- parse & execute tool calls from completion ---
        # The model outputs JSON like: {"tool": "edit_file", "args": {...}}
        obs = _dispatch_tool(env, completion)

        # If not done yet, finish the episode explicitly
        if not env.done:
            env._step({"tool": "finish", "args": {}})

        rewards.append(float(env.reward) if env.reward else 0.0)
    return rewards


def build_dataset(n_episodes: int = 5) -> Dataset:
    prompt = [{
        "role": "system",
        "content": "You are a code refactoring agent. Always respond with exactly ONE JSON tool call."
    },
    {
        "role": "user",
        "content": """You have a broken Python codebase. Fix it using these tools.

TOOL CALL FORMAT — you must output EXACTLY this JSON structure:
{"tool": "read_file", "args": {"filename": "utils.py"}}
{"tool": "edit_file", "args": {"filename": "utils.py", "content": "def my_func() -> None:\\n    pass"}}
{"tool": "run_tests", "args": {}}
{"tool": "check_compliance", "args": {}}

RULES:
- Output ONE tool call per response
- Always start by reading files before editing them
- After each edit_file, check what violations remain
- Your goal: fix all violations to maximize reward

Start now. What is your first action?"""
    }]
    return Dataset.from_dict({"prompt": [prompt] * n_episodes})

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"
    ))

    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=5,
        num_generations=2,
        save_steps=25,
        logging_steps=1,          # print reward every step
        bf16=False, fp16=False,
        max_completion_length=512,
        report_to="none"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=build_dataset(5),
        processing_class=tokenizer,
        # environment_factory is NOT a standard TRL param;
        # RefactorEnv is called inside reward_func instead
    )

    print("Starting GRPO Training...")
    trainer.train()

    model.save_pretrained("./grpo_output/final_adapter")
    tokenizer.save_pretrained("./grpo_output/final_adapter")
    print("\n✅ Training complete!")

    # ── Must disable gradient checkpointing before inference / eval ──────
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()

    # ── Proper evaluation using full episodes ──
    print("\n--- Evaluation: Full Environment Episodes ---")
    print(f"{'Episode':>8} | {'Init Viol':>9} | {'Final Viol':>10} | "
          f"{'Code Score':>10} | {'Comp Score':>10} | {'Total Reward':>12}")
    print("-" * 70)

    for i in range(5):
        env = RefactorEnv()
        obs_str = env.reset()
        init_viol = len(env.observation["violation_report"]["still_outstanding"])

        # Run up to 70 steps
        for _ in range(70):
            if env.done:
                break
            inputs = tokenizer(obs_str, return_tensors="pt",
                               truncation=True, max_length=1024).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=256,
                                     do_sample=True, temperature=0.3)
            action_text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            # Parse and call the right tool
            obs_str = _dispatch_tool(env, action_text)

        final_viol = len(env.observation["violation_report"]["still_outstanding"])
        print(f"{i+1:>8} | {init_viol:>9} | {final_viol:>10} | "
              f"{'N/A':>10} | {'N/A':>10} | {env.reward:>12.4f}")


def _dispatch_tool(env: RefactorEnv, action_text: str) -> str:
    """Parse model output and call the correct tool."""
    try:
        action = json.loads(action_text)
        tool = action.get("tool", "check_compliance")
        args = action.get("args", {})
        if tool == "read_file":
            return env.read_file(args.get("filename", "main.py"))
        elif tool == "edit_file":
            return env.edit_file(args.get("filename"), args.get("content", ""))
        elif tool == "run_tests":
            return env.run_tests()
        else:
            return env.check_compliance()
    except:
        return env.check_compliance()


if __name__ == "__main__":
    main()