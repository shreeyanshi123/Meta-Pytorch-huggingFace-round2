import os
import requests
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model
import datasets

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

def reward_function(completions, prompts):
    rewards = []
    
    for prompt, completion in zip(prompts, completions):
        try:
            res = requests.post(f"{ENV_URL}/reset")
            if res.status_code != 200:
                rewards.append(0.0)
                continue
                
            episode_id = res.json().get("episode_id")
            if not episode_id:
                rewards.append(0.0)
                continue
            
            action = {"tool": "check_compliance", "args": {}}
            requests.post(f"{ENV_URL}/step", json={"episode_id": episode_id, "action": action})
            
            action = {"tool": "finish", "args": {}}
            res = requests.post(f"{ENV_URL}/step", json={"episode_id": episode_id, "action": action})
            
            if res.status_code != 200:
                rewards.append(0.0)
                continue
                
            data = res.json()
            reward = data.get("reward", 0.0)
            if reward is None:
                reward = 0.0
                
            rewards.append(reward)
            
            if wandb.run:
                info = data.get("info", {})
                wandb.log({
                    "reward_mean": reward,
                    "CodeScore": info.get("code_score", {}).get("total", 0.0),
                    "ComplianceScore": info.get("compliance_score", 0.0)
                })
        except Exception as e:
            print(f"Reward calculation error: {e}")
            rewards.append(0.0)

    return rewards

def main():
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set up 4-bit quantization for Colab T4
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=500,
        num_generations=4,
        save_steps=50,
        logging_steps=10,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none"
    )
    
    dummy_data = {"prompt": ["Refactor this codebase to comply with standards."] * 100}
    dummy_dataset = datasets.Dataset.from_dict(dummy_data)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=dummy_dataset,
        processing_class=tokenizer,
    )
    
    print("Starting GRPO Training...")
    trainer.train()
    
    print("\n--- Evaluation ---")
    print(f"{'Episode ID':<36} | {'Init Viol':<10} | {'Final Viol':<10} | {'Code Score':<10} | {'Comp Score':<10} | {'Total Reward':<10}")
    print("-" * 100)
    
    for _ in range(5):
        try:
            res = requests.post(f"{ENV_URL}/reset")
            if res.status_code == 200:
                data = res.json()
                episode_id = data["episode_id"]
                init_viol = len(data["observation"]["violation_report"]["still_outstanding"])
                
                requests.post(f"{ENV_URL}/step", json={"episode_id": episode_id, "action": {"tool": "finish", "args": {}}})
                res = requests.post(f"{ENV_URL}/step", json={"episode_id": episode_id, "action": {"tool": "finish", "args": {}}})
                
                if res.status_code != 200:
                    continue
                    
                data = res.json()
                info = data.get("info", {})
                
                final_viol = len(data["observation"]["violation_report"]["still_outstanding"])
                code_score = info.get("code_score", {}).get("total", 0.0)
                comp_score = info.get("compliance_score", 0.0)
                reward = data.get("reward", 0.0)
                
                if reward is None: reward = 0.0
                
                print(f"{episode_id:<36} | {init_viol:<10} | {final_viol:<10} | {code_score:<10.4f} | {comp_score:<10.4f} | {reward:<10.4f}")
        except Exception as e:
            print(f"Eval error: {e}")

if __name__ == "__main__":
    main()
