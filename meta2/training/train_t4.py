"""GRPO training script specifically optimized for Google Colab T4 GPU (16GB VRAM).

This script uses Unsloth with 4-bit quantization, FP16 (since T4 lacks native BF16),
and severely reduced batch sizes to prevent Out-Of-Memory (OOM) errors.
"""

import os
import sys
import types
import time
import torch

# ---------------------------------------------------------------------------
# Stub out missing optional deps
# ---------------------------------------------------------------------------
for _mod in [
    "vllm", "vllm.distributed", "vllm.distributed.device_communicators",
    "vllm.distributed.device_communicators.pynccl", "vllm.distributed.utils",
    "llm_blender", "liger_kernel"
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

sys.modules["vllm.distributed.device_communicators.pynccl"].PyNcclCommunicator = type("PyNcclCommunicator", (), {})
sys.modules["vllm.distributed.utils"].StatelessProcessGroup = type("StatelessProcessGroup", (), {})

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLED"] = "true"

from trl import GRPOConfig, GRPOTrainer

try:
    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("❌ Unsloth is strictly recommended for T4 GPU training to avoid OOM.")
    UNSLOTH_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import dataset and reward functions from the main script
from training.train_grpo import MODEL_NAME, create_training_dataset, reward_function

def main():
    print("="*60)
    print("🚀 Initializing T4-Optimized GRPO Training")
    print("="*60)
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected! Please select a T4 GPU runtime in Colab.")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Detected GPU: {gpu_name}")

    if not UNSLOTH_AVAILABLE:
        print("Falling back to standard Hugging Face PEFT (High chance of OOM on T4)")

    print("Loading tokenizer and model in 4-bit...")
    
    max_seq_length = 1536 # Reduced to save VRAM on T4
    
    if UNSLOTH_AVAILABLE:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            max_lora_rank=16, # Reduced LoRA rank for T4
            gpu_memory_utilization=0.5, # Aggressive memory limit for T4
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            load_in_4bit=True,
        )
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../grpo_output_t4"))
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # T4 OPTIMIZED SETTINGS
    # -------------------------------------------------------------
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,       # MUST be 1 for T4
        gradient_accumulation_steps=4,       # Increased to compensate for batch size
        max_steps=50,                        # Lower steps for a quick Colab run
        num_generations=2,                   # Minimum required by GRPO, keeps memory low
        generation_batch_size=1,             # Generate 1 at a time to prevent OOM
        max_completion_length=384,           # Reduced output limit
        save_steps=25,
        logging_steps=5,
        bf16=False,                          # T4 DOES NOT support native BF16
        fp16=True,                           # Use FP16 for T4
        report_to="none",
        gradient_checkpointing=True,         # Mandatory for T4 
    )

    # Smaller dataset to fit context lengths and train faster
    full_dataset = create_training_dataset(num_episodes=50)
    train_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)["train"]

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Training on T4...")
    torch.cuda.empty_cache()
    trainer.train()

    final_path = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_path)
    print(f"✅ Training complete! Adapter saved to {final_path}")

if __name__ == "__main__":
    main()
