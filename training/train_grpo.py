import os
import re
import json
import requests
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# ─── Code samples for varied training prompts ───────────────────────────
CODE_SAMPLES = [
    {
        "filename": "utils.py",
        "code": """import os, sys, json
def processData(x):
    result = []
    for i in range(len(x)):
        if x[i] > 0:
            result.append(x[i] * 2)
        else:
            result.append(0)
    return result

def calc(a,b,c,d,e,f):
    return a+b+c+d-e*f

class myClass:
    def __init__(self, val):
        self.val = val
    def get(self):
        return self.val
""",
        "issues": "snake_case violations, no type hints, no docstrings, too many parameters in calc()"
    },
    {
        "filename": "data_handler.py",
        "code": """import requests
import json
import os

def getData(url):
    r = requests.get(url)
    data = json.loads(r.text)
    result = []
    for item in data:
        tmp = {}
        tmp['name'] = item['name']
        tmp['value'] = item['value']
        tmp['id'] = item['id']
        result.append(tmp)
    return result

def saveData(data, path):
    f = open(path, 'w')
    f.write(json.dumps(data))
    f.close()

def loadData(path):
    f = open(path, 'r')
    data = json.loads(f.read())
    f.close()
    return data
""",
        "issues": "no context managers for files, no error handling, camelCase naming, no type hints"
    },
    {
        "filename": "api_client.py",
        "code": """import requests

class apiClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token

    def makeRequest(self, endpoint, method, data=None):
        url = self.base_url + endpoint
        headers = {'Authorization': 'Bearer ' + self.token}
        if method == 'GET':
            r = requests.get(url, headers=headers)
        elif method == 'POST':
            r = requests.post(url, headers=headers, json=data)
        elif method == 'PUT':
            r = requests.put(url, headers=headers, json=data)
        elif method == 'DELETE':
            r = requests.delete(url, headers=headers)
        return r.json()

    def getUser(self, id):
        return self.makeRequest('/users/' + str(id), 'GET')

    def createUser(self, data):
        return self.makeRequest('/users', 'POST', data)
""",
        "issues": "camelCase methods, string concatenation instead of f-strings, no error handling, no type hints"
    },
    {
        "filename": "logger.py",
        "code": """import datetime

log_data = []

def log(msg):
    global log_data
    log_data.append(str(datetime.datetime.now()) + ' ' + msg)

def getLog():
    global log_data
    return log_data

def clearLog():
    global log_data
    log_data = []

def saveLog(path):
    global log_data
    f = open(path, 'w')
    for l in log_data:
        f.write(l + '\\n')
    f.close()
""",
        "issues": "global state, no class encapsulation, no log levels, no context manager, camelCase naming"
    },
    {
        "filename": "validator.py",
        "code": """def validate(data):
    errors = []
    if not data.get('name'):
        errors.append('name required')
    if not data.get('email'):
        errors.append('email required')
    if data.get('email') and '@' not in data['email']:
        errors.append('invalid email')
    if not data.get('age'):
        errors.append('age required')
    if data.get('age') and (data['age'] < 0 or data['age'] > 150):
        errors.append('invalid age')
    if not data.get('password'):
        errors.append('password required')
    if data.get('password') and len(data['password']) < 8:
        errors.append('password too short')
    return errors
""",
        "issues": "no type hints, no docstring, deeply nested logic, magic numbers, no validation class"
    },
]

SYSTEM_PROMPT = """You are an expert Python code refactoring agent. You will receive Python code with quality issues.
Your task is to refactor the code following these engineering standards:
1. Use snake_case for functions and variables
2. Add type hints to all function signatures
3. Add docstrings to all functions and classes
4. Use context managers (with statements) for file operations
5. Handle errors with try/except blocks
6. Follow single responsibility principle
7. Remove global state, use classes instead
8. Use f-strings instead of string concatenation
9. Add proper logging instead of print statements
10. Keep functions under 20 lines

Output your refactored code as a JSON array:
[{"filename": "example.py", "content": "refactored code here"}]

IMPORTANT: Output ONLY the JSON array, no explanations before or after."""


def build_training_prompts():
    """Build varied training prompts from code samples."""
    prompts = []
    for sample in CODE_SAMPLES:
        prompt = f"""Refactor the following Python file to comply with engineering standards.

### {sample['filename']}
```python
{sample['code']}
```

Known issues: {sample['issues']}

Output your refactored version as a JSON array with "filename" and "content" keys."""
        # Repeat each prompt to fill the dataset
        prompts.extend([prompt] * 20)
    return prompts


def reward_function(completions, prompts, **kwargs):
    """
    Evaluate model completions based on code quality.
    This creates reward VARIANCE between generations, which is critical for GRPO.
    """
    rewards = []

    for completion in completions:
        score = 0.0

        # ── 1. Format reward: does it output parseable JSON? (0.0 - 0.2) ──
        edits = []
        try:
            # Try to find a JSON array in the completion
            json_match = re.search(r'\[.*\]', completion, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list) and len(parsed) > 0:
                    score += 0.15
                    edits = parsed
                    # Bonus for having correct keys
                    if all(isinstance(e, dict) and "filename" in e and "content" in e for e in parsed):
                        score += 0.05
        except (json.JSONDecodeError, Exception):
            pass

        # ── 2. Valid Python reward: can the code compile? (0.0 - 0.25) ──
        valid_code_count = 0
        total_code = max(len(edits), 1)
        for edit in edits:
            if isinstance(edit, dict) and "content" in edit:
                try:
                    compile(edit["content"], edit.get("filename", "<string>"), "exec")
                    valid_code_count += 1
                except SyntaxError:
                    pass
        if valid_code_count > 0:
            score += 0.25 * (valid_code_count / total_code)

        # ── 3. Style compliance reward (0.0 - 0.25) ──
        code_text = ""
        for edit in edits:
            if isinstance(edit, dict) and "content" in edit:
                code_text += edit["content"] + "\n"

        if code_text:
            style_points = 0
            checks = 0

            # snake_case functions (no camelCase)
            checks += 1
            camel_funcs = re.findall(r'def\s+[a-z]+[A-Z]', code_text)
            if len(camel_funcs) == 0:
                style_points += 1

            # Type hints present
            checks += 1
            typed_funcs = re.findall(r'def\s+\w+\([^)]*:\s*\w+', code_text)
            all_funcs = re.findall(r'def\s+\w+\(', code_text)
            if len(all_funcs) > 0 and len(typed_funcs) / max(len(all_funcs), 1) > 0.5:
                style_points += 1

            # Docstrings present
            checks += 1
            docstrings = re.findall(r'""".*?"""', code_text, re.DOTALL)
            if len(docstrings) >= 1:
                style_points += 1

            # Context managers for file ops
            checks += 1
            has_open = "open(" in code_text
            has_with = "with open(" in code_text
            if not has_open or has_with:
                style_points += 1

            # Error handling
            checks += 1
            if "try:" in code_text or "except" in code_text:
                style_points += 1

            if checks > 0:
                score += 0.25 * (style_points / checks)

        # ── 4. Code substantiveness: not trivially short (0.0 - 0.15) ──
        code_len = len(code_text.strip())
        if code_len > 200:
            score += 0.10
        if code_len > 500:
            score += 0.05

        # ── 5. Penalty: if completion is empty or just prose (0.0 - 0.1) ──
        if "def " in completion or "class " in completion:
            score += 0.1

        rewards.append(round(min(score, 1.0), 4))

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
    )
    # Prepare quantized model for training (casts non-quantized layers properly)
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    training_args = GRPOConfig(
        output_dir="./grpo_output",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_steps=50,
        num_generations=2,
        save_steps=25,
        logging_steps=5,
        bf16=False,
        fp16=False,
        max_completion_length=512,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none"
    )

    # Build dataset with varied, informative prompts
    prompts = build_training_prompts()
    train_dataset = datasets.Dataset.from_dict({"prompt": prompts})

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO Training...")
    trainer.train()

    # Save the final adapter
    model.save_pretrained("./grpo_output/final_adapter")
    tokenizer.save_pretrained("./grpo_output/final_adapter")
    print("\n✅ Training complete! Adapter saved to ./grpo_output/final_adapter")

    # ── Quick evaluation ──
    print("\n--- Evaluation: Reward scores on 5 test prompts ---")
    test_prompts = [build_training_prompts()[i] for i in range(0, 25, 5)]
    for i, prompt in enumerate(test_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
        completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        reward = reward_function([completion], [prompt])[0]
        print(f"  Prompt {i+1}: reward = {reward:.4f} | length = {len(completion)} chars")

if __name__ == "__main__":
    main()
