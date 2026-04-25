import os
import json
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """
You are an expert software engineer tasked with refactoring a broken Python codebase while strictly adhering to 150 cascading engineering rules.
Your ultimate goal is to maximize the final reward, which is: CodeScore * ComplianceScore.

You have access to 4 tools:
1. read_file: {"tool": "read_file", "args": {"filename": "<str>"}}
2. edit_file: {"tool": "edit_file", "args": {"filename": "<str>", "content": "<str>"}}
3. run_tests: {"tool": "run_tests", "args": {}}
4. check_compliance: {"tool": "check_compliance", "args": {}}

You must budget your steps wisely (max 70 steps). 
Whenever you make changes, ensure they resolve outstanding compliance rules without triggering contradictory ones.
Respond ONLY with a valid JSON object representing a tool call. Do not include any other text.
Example: {"tool": "run_tests", "args": {}}
"""

def run_inference(observation: dict) -> dict:
    steps_remaining = observation.get("steps_remaining", 0)
    report = observation.get("violation_report", {})
    files = list(observation.get("files", {}).keys())
    
    prompt = f"""
    Steps remaining: {steps_remaining}
    Files available: {files}
    Violation report: {json.dumps(report, indent=2)}
    
    What is your next action?
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=512
        )
        action_json = response.choices[0].message.content.strip()
        
        if action_json.startswith("```json"):
            action_json = action_json[7:-3].strip()
        elif action_json.startswith("```"):
            action_json = action_json[3:-3].strip()
            
        return json.loads(action_json)
    except Exception as e:
        print(f"Inference error: {e}")
        return {"tool": "check_compliance", "args": {}}
