import subprocess
import threading
import time
import requests
import gradio as gr

def start_fastapi():
    subprocess.Popen(["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"])

threading.Thread(target=start_fastapi, daemon=True).start()
time.sleep(3)

def reset_env():
    r = requests.post("http://localhost:7860/reset", json={})
    return r.json()

demo = gr.Interface(fn=reset_env, inputs=None, outputs="json",
    title="Constrained Refactor Gauntlet",
    description="Click to start a new RL episode. Endpoints: /reset /step /health/green /dashboard/co2")
demo.launch(server_port=7860)
