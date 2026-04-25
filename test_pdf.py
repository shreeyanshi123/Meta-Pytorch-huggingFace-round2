import sys
import subprocess

def run_pdf2txt(pdf_file):
    try:
        res = subprocess.run(['pdftotext', pdf_file, '-'], capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout
    except Exception:
        pass
    return None

print("Checking pdftotext...")
output = run_pdf2txt('OpenEnv_Hackathon_FAQs.pdf')
if output:
    lines = output.split('\n')
    for line in lines:
        if any(x in line.lower() for x in ['model', 'llama', 'qwen', 'openenv', 'pytorch', 'grpo', 'rubric']):
            print(line.strip())
else:
    print("pdftotext failed or not installed")
