import sys
try:
    import fitz
except ImportError:
    print("fitz not installed")
    sys.exit(1)

import os
pdfs = ['Meta_OpenEnv_Hackathon_Participant_Help_Guide.pdf', 'OpenEnv_Hackathon_FAQs.pdf']
for pdf in pdfs:
    try:
        doc = fitz.open(pdf)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        with open(pdf.replace('.pdf', '_clean.txt'), 'w') as f:
            f.write(text)
        print(f"Successfully extracted {pdf}")
    except Exception as e:
        print(f"Error on {pdf}: {e}")
