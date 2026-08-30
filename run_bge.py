import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
log_path = os.path.abspath("bge_log.txt")

with open(log_path, "w", encoding="utf-8") as f:
    f.write("Starting rebuild_rag_bge.py execution...\n")
    f.flush()

os.system(f'py rebuild_rag_bge.py >> "{log_path}" 2>&1')
