# utils/logger.py

import time

def log_progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
