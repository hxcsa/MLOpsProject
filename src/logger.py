import logging
import os
from datetime import datetime

LOGS_DIR = 'logs'
os.makedirs(LOGS_DIR, exist_ok=True)

log_file = os.path.join(LOGS_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    filename=log_file,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)