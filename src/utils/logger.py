import logging
import os

LOG_FILE = "logs/app.log"

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_logger(name: str):
    return logging.getLogger(name)