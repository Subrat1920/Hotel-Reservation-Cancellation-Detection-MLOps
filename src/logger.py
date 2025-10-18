import logging
import os

# Create logs directory if not exists
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, "application.log")

# --- FIX STARTS HERE ---
# Get the root logger
root_logger = logging.getLogger()

# If no handler is attached yet, configure logging
if not root_logger.handlers:
    log_formatter = logging.Formatter(
        "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Optional: also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)

# Expose the logging module as before
logging = logging
# --- FIX ENDS HERE ---
