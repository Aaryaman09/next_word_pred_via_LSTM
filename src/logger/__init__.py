from loguru import logger

# Remove default handler (prevents duplicate logs)
logger.remove()

# Configure logging
logger.add("app.log", rotation="5 MB", level="INFO", format="{time} | {level} | {message}")

# Function to get the logger instance (optional)
def get_logger():
    return logger
