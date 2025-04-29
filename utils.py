# utils.py
import logging
import sys
import os
from logging.handlers import RotatingFileHandler

def setup_logging(
    log_level: int = logging.INFO,
    log_to_console: bool = True,
    log_to_file: bool = True,
    log_dir: str = "logs",
    log_filename: str = "agent_run.log",
    max_log_size_mb: int = 10,
    backup_count: int = 3,
    log_format: str = '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
):
    """
    Configures logging for the agent application.

    Args:
        log_level: The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        log_to_console: Whether to output logs to the console (stdout).
        log_to_file: Whether to output logs to a rotating file.
        log_dir: The directory to store log files in.
        log_filename: The name of the log file.
        max_log_size_mb: Maximum size of the log file in megabytes before rotation.
        backup_count: Number of backup log files to keep.
        log_format: The format string for log messages.
        date_format: The format string for the timestamp in log messages.
    """
    # Get the root logger
    # Using a specific named logger can also be good practice if utils is part of a larger library
    # logger = logging.getLogger("godel_agent_logger") # Example named logger
    logger = logging.getLogger() # Get root logger to configure all module loggers
    logger.setLevel(log_level) # Set the minimum level for the root logger

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # --- Console Handler ---
    if log_to_console:
        # Check if a console handler already exists to avoid duplication
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            # Set level for handler (can be different from root logger if needed)
            # console_handler.setLevel(log_level)
            logger.addHandler(console_handler)
            # print("Console logging handler added.") # Debug print
        # else:
            # print("Console logging handler already exists.") # Debug print


    # --- File Handler ---
    if log_to_file:
        try:
            # Create log directory if it doesn't exist
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            log_filepath = os.path.join(log_dir, log_filename)

            # Check if a file handler for this specific file already exists
            if not any(isinstance(h, RotatingFileHandler) and h.baseFilename == os.path.abspath(log_filepath) for h in logger.handlers):
                # Use RotatingFileHandler for backups
                max_bytes = max_log_size_mb * 1024 * 1024
                file_handler = RotatingFileHandler(
                    log_filepath,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                # Set level for handler
                # file_handler.setLevel(log_level)
                logger.addHandler(file_handler)
                # print(f"Rotating file logging handler added for {log_filepath}.") # Debug print
            # else:
                # print(f"Rotating file logging handler for {log_filepath} already exists.") # Debug print

        except Exception as e:
            # Fallback to console if file logging setup fails
            print(f"Error setting up file logging to {os.path.join(log_dir, log_filename)}: {e}", file=sys.stderr)
            print("Logging to console only.", file=sys.stderr)
            # Ensure console logging is definitely enabled if file logging fails
            if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
                 # Add console handler if it wasn't added and log_to_console was false initially
                 console_handler = logging.StreamHandler(sys.stdout)
                 console_handler.setFormatter(formatter)
                 logger.addHandler(console_handler)


    # Test message (optional)
    # logger.debug("Logging setup complete.")

# Example of calling setup at module level (optional, usually called from main script)
# if __name__ == "__main__":
#     setup_logging(log_level=logging.DEBUG)
#     logging.info("Testing logging from utils.py")
#     logging.debug("This is a debug message.")
#     logging.warning("This is a warning.")