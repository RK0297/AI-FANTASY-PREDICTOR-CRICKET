import logging

def setup_logging(logfile='app.log', level=logging.INFO):
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        filename=logfile,
        level=level,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    return logging.getLogger()

def log_message(message, level="info"):
    """
    Logs a message with the specified level.
    """
    logger = logging.getLogger()
    if level.lower() == "debug":
        logger.debug(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        logger.info(message)


