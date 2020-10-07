import logging.config


# Configuration of Logging module with colorization of log lines
_LOGCONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": '%(asctime)s | %(levelname)s | %(message)s',
            "datefmt": '%H:%M:%S'
        },
    },
    "handlers": {
        "console": {
            "class": "colorstreamhandler.ColorStreamHandler",
            "stream": "ext://sys.stderr",
            "level": "INFO",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


def config():
    logging.config.dictConfig(_LOGCONFIG)
