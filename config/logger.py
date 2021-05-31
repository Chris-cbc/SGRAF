import logging.config

CONFIG = {
    "version": 1,
    "formatters": {
        "fmt": {
            "format":
                "[%(asctime)s][%(levelname)s][%(filename)s][line:%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "fmt",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "fmt",
            "filename": "../logs/sgraf.log",
            "mode": "a",
            "encoding": "utf-8"
        }
    },
    "loggers": {
        "main": {
            "level": "DEBUG",
            "handlers": ["file"]
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


class Logging(object):
    def __init__(self):
        logging.config.dictConfig(CONFIG)
        self.log = logging.getLogger('main')
        assert self.log
