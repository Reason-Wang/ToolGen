import os
from typing import Union
from logging import WARNING, getLogger, INFO, StreamHandler, FileHandler, Formatter, DEBUG


class Logger:
    def __init__(self, logger, rank: int):
        self.logger = logger
        self.rank = rank

    def info(self, msg):
        if self.rank in [-1, 0]:
            self.logger.info(msg)

    def debug(self, msg):
        if self.rank in [-1, 0]:
            self.logger.debug(msg)

    def warning(self, msg):
        if self.rank in [-1, 0]:
            self.logger.warning(msg)


def get_logger(directory, level="INFO", rank: int=-1):
    # print(f"Local rank: {local_rank}")
    os.makedirs(directory, exist_ok=True)
    filename = directory + '/train'
    logger = getLogger(__name__)
    logger.propagate = False
    logger.handlers.clear()
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    # if level == "INFO":
    #     handler2 = FileHandler(filename=f"{filename}.log")
    #     handler2.setFormatter(Formatter("%(message)s"))
    #     logger.addHandler(handler2)
    #     logger.setLevel(INFO)
    # elif level == "DEBUG":
    #     handler1 = StreamHandler()
    #     handler1.setFormatter(Formatter("%(message)s"))
    #     handler2 = FileHandler(filename=f"{filename}.log")
    #     handler2.setFormatter(Formatter("%(message)s"))
    #     logger.addHandler(handler1)
    #     logger.addHandler(handler2)
    if level == "WARNING":
        logger.setLevel(WARNING)
    elif level == "INFO":
        logger.setLevel(INFO)
    elif level == "DEBUG":
        logger.setLevel(DEBUG)
    else:
        raise ValueError(f"Unknown level: {level}")

    logger = Logger(logger, rank)

    return logger


if __name__=="__main__":
    logger = get_logger("test", level="INFO")
    logger.info("test")