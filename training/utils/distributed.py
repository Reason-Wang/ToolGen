import os


def is_main_process():
    # Check if the current process is the main process
    rank = int(os.environ.get('RANK', -1))
    return rank == 0 or rank == -1


def get_rank():
    # When using this function, make sure to call it after deepspeed is initialized
    # Using launcher or deepspeed.initialize()
    # Get the current rank
    rank = int(os.environ.get('RANK', -1))
    return rank
