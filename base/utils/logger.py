import os
import sys
import logging

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
  logger = logging.getLogger(logger_name)
  if logger.hasHandlers():  # Already existed
    raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                     f'Please use another name, or otherwise the messages '
                     f'may be mixed between these two loggers.')

  logger.setLevel(logging.DEBUG)
  formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

  # Print log message with `INFO` level or above onto the screen.
  sh = logging.StreamHandler(stream=sys.stdout)
  sh.setLevel(logging.INFO)
  sh.setFormatter(formatter)
  logger.addHandler(sh)

  if not logfile_name:
    return logger

  work_dir = work_dir or DEFAULT_WORK_DIR
  logfile_name = os.path.join(work_dir, logfile_name)
  if os.path.isfile(logfile_name):
    print(f'Log file `{logfile_name}` has already existed!')
    while True:
      decision = input(f'Would you like to overwrite it (Y/N): ')
      decision = decision.strip().lower()
      if decision == 'n':
        raise SystemExit(f'Please specify another one.')
      if decision == 'y':
        logger.warning(f'Overwriting log file `{logfile_name}`!')
        break

  os.makedirs(work_dir, exist_ok=True)

  # Save log message with all levels in log file.
  fh = logging.FileHandler(logfile_name)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  logger.addHandler(fh)

  return logger
