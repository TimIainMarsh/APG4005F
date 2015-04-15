
import datetime

def log_timing_decorator(task_name, logger):
    def wrap_f(f):
        def wrapped_f(*args, **kwargs):
            now = datetime.datetime.now()
            res = f(*args, **kwargs)	
            dt = datetime.datetime.now() - now
            logger.error("%s - %s" % (task_name, dt))
            return res
        return wrapped_f
    return wrap_f
    
class log_timing(object):
    def __init__(self, task_name, logger):
        self.task_name = task_name
        self.logger = logger

    def __enter__(self):
        self.now = datetime.datetime.now()

    def __exit__(self, type, value, traceback):
        dt = datetime.datetime.now() - self.now
        self.logger.error("%s - %s" % (self.task_name, dt))
