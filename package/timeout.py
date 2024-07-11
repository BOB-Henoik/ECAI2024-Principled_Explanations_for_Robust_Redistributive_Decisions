from multiprocessing.pool import ThreadPool
from functools import wraps

TIMEOUT = 150


# def timeout():
#     """Timeout decorator, TIMEOUT = 150 seconds."""


def timeout_decorator(item):
    """Wrap the original function."""

    @wraps(item)
    def func_wrapper(*args, **kwargs):
        """Closure for function."""
        with ThreadPool(processes=1) as pool:
            async_result = pool.apply_async(item, args, kwargs)
            # raises a TimeoutError if execution exceeds max_timeout
            return async_result.get(TIMEOUT)

    return func_wrapper

    # return timeout_decorator
