import time


def get_time():
    timestamp = time.time()
    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    return readable_time
