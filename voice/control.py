import threading

stop_flag = threading.Event()

def stop_speaking():
    stop_flag.set()

def reset_stop_flag():
    stop_flag.clear()