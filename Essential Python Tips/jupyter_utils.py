import logging
import os

def current_memory_check() :
    from memory_profiler import memory_usage
    mem_usage = memory_usage(-1, interval=1, timeout=1)[0]
    usage = f"{mem_usage:.3f} MiB"
    return usage
    

def current_notebook_name() :
    import ipyparams
    notebook_name = ipyparams.notebook_name
    return notebook_name