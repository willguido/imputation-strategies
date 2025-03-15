#!/usr/bin/env python

import time
import psutil
import logging
import subprocess
import tracemalloc  # for peak memory tracking

AC_SCRIPT = "/AutoComplete/experiments/run_ac.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ac_cost.log"),  
        logging.StreamHandler()  
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage_mb():
    process = psutil.Process()
    return process.memory_info().rss / (1024**2) 

# Run AC and log computational costs
def run_autocomplete():
    start_time = time.time()
    start_mem = get_memory_usage_mb()
    tracemalloc.start()  

    cmd = ["python", AC_SCRIPT]

    logger.info(f"Running AC with command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        logger.info("AutoComplete completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"AC failed with error: {e}")
        return

    end_time = time.time()
    end_mem = get_memory_usage_mb()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()  

    total_runtime = round(end_time - start_time, 3)
    total_memory = round(end_mem - start_mem, 3)
    peak_memory = round(peak_mem / (1024**2), 3) 

    logger.info(f"AC Runtime: {total_runtime}s, Total memory: {total_memory}MB, Peak memory: {peak_memory}MB")

if __name__ == "__main__":
    run_autocomplete()