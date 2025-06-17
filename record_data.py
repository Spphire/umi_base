from diffusion_policy.real_world.record_data_manager_utils import DataRecordManager

import os
import psutil
import time
import sys
from loguru import logger

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 8
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)


def main(args=None):
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Data Recorder')
    parser.add_argument('--save_base_dir', type=str)
    parser.add_argument('--save_file_dir', type=str, default='tests')
    parser.add_argument('--save_file_name', type=str, default=None, help='File name of the save file')
    parser.add_argument('--save_to_disk', action='store_true', default=False, help='Whether to save the data to disk')
    parser.add_argument('--debug', action='store_true', default=False, help='Whether to print debug messages')
    args = parser.parse_args()
    
    data_record_manager = DataRecordManager(
        record_base_dir=args.save_base_dir,
        record_file_dir=args.save_file_dir,
        record_debug=args.debug,
        save_to_disk=args.save_to_disk, 
        record_file_name=args.save_file_name
    )
    
    try:
        data_record_manager.start_recording()
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
    finally:
        data_record_manager.stop_recording()
        time.sleep(2)  # Reduced wait time
        sys.exit(0)


if __name__ == '__main__':
    main()