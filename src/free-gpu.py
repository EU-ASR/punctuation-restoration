#!/usr/bin/env python3

# Originally from:
# https://gist.github.com/99991/fd006be1d01c6777eb9e3b4dddbde3c6

import sys

import subprocess, getpass

def get_gpu_usage():
    """
    Returns a dict which contains information about memory usage for each GPU.

    In the following output, the GPU with id "0" uses 5774 MB of 16280 MB.
    253 MB are used by other users, which means that we are using 5774 - 253 MB.

    {
        "0": {
            "used": 5774,
            "used_by_others": 253,
            "total": 16280
        },
        "1": {
            "used": 5648,
            "used_by_others": 253,
            "total": 16280
        }
    }

    """

    # Name of current user, e.g. "root"
    current_user = getpass.getuser()

    # Find mapping from process ids to user names
    command = ["ps", "axo", "pid,user"]
    output = subprocess.check_output(command).decode("utf-8")
    pid_user = dict(row.strip().split()
        for row in output.strip().split("\n")[1:])

    # Find all GPUs and their total memory
    command = ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv"]
    output = subprocess.check_output(command).decode("utf-8")
    total_memory = dict(row.replace(",", " ").split()[:2]
        for row in output.strip().split("\n")[1:])

    # Store GPU usage information for each GPU
    gpu_usage = {gpu_id: {"used": 0, "used_by_others": 0, "total": int(total)}
        for gpu_id, total in total_memory.items()}

    # Use nvidia-smi to get GPU memory usage of each process
    command = ["nvidia-smi", "pmon", "-s", "m", "-c", "1"]
    output = subprocess.check_output(command).decode("utf-8")
    for row in output.strip().split("\n"):
        if row.startswith("#"): continue

        gpu_id, pid, type, mb, command = row.split()

        # Special case to skip weird output when no process is running on GPU
        if pid == "-": continue

        gpu_usage[gpu_id]["used"] += int(mb)

        # If the GPU user is different from us
        if pid_user[pid] != current_user:
            gpu_usage[gpu_id]["used_by_others"] += int(mb)

    return gpu_usage

def get_free_gpus(max_usage_mb=0):
    """
    Returns the ids of GPUs which are not occupied by other processes (0MB).
    """

    return [gpu_id for gpu_id, usage in get_gpu_usage().items()
        if usage["used"] <= max_usage_mb]

if __name__ == "__main__":
    import json

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--ngpus', help="number of GPUs to be allocated", type=int, default=1)
    args = parser.parse_args()

    print("GPU memory usage information:", file=sys.stderr)
    try:
        print(json.dumps(get_gpu_usage(), indent=4), file=sys.stderr)
    except:
        print("[[ ERROR: NO GPUs IN THE COMPUTER ??? ]]", file=sys.stderr)
        raise

    print("", file=sys.stderr)
    print("GPU ids of free GPUs:", get_free_gpus(), file=sys.stderr)

    free_gpu_list = get_free_gpus()
    if args.ngpus > len(free_gpu_list):
        print(f"Error: Asking for {args.ngpus}, but only {len(free_gpu_list)} GPUs are free.")
        sys.exit(1)

    return_string = ",".join(free_gpu_list[:args.ngpus])
    print(f"Returning: '{return_string}'", file=sys.stderr)

    print(return_string)

