"""Read-only macOS pressure and process identity, distinct from the allocator."""

import os
import re
import resource
import subprocess


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    start = subprocess.check_output(
        ["/bin/ps", "-p", str(pid), "-o", "lstart="], text=True
    ).strip()
    return {"pid": pid, "start_time": start}


def memory():
    rss = (
        int(
            subprocess.check_output(
                ["/bin/ps", "-p", str(os.getpid()), "-o", "rss="], text=True
            ).strip()
        )
        * 1024
    )
    swap = subprocess.check_output(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True
    ).strip()
    matched = re.search(r"used = ([0-9.]+)([MG])", swap)
    if not matched:
        raise RuntimeError("Unable to measure system swap.")
    return {
        "rss_bytes": rss,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "swap_used_bytes": int(
            float(matched[1]) * 1024 ** (2 if matched[2] == "M" else 3)
        ),
        "swap": swap,
    }
