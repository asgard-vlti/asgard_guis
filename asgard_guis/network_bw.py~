#!/usr/bin/env python
"""
Create a python 2.6 compatible program that 
finds the number from:
"/sys/class/net/" + interface + /statistics/tx_bytes"
with the default being p3p1 once per second and 
prints it to a non-scrolling terminal, along with an ascii
display of this number as a bar graph. 

The program should run indefinitely until interrupted by the user.
"""

import sys
import time
import os
import subprocess

def get_tx_bytes(interface):
    """Read tx_bytes from sysfs for the given interface."""
    path = "/sys/class/net/" + interface + "/statistics/tx_bytes"
    try:
        with open(path, 'r') as f:
            return long(f.read().strip())
    except (IOError, OSError):
        return None


def get_remote_tx_bytes(remote_host, remote_path, tmp_file):
    """Fetch tx_bytes from a remote host via rcp into a temp file, then read it."""
    devnull = open(os.devnull, 'w')
    try:
        ret = subprocess.call(
            ['rcp', remote_host + ':' + remote_path, tmp_file],
            stdout=devnull, stderr=devnull
        )
    finally:
        devnull.close()
    if ret != 0:
        return None
    try:
        with open(tmp_file, 'r') as f:
            return long(f.read().strip())
    except (IOError, OSError):
        return None


def format_line(label, delta):
    """Format a single display line with label, rate, and bar."""
    bar = draw_bar(delta, 100000000, 50)
    if delta >= 100000000:
        return '\033[1m%s: XXX,XXX kbytes/s %s\033[0m' % (label, bar)
    elif delta >= 1000000:
        return '\033[1m%s: %3d,%03d kbytes/s %s\033[0m' % (
            label, delta // 1000000, (delta // 1000) % 1000, bar)
    else:
        return '\033[1m%s: %7d kbytes/s %s\033[0m' % (label, delta // 1000, bar)


def draw_bar(value, max_value=100000000, width=50):
    """Draw an ASCII bar graph representation of the value."""
    if max_value == 0:
        ratio = 0.0
    else:
        ratio = float(value) / max_value
    
    # Clamp ratio to [0, 1]
    if ratio > 1.0:
        ratio = 1.0
    
    filled = int(width * ratio)
    if value < 20000000:
        bar = '\033[92m'
    elif value < 50000000:
        bar = '\033[93m'
    else:
        bar = '\033[91m'
    bar += '[' + '#' * filled + '-' * (width - filled) + ']\033[0m'
    return bar


def main():
    """Main program loop."""
    local_interface = 'p3p1'
    remote_host = 'wag'
    remote_path = '/sys/class/net/p7p2/tx_bytes'
    tmp_file = '/tmp/wag_tx_bytes'

    prev_local = 0
    prev_wag = 0

    try:
        while True:
            current_local = get_tx_bytes(local_interface)
            current_wag = get_remote_tx_bytes(remote_host, remote_path, tmp_file)

            if current_local is not None:
                delta_local = max(0, current_local - prev_local)
                prev_local = current_local
                line1 = format_line('Local TX', delta_local)
            else:
                line1 = '\033[1mLocal TX: Error reading interface\033[0m'

            if current_wag is not None:
                delta_wag = max(0, current_wag - prev_wag)
                prev_wag = current_wag
                line2 = format_line('wag TX  ', delta_wag)
            else:
                line2 = '\033[1mwag TX  : Error reading remote\033[0m'

            # Overwrite both lines in place using ANSI cursor control:
            # clear & print line 1, drop to line 2, clear & print it,
            # then move cursor back up to line 1 for the next iteration.
            sys.stdout.write('\r\033[K' + line1)
            sys.stdout.write('\n\r\033[K' + line2)
            sys.stdout.write('\033[1A')
            sys.stdout.flush()

            time.sleep(1)
    except KeyboardInterrupt:
        sys.stdout.write('\n\n')
        print("Program interrupted by user. Exiting.")


if __name__ == '__main__':
    main()
