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

def get_tx_bytes(interface):
    """Read tx_bytes from sysfs for the given interface."""
    path = "/sys/class/net/" + interface + "/statistics/tx_bytes"
    try:
        with open(path, 'r') as f:
            return long(f.read().strip())
    except (IOError, OSError):
        return None


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
    interface = 'p3p1'
    prev_bytes = 0
    
    try:
        while True:
            current_bytes = get_tx_bytes(interface)
            
            if current_bytes is not None:
                bytes_delta = current_bytes - prev_bytes
                
                # Ensure delta is non-negative
                if bytes_delta < 0:
                    bytes_delta = 0
                
                prev_bytes = current_bytes
                
                # Create bar graph (scale bytes for display)
                bar = draw_bar(bytes_delta, 100000000, 50)
                
                # Print to terminal with carriage return for non-scrolling output
                if bytes_delta >= 100000000:
                    sys.stdout.write('\r\033[1mTX: XXX,XXX kbytes/s %s' % 
                                (bar))
                elif bytes_delta >= 1000000:
                    sys.stdout.write('\r\033[1mTX: %3d,%03d kbytes/s %s' % 
                                (bytes_delta//1000000, (bytes_delta/1000) % 1000, bar))
                else:
                    sys.stdout.write('\r\033[1mTX: %7d kbytes/s %s' % 
                                (bytes_delta//1000, bar))
                sys.stdout.flush()
            else:
                sys.stdout.write('\rError: Could not read from interface %s' % interface)
                sys.stdout.flush()
            
            time.sleep(1)
    except KeyboardInterrupt:
        sys.stdout.write('\n')
        print("Program interrupted by user. Exiting.")


if __name__ == '__main__':
    main()
