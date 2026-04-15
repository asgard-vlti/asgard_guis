"""
put solarstein in skymode i.e. flippers down, sbb off
"""

import time
import asgard_guis.utils as agu


def main():
    # Open socket connection to DM server
    mds_socket = agu.open_socket_connection("MDS")

    for beam_no in range(1, 5):
        msg = f"moveabs SSF{beam_no} 0.0"
        response = agu.send_and_get_response(mds_socket, msg)
        time.sleep(0.5)

    print("Flippers down")

    msg = "off SBB"
    response = agu.send_and_get_response(mds_socket, msg)
    print("SBB off")

    print("On sky. Camera gain may or may not need changing (and a new dark)")
