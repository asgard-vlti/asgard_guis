"""
put solarstein in labmode i.e. flippers up, sbb position and sbb on
"""

import asgard_guis.utils as agu


def main():
    # Open socket connection to DM server
    mds_socket = agu.open_socket_connection("MDS")

    msg = f"asg_setup SSS NAME SBB"  # uses the same information as an eso setup command
    response = agu.send_and_get_response(mds_socket, msg)

    for beam_no in range(1, 5):
        msg = f"moveabs SSF{beam_no} 1.0"
        response = agu.send_and_get_response(mds_socket, msg)

    msg = "on SBB"
    response = agu.send_and_get_response(mds_socket, msg)
