"""
put solarstein in skymode i.e. entrance flippers down, SBB off
"""

import time
import asgard_guis.utils as agu
import zmq

from asgard_guis.cmd_scripts import camera_mode_settings


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

    cam_server_socket = agu.open_socket_connection("cam_server")
    cam_server_socket.setsockopt(zmq.RCVTIMEO, 10000)
    try:
        fps, gain, nbreads, source = camera_mode_settings.restore_sky_settings(
            cam_server_socket
        )
    except (OSError, ValueError, RuntimeError, zmq.ZMQError) as exc:
        raise SystemExit(f"Sky Mode camera restore failed: {exc}") from exc

    print(
        f"On sky. Camera restored to FPS {fps:g}, gain {gain}, "
        f"NDMR reads {nbreads} ({source})"
    )
