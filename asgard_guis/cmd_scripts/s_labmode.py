"""
put solarstein in labmode i.e. flippers up, sbb position and sbb on
"""

import asgard_guis.utils as agu
import time
import zmq

from asgard_guis.cmd_scripts import camera_mode_settings


def main():
    # Open socket connection to DM server
    mds_socket = agu.open_socket_connection("MDS")
    cam_server_socket = agu.open_socket_connection("cam_server")
    cam_server_socket.setsockopt(zmq.RCVTIMEO, 10000)

    try:
        saved = camera_mode_settings.save_snapshot_if_absent(cam_server_socket)
        camera_mode_settings.send_camera_command(cam_server_socket, "ndmr_mode 1")
        camera_mode_settings.send_camera_command(
            cam_server_socket, f"set_fps {camera_mode_settings.DEFAULT_FPS}"
        )
        camera_mode_settings.send_camera_command(cam_server_socket, "set_gain 3")
    except (OSError, ValueError, RuntimeError, zmq.ZMQError) as exc:
        raise SystemExit(f"Lab Mode aborted before moving mechanisms: {exc}") from exc

    if saved:
        print("Previous camera gain, FPS, and NDMR mode saved")
    else:
        print("Keeping previous camera settings snapshot")
    print("Camera set to GCDS mode, FPS 1000, and gain 3")

    msg = "off SBB"
    response = agu.send_and_get_response(mds_socket, msg)
    time.sleep(2.5)

    msg = "make_dark"
    response = agu.send_and_get_response(cam_server_socket, msg)

    msg = f"asg_setup SSS NAME SBB"  # uses the same information as an eso setup command
    response = agu.send_and_get_response(mds_socket, msg)

    print("Moving solartein to SBB")

    for beam_no in range(1, 5):
        msg = f"moveabs SSF{beam_no} 1.0"
        response = agu.send_and_get_response(mds_socket, msg)
        time.sleep(0.5)

    print("Flippers up")

    msg = "on SBB"
    response = agu.send_and_get_response(mds_socket, msg)

    print("SBB on")
