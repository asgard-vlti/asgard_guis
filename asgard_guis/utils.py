import zmq

SOCKETS = {
    "cam_server": 6667,
    "DM_server": 6666,
    "heimdallr": 6660,
    "baldr": 6662,
    "MDS": 5555,
}


def open_socket_connection(name, pc_alias="mimir"):
    """Open a socket connection to the specified server."""
    if name not in SOCKETS:
        raise ValueError(f"Unknown server name: {name}")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{pc_alias}:{SOCKETS[name]}")
    return socket


def send_and_get_response(socket, cmd):
    """Send a command to the server and return the response."""
    socket.send_string(cmd)
    try:
        response = socket.recv().decode("ascii")
    except zmq.Again as e:
        print(f"Error: {e}")
        return None
    return response


# Load server list from sockets file
def load_servers():
    return [(name, port) for name, port in SOCKETS.items()]
