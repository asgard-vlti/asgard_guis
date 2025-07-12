SOCKETS = {
    "cam_server": 6667,
    "DM_server": 6666,
    "heimdallr": 6660,
    "baldr": 6662,
    "MDS": 5555,
}


# Load server list from sockets file
def load_servers():
    return [(name, port) for name, port in SOCKETS.items()]
